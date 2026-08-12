#
# Copyright (C) 2025 - 2026, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from poses.feature_detector import DescribedKeypoints
from utils import split_into_clusters, quat_to_rotmat, rotmat_to_quat

if TYPE_CHECKING:
    from argparse import Namespace
    from scene.keyframe import Keyframe
    from poses.ba_problem import BAProblem
    from poses.matcher import Matcher
    from scene.hierarchy import HierarchyStructure


class LoopClosureMixin:
    """Mixin providing loop closure, bundle adjustment, and keyframe matching for SceneModel."""

    args: "Namespace"
    keyframes: List["Keyframe"]
    ba_problem: "BAProblem"
    matcher: "Matcher"
    f: torch.Tensor
    centre: torch.Tensor
    valid_Rt_cache: torch.Tensor
    vpr_features: Optional[torch.Tensor]
    are_test: torch.Tensor
    prev_keyframes_odo: Optional[List["Keyframe"]]
    prev_keyframes_loop: Optional[List["Keyframe"]]
    num_prev_keyframes_check: int
    min_loop_size: int
    min_matches_edge: int
    kf_id: torch.Tensor
    log_scaling: torch.Tensor
    gaussian_params: Dict[str, dict]
    hierarchy: "HierarchyStructure"

    def local_ba(self):
        # Serialise the in-place pose writes against a concurrent optimization step.
        with self._scene_mutation_lock():
            # Select keyframes to optimize based on connectivity
            keyframes_ids = self.ba_problem.select_keyframes([self.keyframes[-1].index], shuffle_ratio=0.9)
            keyframes = [self.keyframes[i] for i in keyframes_ids]

            # Run the optimization
            result = self.ba_problem.local_ba(keyframes, self.f, self.centre)
            if result == []:
                return

            for keyframe in keyframes:
                self.valid_Rt_cache[keyframe.index] = False

            self.update_f(keyframes[-1].f)

    def global_ba(self):
        with self._scene_mutation_lock():
            result = self.ba_problem.global_ba(self.keyframes, self.f, self.centre)
            if result == []:
                return
            for keyframe in self.keyframes:
                self.valid_Rt_cache[keyframe.index] = False
            self.update_f(self.keyframes[-1].f)

    def apply_loop_closure_to_gaussians(
            self,
            gaussian_relocation_parameters
        ):
        """
        Applies batched Sim(3) loop-closure corrections to all affected Gaussians.
        Follows the same pattern as landmark relocation in ba_problem.py:1215
        """
        if gaussian_relocation_parameters == {}:
            return
        # Unpack relocation parameters
        kf_id_tensor = gaussian_relocation_parameters['kf_id_tensor']
        all_scales = gaussian_relocation_parameters['all_scales']
        all_dR = gaussian_relocation_parameters['all_dR']
        all_dt = gaussian_relocation_parameters['all_dt']

        # Identify affected gaussians (same pattern as landmarks in ba_problem.py:1215)
        gaussian_kf_ids = self.kf_id
        gaussian_mask = torch.isin(gaussian_kf_ids, kf_id_tensor)

        if gaussian_mask.any():
            affected_gaussian_ids = torch.where(gaussian_mask)[0]
            affected_gaussian_kfs = gaussian_kf_ids[affected_gaussian_ids]

            # Find which index in 'kf_id_tensor' corresponds to each gaussian's parent
            # (N_gauss, 1) == (1, N_kfs)
            kf_indices = (affected_gaussian_kfs.unsqueeze(1) == kf_id_tensor.unsqueeze(0)).int().argmax(dim=1)

            # Gather the Sim3 params applied to the parents
            g_scales = all_scales[kf_indices]       # (N_affected, )
            g_dR = all_dR[kf_indices]               # (N_affected, 3, 3)
            g_dt = all_dt[kf_indices]               # (N_affected, 3)

            # Apply the Sim3 Transform
            # P_new = s * R * P_old + t

            # Positions
            old_xyz = self.gaussian_params["xyz"]["val"][affected_gaussian_ids]  # (N_affected, 3)
            # Rotate: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
            rotated_xyz = torch.bmm(g_dR, old_xyz.unsqueeze(-1)).squeeze(-1)
            # Scale and Translate
            new_xyz = g_scales.unsqueeze(1) * rotated_xyz + g_dt

            # Rotations
            old_Rg = quat_to_rotmat(self.get_rotation(affected_gaussian_ids))   # (N_affected, 3, 3)
            new_Rg = torch.bmm(g_dR, old_Rg)  # (N_affected, 3, 3)
            new_q = rotmat_to_quat(new_Rg)

            # Scaling (log-space!)
            new_scaling = self.log_scaling[affected_gaussian_ids, :] + torch.log(g_scales).unsqueeze(1)

            # Update in place
            with torch.no_grad():
                self.gaussian_params["xyz"]["val"][affected_gaussian_ids] = new_xyz
                self.gaussian_params["rotation"]["val"][affected_gaussian_ids] = new_q
                self.gaussian_params["scaling"]["val"][affected_gaussian_ids] = new_scaling

                # Reset Adam state for the relocated Gaussians.
                for key in ("xyz", "rotation", "scaling"):
                    for state in self.hierarchy.OPTIM_STATE_ATTRS:
                        if state in self.gaussian_params[key]:
                            self.gaussian_params[key][state][affected_gaussian_ids] = 0

    @torch.no_grad()
    def close_loop(self, Rt_loop):
        # Serialise the in-place pose/Gaussian rewrites against a concurrent step.
        with self._scene_mutation_lock():
            relocation_params = self.ba_problem.close_loop(
                self.keyframes, self.prev_keyframes_odo, self.prev_keyframes_loop, Rt_loop,
                self.f, self.centre,
            )

            for keyframe in self.keyframes:
                self.valid_Rt_cache[keyframe.index] = False
            if self.args.lc_move_gaussians:
                self.apply_loop_closure_to_gaussians(relocation_params)

    # --- Keyframe Matching / Retrieval ---

    @torch.no_grad()
    def _vpr_cossim_flat(self, vpr_feature: torch.Tensor) -> torch.Tensor:
        """Per-keyframe flat cosine-similarity vector.

        ``self.vpr_features`` has shape ``[n_kf, S, D]`` and the query
        ``vpr_feature`` has shape ``[N_q, D]``. The full pairwise cossim is
        ``[n_kf, S * N_q]`` -- one entry per (stored-slot, query-slot) pair
        per keyframe.
        """
        cossim = self.vpr_features @ vpr_feature.T   # [n_kf, S, N_q]
        return cossim.reshape(cossim.shape[0], -1)

    @torch.no_grad()
    def get_most_similar_keyframe(self, vpr_feature: torch.Tensor, train_only=True):
        flat = self._vpr_cossim_flat(vpr_feature)
        N_q = vpr_feature.shape[0]
        cossim_best, best_idx = flat.max(dim=1)
        # Best query slot that contributed, used as the descriptor rotation.
        best_rot = best_idx % N_q
        if train_only:
            cossim_best[self.are_test] = -1e9
        index = torch.argmax(cossim_best).item()
        return self.keyframes[index], cossim_best[index], best_rot[index]

    @torch.no_grad()
    def _select_candidate_keyframes(self, vpr_feature, n, desc_kpts):
        # Get top num_prev_keyframes_check matches by VPR similarity as candidates
        n_candidates = min(self.num_prev_keyframes_check, len(self.keyframes))
        flat = self._vpr_cossim_flat(vpr_feature)
        N_q = vpr_feature.shape[0]
        cossim_best, best_idx = flat.max(dim=1)
        best_rot = best_idx % N_q
        _, top_candidates = torch.topk(cossim_best, n_candidates)
        candidate_best_rot = best_rot[top_candidates]

        # Use feature matching to select best from VPR candidates.
        candidate_feats = [self.keyframes[idx.item()].desc_kpts.feats_flat() for idx in top_candidates]
        query_feats = desc_kpts.full_feats_flat()
        if query_feats.shape[0] == 1:
            candidate_best_rot = torch.zeros_like(candidate_best_rot)
        elif candidate_best_rot.numel() > 0 and (
            int(candidate_best_rot.min().item()) < 0
            or int(candidate_best_rot.max().item()) >= query_feats.shape[0]
        ):
            raise RuntimeError(
                "VPR candidate rotations are incompatible with descriptor rotations: "
                f"rot_range=[{int(candidate_best_rot.min().item())}, "
                f"{int(candidate_best_rot.max().item())}], query_feats_shape={tuple(query_feats.shape)}"
            )
        n_matches = self.matcher.batched_evaluate_match(candidate_feats, query_feats, candidate_best_rot)

        return top_candidates, candidate_best_rot, n_matches

    @torch.no_grad()
    def _match_candidates(self, desc_kpts, top_candidates, candidate_best_rot, n_matches, n):
        # Select top n by number of matches
        _, best_by_matches = torch.topk(n_matches, min(n, len(n_matches)))
        all_selected_indices = top_candidates[best_by_matches]
        all_selected_rotations = candidate_best_rot[best_by_matches]

        # Run more robust lighter glue matcher
        selected_indices = []
        for i, index in enumerate(all_selected_indices):
            index = index.item()
            matches = self.matcher(
                desc_kpts,
                self.keyframes[index].desc_kpts,
                rotation_idx=all_selected_rotations[i].item(),
                remove_outliers=True,
                update_kpts_flag="inliers",
                kID=len(self.keyframes),
                kID_other=index,
                use_lightglue=self.args.use_lightglue,
                min_match_count=self.args.min_match_count,
            )
            if matches is not None:
                selected_indices.append(index)
        return selected_indices

    @torch.no_grad()
    def _detect_loop_closure(self, desc_kpts, selected_indices, n_matches, top_candidates, candidate_best_rot, n):
        dists = torch.tensor(self.ba_problem.pose_graph.get_dists(selected_indices, selected_indices))

        # Check if matched keyframes are disconnected. Is an hint there might be a loop to close
        if dists.max() > self.min_loop_size:
            # Additional matches to cover the loop closure
            _, best_by_matches = torch.topk(n_matches, min(n * 2, len(n_matches)))
            all_selected_indices = top_candidates[best_by_matches]
            all_selected_rotations = candidate_best_rot[best_by_matches]
            for i, index in enumerate(all_selected_indices):
                index = index.item()
                if index not in selected_indices:
                    matches = self.matcher(
                        desc_kpts,
                        self.keyframes[index].desc_kpts,
                        rotation_idx=all_selected_rotations[i].item(),
                        remove_outliers=True,
                        update_kpts_flag="inliers",
                        kID=len(self.keyframes),
                        kID_other=index,
                        use_lightglue=self.args.use_lightglue,
                        min_match_count=self.args.min_match_count,
                    )
                    if matches is not None:
                        selected_indices.append(index)

            dists = torch.tensor(self.ba_problem.pose_graph.get_dists(selected_indices, selected_indices))

            clusters_t = split_into_clusters(dists)
            clusters = [[selected_indices[i] for i in cluster] for cluster in clusters_t]

            ## Loop closure detection
            self.min_matches_edge = self.ba_problem.min_matches_edge
            matches_robustness = [[len(desc_kpts.matches[id].kpts) for id in c] for c in clusters]
            total_matches_robustness = [sum(m_r) for m_r in matches_robustness]
            max_matches_robustness = [max(m_r) for m_r in matches_robustness]
            robust_matches = [[len(desc_kpts.matches[id].kpts) >= self.min_matches_edge for id in c] for c in clusters]
            n_robust_pairs = [sum(r_m) for r_m in robust_matches]
            # odo_edges are the cluster with most matches
            odo_edges, loop_edges = [c for c, _ in sorted(zip(clusters, total_matches_robustness), key=lambda x: (x[1], max(x[0])), reverse=True)]
            # Flag the loop edges
            for index in loop_edges:
                desc_kpts.matches[index].is_loop = True
                self.keyframes[index].desc_kpts.matches[len(self.keyframes)].is_loop = True
            # Detect the loop (enough matches and keyframes in each clusters)
            is_loop = (
                min(n_robust_pairs) > n // 2
                and min(max_matches_robustness) > 0.5 * max(max_matches_robustness)
            )
        else:
            odo_edges = selected_indices
            loop_edges = []
            is_loop = False

        return odo_edges, loop_edges, is_loop

    @torch.no_grad()
    def get_most_similar_keyframes(self, vpr_feature: torch.Tensor, n: int, desc_kpts: DescribedKeypoints):
        top_candidates, candidate_best_rot, n_matches = self._select_candidate_keyframes(
            vpr_feature, n, desc_kpts)

        selected_indices = self._match_candidates(
            desc_kpts, top_candidates, candidate_best_rot, n_matches, n)
        if len(selected_indices) == 0:
            return [], [], False

        odo_edges, loop_edges, loop_closure_detected = self._detect_loop_closure(
            desc_kpts, selected_indices, n_matches, top_candidates, candidate_best_rot, n)

        self.prev_keyframes_odo = [self.keyframes[i] for i in odo_edges]
        self.prev_keyframes_loop = [self.keyframes[i] for i in loop_edges]

        return self.prev_keyframes_odo, self.prev_keyframes_loop, loop_closure_detected
