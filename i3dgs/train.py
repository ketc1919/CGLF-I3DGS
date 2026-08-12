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


import os
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from typing import Dict, List

import numpy as np

import torch
from tqdm import tqdm
import random
from threading import Thread

from args import get_args
from scene.dense_extractor import DenseExtractor
from dataloaders.image_dataset import ImageDataset
from poses.feature_detector import DescribedKeypoints, Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer
from poses.triangulator import Triangulator
from scene.keyframe import Keyframe
from scene.vpr_model import VPRModelWrapper
from scene.scene_model import SceneModel
from cglf_export import export_cglf_scene, preflight_cglf_export
from gaussianviewer import GaussianViewer
from graphdecoviewer.types import ViewerMode

from utils import (
    increment_runtime,
    get_time,
)

class ReconstructionTask:
    def __init__(self, args):
        self.args = args

        self.dataset = ImageDataset(self.args)
        if self.args.cglf_scene_path:
            preflight_cglf_export(
                output_path=self.args.cglf_scene_path,
                input_image_paths=self.dataset.image_paths,
            )
        self.model_path = self.args.model_path
        self.height, self.width = self.dataset.get_image_size()
        
        self.viewer = None
        self.viewer_thd = None
        self.scene_model = None

    def stop_viewer(self):
        viewer = self.viewer
        viewer_thd = self.viewer_thd
        self.viewer = None
        self.viewer_thd = None

        if viewer is None:
            return
        viewer.end()
        if viewer_thd is not None and viewer_thd.is_alive():
            viewer_thd.join(timeout=5.0)

    def run(self):
        self.initialize_modules()
        if not self.run_reconstruction():
            return
        # Keep viewer alive
        if (
            self.args.viewer_mode != "none"
            and self.viewer is not None
            and getattr(self.viewer, "keep_alive", self.args.keep_alive)
        ):
            self.viewer.throttling = False # Disable throttling when done training
            while self.viewer.running and self.viewer.keep_alive:
                time.sleep(1)
    
    def initialize_modules(self):
        print("Initializing modules and running just in time compilation, may take a while...")
        s = (self.width**2 + self.height**2)**0.5
        max_error = self.args.match_max_error * s
        max_pnp_error = self.args.pnp_max_error * s
        self.ba_outlier_threshold = self.args.ba_outlier_threshold * s
        self.min_displacement = max(25, self.args.min_displacement * s)

        def build_matcher():
            n_kpts_pool = self.args.num_kpts
            return Matcher(self.args.fundmat_samples, max_error, n_kpts_pool, (self.width, self.height),
                           lg_n_layers=self.args.lg_n_layers, use_rotated_descriptors=self.args.use_rotated_descriptors,
                           min_cossim=self.args.match_min_cossim, max_candidates=self.args.num_prev_keyframes_check,
                           enable_lightglue=self.args.use_lightglue)

        def build_triangulator():
            return Triangulator(
                self.args.num_kpts, self.args.num_prev_keyframes_miniba_incr, self.args.triangulator_max_error * s,
                min_dis=self.args.triangulator_min_dis * s,
            )

        def build_dense_extractor():
            return DenseExtractor(self.width, self.height)

        def build_vpr_model():
            return VPRModelWrapper(self.width, self.height, use_rotated_descriptors=self.args.use_rotated_descriptors)

        def build_detector():
            return Detector(self.args.num_kpts, self.width, self.height,
                            use_rotated_descriptors=self.args.use_rotated_descriptors, batch=1)

        self.matcher = build_matcher()
        self.triangulator = build_triangulator()
        self.dense_extractor = build_dense_extractor()
        self.vpr_model = build_vpr_model()
        self.pose_initializer = PoseInitializer(
            self.width,
            self.height,
            self.triangulator,
            self.matcher,
            max_pnp_error,
            self.ba_outlier_threshold,
            self.args,
        )
        self.detector = build_detector()

    def setup_viewer(self):
        if self.args.viewer_mode in ["server", "local"]:
            viewer_mode = ViewerMode.SERVER if self.args.viewer_mode == "server" else ViewerMode.LOCAL
            self.viewer = GaussianViewer.from_scene_model(self.scene_model, viewer_mode)
            self.viewer.keep_alive = self.args.keep_alive
            self.viewer_thd = Thread(target=self.viewer.run, args=(self.args.ip, self.args.port), daemon=True)
            self.viewer_thd.start()
            self.viewer.throttling = True  # Enable throttling when training

    def update_pbar(self, pbar, metrics, runtimes, shelf_dicts, kf_stats, frameID):
        ## Intermediate evaluation
        if (
            len(self.scene_model.keyframes) % self.args.test_frequency == 0
            and self.args.test_frequency > 0
            and (self.args.test_hold > 0 or self.args.eval_poses)
        ):
            metrics_t = self.scene_model.evaluate(self.args.eval_poses)
            for key, value in metrics_t.items():
                metrics[key] = value

        ## Save intermediate model
        if (
            frameID % self.args.save_every == 0
            and self.args.save_every > 0
            and frameID > 0
        ):
            self.scene_model.save(
                os.path.join(self.args.model_path, "progress", f"{frameID:05d}")
            )

        bar_postfix = []
        for key, value in metrics.items():
            bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
        if self.args.display_runtimes:
            for key, value in runtimes.items():
                if value[1] > 0:
                    bar_postfix += [
                        f"\033[35m{key}:{1000 * value[0] / value[1]:.1f}\033[0m"
                    ]
        if len(shelf_dicts) > 0:
            bar_postfix += [f"\033[36mShelf:{len(shelf_dicts)}\033[0m"]
            # Count rejection reasons from shelf_dicts
            reason_counts = {}
            for sd in shelf_dicts:
                reason = sd.get("reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in reason_counts.items():
                bar_postfix += [f"\033[35m{reason}:{count}\033[0m"]
        for key, value in kf_stats.items():
            if value > 0:
                bar_postfix += [f"\033[94m{key}:{value}\033[0m"]
        bar_postfix += [
            f"\033[36mf:{self.scene_model.f.item():.1f}",
            f"\033[36mnKF:{len(self.scene_model.keyframes)}\033[0m",
            f"\033[36mnGS:{self.scene_model.n_active_gaussians}\033[0m",
        ]
        pbar.set_postfix_str(",".join(bar_postfix), refresh=True)
    
    def incremental_recons(
        self,
        image: torch.Tensor,
        info: dict,
        desc_kpts: DescribedKeypoints,
        vpr_feature: torch.Tensor,
        kf_stats: Dict[str, int],
        runtimes: Dict[str, List[float]],
        f,
        index,
        force_add: bool = False,
    ):
        n_keyframes = len(self.scene_model.keyframes)
        start_time = get_time()
        prev_keyframes_odo, prev_keyframes_loop, close_loop = self.scene_model.get_most_similar_keyframes(
            vpr_feature, self.args.num_prev_keyframes_miniba_incr, desc_kpts
        )
        increment_runtime(runtimes["VPR"], start_time)
        start_time = get_time()
        need_in_between = info.get("need_in_between", False)
        Rt, reason = self.pose_initializer.initialize_incremental(
            prev_keyframes_odo, desc_kpts, n_keyframes, force_add, info["is_test"] or need_in_between,
        )

        if Rt is None:
            return False, reason

        if close_loop:
            # Get initial loop transform
            Rt_loop, _ = self.pose_initializer.initialize_incremental(
                prev_keyframes_loop, desc_kpts, n_keyframes, True, True,
            )
            if self.args.use_colmap_poses:
                Rt_loop = info["Rt"]
            if Rt_loop is None:
                close_loop = False
                
        increment_runtime(runtimes["BAI"], start_time)
        start_time = get_time()
        if self.args.use_colmap_poses:
            Rt = info["Rt"]
        keyframe = Keyframe(
            image,
            info,
            desc_kpts,
            Rt,
            n_keyframes,
            f,
            self.dense_extractor,
            self.triangulator,
            self.args,
        )
        keyframe.update_3dpts(self.scene_model.keyframes)
        increment_runtime(runtimes["tri"], start_time)
        if keyframe.desc_kpts.has_pt3d.sum() < self.args.min_3d_points and not force_add:
            for kf in self.scene_model.keyframes:
                kf.desc_kpts.matches.pop(index, None)
            return False, "low_nTri"
        else:
            start_time = get_time()
            self.scene_model.add_keyframe(keyframe, vpr_feature)
            ## Add 3D points from the loop closure
            if close_loop:
                keyframe_loop = Keyframe(
                    image,
                    info,
                    desc_kpts,
                    Rt_loop,
                    n_keyframes,
                    f,
                    self.dense_extractor,
                    self.triangulator,
                    self.args,
                )
                keyframe_loop.update_3dpts(self.scene_model.keyframes)
                self.scene_model.ba_problem.append(self.scene_model.keyframes, keyframe_loop)
            increment_runtime(runtimes["Add"], start_time)
            start_time = get_time()
            self.scene_model.ba_problem.append(self.scene_model.keyframes, keyframe)

            increment_runtime(runtimes["BBAL"], start_time)
            # Loop closure
            if close_loop:
                start_time = get_time()
                kf_stats["nLoop"] += 1
                kf_stats["lastLoop"] = n_keyframes
                self.scene_model.close_loop(Rt_loop)
                increment_runtime(runtimes["loop"], start_time)
            else:
                start_time = get_time()
                self.scene_model.local_ba()
                self.pose_initializer.f = self.scene_model.f
                for _ in range(self.args.num_global_ba_rounds):
                    self.scene_model.global_ba()
                increment_runtime(runtimes["BAL"], start_time)

            # Gaussian initialization
            start_time = get_time()
            self.scene_model.add_new_gaussians()
            increment_runtime(runtimes["Init"], start_time)

            start_time = get_time()
            self.scene_model.optimization_loop(self.args.num_iterations)
            increment_runtime(runtimes["Opt"], start_time)


            if n_keyframes > 2 and n_keyframes % self.args.hierarchy_update_freq == 0 and self.args.hierarchy:
                start_time = get_time()
                self.scene_model.check_and_create_hierarchy()
                increment_runtime(runtimes["Hier"], start_time)
        
        return True, None
    
    def try_unshelve(self, shelf_dicts, kf_stats, runtimes, f, force_add_test=False):
        cossim_max, max_index = self.get_shelf_best_match(shelf_dicts)
        shelf_dict = shelf_dicts.pop(max_index)
        added_shelf_keyframe, reason = self.incremental_recons(
            shelf_dict["image"].cuda(),
            shelf_dict["info"],
            shelf_dict["kpts"],
            shelf_dict["vpr_feature"],
            kf_stats,
            runtimes,
            f,
            len(self.scene_model.keyframes),
            force_add=force_add_test and shelf_dict["info"]["is_test"], # Force add test keyframes after first pass
        )
        if not added_shelf_keyframe:
            # No need to shelf if the frame has low parallax
            if reason == "no_parallax":
                kf_stats["no_parallax"] += 1
                return None
            # Shelf if pose estimation failure
            else:
                shelf_dict["reason"] = reason
                return shelf_dict
        else:
            return None
    
    
    def get_shelf_best_match(self, shelf_dicts):
        # shelved_vpr_features: [N_shelf, N_q, D] where N_q is the VPR query-slot
        # count (1 upright descriptor or 4 rotated descriptors).
        shelved_vpr_features = torch.stack(
            [shelf_dict["vpr_feature"] for shelf_dict in shelf_dicts], dim=0
        )
        N_shelved = shelved_vpr_features.shape[0]
        N_q = shelved_vpr_features.shape[1]
        D = shelved_vpr_features.shape[-1]
        shelved_flat = shelved_vpr_features.view(-1, D)                # [N_shelf*N_q, D]
        # scene_model.vpr_features: [n_kf, S, D]
        cossim = self.scene_model.vpr_features @ shelved_flat.T        # [n_kf, S, N_shelf*N_q]
        S = self.scene_model.vpr_features.shape[1]
        cossim = (
            cossim
            .view(-1, S, N_shelved, N_q)
            .amax(dim=3)   # collapse query slots
            .amax(dim=1)   # collapse stored slots
        )                                                              # [n_kf, N_shelf]
        return cossim.max(dim=0).values.max(dim=0)

    def try_bootstrap_from_shelf(self, bootstrap_shelf, force=False):
        """
        Try to find a set of num_keyframes_miniba_bootstrap frames from the bootstrap shelf
        that have good pairwise VPR similarity and enough 2D matches.
        If force=True, try with however many frames are available (at least 2).
        Returns (selected_dicts, selected_desc_kpts) if successful, else None.
        """
        n = len(bootstrap_shelf)
        if n < 2 or (not force and n < self.args.num_keyframes_miniba_bootstrap):
            return None

        # Compute pairwise VPR cosine similarity.
        # Each shelf entry has VPR features shaped [N_q, D] -- N_q is 1
        # or 4 when rotated descriptors are enabled.
        feat_dim = bootstrap_shelf[0]["vpr_feature"].shape[-1]
        flat_list = [d["vpr_feature"].view(-1, feat_dim) for d in bootstrap_shelf]
        flat_stack = torch.stack(flat_list, dim=0)                                  # [N, N_q, D]
        N_q = flat_stack.shape[1]
        flat_norm = flat_stack / (flat_stack.norm(dim=-1, keepdim=True) + 1e-8)

        # Symmetric sim_matrix: best of all N_q x N_q slot combinations between
        # every pair (i, j).
        flat_2d = flat_norm.view(n * N_q, feat_dim)
        sym_sim_matrix = (
            (flat_2d @ flat_2d.T)
            .view(n, N_q, n, N_q)
            .amax(dim=3).amax(dim=1)
        )
        sym_sim_matrix.fill_diagonal_(-float('inf'))

        # Asymmetric rotation index: canonical of ref (slot 0) vs all N_q slots of
        # query.
        canonical = flat_norm[:, 0, :]                                              # [N, D]
        rot_idx_matrix = (canonical @ flat_2d.T).view(n, n, N_q).max(dim=2).indices  # [N, N]

        # Find best matching pair using symmetric similarity
        best_pair_idx = sym_sim_matrix.argmax()
        i, j = best_pair_idx.item() // n, best_pair_idx.item() % n

        # Check 2D matches for the best pair; rotation_idx is for j (the query)
        pair_matches = self.matcher(bootstrap_shelf[j]["kpts"], bootstrap_shelf[i]["kpts"], rot_idx_matrix[i, j].item())
        if len(pair_matches.kpts) < self.args.min_bootstrap_matches_count:
            return None

        selected = [i, j]
        skipped = set()

        # Iteratively add the best matching remaining frame
        while len(selected) < self.args.num_keyframes_miniba_bootstrap:
            remaining = [k for k in range(n) if k not in selected and k not in skipped]
            if not remaining:
                break

            # Find the remaining frame (query) with the best VPR sim to any selected frame (ref)
            best_k = max(remaining, key=lambda k: max(sym_sim_matrix[s, k].item() for s in selected))

            # Check 2D matches; best_k is the query, its best-matching selected frame is the ref
            best_match_in_selected = max(selected, key=lambda s: sym_sim_matrix[s, best_k].item())
            candidate_matches = self.matcher(
                bootstrap_shelf[best_k]["kpts"],
                bootstrap_shelf[best_match_in_selected]["kpts"],
                rot_idx_matrix[best_match_in_selected, best_k].item(),
                remove_outliers=True,
            )
            
            if not force and len(candidate_matches.kpts) < self.args.min_init_matches_count:
                # Discard current cluster & candidate, and mask them out in the similarity matrix
                skipped.update(selected + [best_k])
                skip_idx = list(skipped)
                sym_sim_matrix[skip_idx, :] = sym_sim_matrix[:, skip_idx] = -float('inf')
                
                selected = []
                
                # Find a new valid pair
                while not selected:
                    idx = sym_sim_matrix.argmax().item()
                    if sym_sim_matrix.view(-1)[idx] == -float('inf'):
                        return None  # Out of viable pairs
                        
                    i, j = idx // n, idx % n
                    pm = self.matcher(bootstrap_shelf[j]["kpts"], bootstrap_shelf[i]["kpts"], rot_idx_matrix[i, j].item())
                    
                    if len(pm.kpts) >= self.args.min_init_matches_count:
                        selected = [i, j]
                    else:
                        # Mask this bad pair and keep searching
                        sym_sim_matrix[[i, j], :] = sym_sim_matrix[:, [i, j]] = -float('inf')
                        skipped.update([i, j])
                continue

            dist = torch.norm(candidate_matches.kpts - candidate_matches.kpts_other, dim=-1)
            if dist.median() > self.min_displacement:
                selected.append(best_k)
            else:
                skipped.add(best_k)
                
        if not force and len(selected) < self.args.num_keyframes_miniba_bootstrap:
            return None

        selected_dicts = [bootstrap_shelf[i] for i in selected]
        selected_desc_kpts = [d["kpts"] for d in selected_dicts]
        return selected_dicts, selected_desc_kpts
    
    def _init_reconstruction_state(self):
        if (
            self.scene_model is None
            or getattr(self.scene_model, "inference_mode", False)
            or self.scene_model.width != self.width
            or self.scene_model.height != self.height
        ):
            self.scene_model = SceneModel(self.width, self.height, self.args, self.matcher)
        else:
            self.scene_model.matcher = self.matcher
            self.scene_model.reset()

        # Initialize the viewer
        self.setup_viewer()

        self._recon_rotation_idx = 0
        self._recon_last_max_valid_cossim = 1
        self._recon_shelf_dicts = []
        self._recon_bootstrap_keyframe_dicts = []
        self._recon_f = self.scene_model.f  # Updated after bootstrap; initialized to scene_model default
        self._recon_ordered_bootstrap = True

        # Dict of runtimes for each step
        runtimes = ["Load", "BAB", "VPR", "BAI", "Add", "BBAL", "BAL", "tri", "loop", "Init", "Opt", "Hier", "anc"]
        self._recon_runtimes = {key: [0, 0] for key in runtimes}
        self._recon_metrics = {}

        # Stats for pose estimation (rejection reasons are stored in shelf_dicts)
        self._recon_kf_stats = {"lastLoop": 0, "nLoop": 0, "no_parallax": 0}

    def _process_bootstrap_frame(self, image, info, desc_kpts, vpr_feature, frameID, pbar):
        bootstrap_keyframe_dicts = self._recon_bootstrap_keyframe_dicts
        ordered_bootstrap = self._recon_ordered_bootstrap
        shelf_dicts = self._recon_shelf_dicts
        runtimes = self._recon_runtimes
        metrics = self._recon_metrics
        kf_stats = self._recon_kf_stats

        if ordered_bootstrap and len(bootstrap_keyframe_dicts) > 0:
            prev_desc_kpts = bootstrap_keyframe_dicts[-1]["kpts"]
            # Match features between the previous and current frame
            curr_prev_matches = self.matcher(desc_kpts, prev_desc_kpts, self._recon_rotation_idx)
            # Determine if we should add a keyframe based on the matches
            dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
            enough_matches = len(curr_prev_matches.kpts) >= self.args.min_bootstrap_matches_count

            if not enough_matches or dist.median() > self.min_displacement or info["is_test"]:
                bootstrap_keyframe_dicts.append(
                    {"image": image.cpu(), "info": info, "vpr_feature": vpr_feature, "kpts": desc_kpts}
                )
            ordered_bootstrap = ordered_bootstrap and enough_matches
            self._recon_ordered_bootstrap = ordered_bootstrap
        else:
            bootstrap_keyframe_dicts.append(
                {"image": image.cpu(), "info": info, "vpr_feature": vpr_feature, "kpts": desc_kpts}
            )
        is_last_frame = frameID == len(self.dataset) - 1
        if len(bootstrap_keyframe_dicts) >= self.args.num_keyframes_miniba_bootstrap or is_last_frame:
            if ordered_bootstrap:
                selected_desc_kpts = [d["kpts"] for d in bootstrap_keyframe_dicts]
                result = bootstrap_keyframe_dicts, selected_desc_kpts
            else:
                result = self.try_bootstrap_from_shelf(bootstrap_keyframe_dicts, force=is_last_frame)
            if result is not None:
                selected_keyframe_dicts, selected_desc_kpts = result
                selected_ids = {id(d) for d in selected_keyframe_dicts}
                remaining_dicts = [d for d in bootstrap_keyframe_dicts if id(d) not in selected_ids]
                start_time = get_time()
                Rts, f, _ = self.pose_initializer.initialize_bootstrap(
                    selected_desc_kpts, selected_keyframe_dicts, ordered_bootstrap)
                increment_runtime(runtimes["BAB"], start_time)
                if Rts is None:
                    # Disconnected match graph after outlier removal; keep all frames in bootstrap shelf
                    self.update_pbar(pbar, metrics, runtimes, shelf_dicts, kf_stats, frameID)
                    return True
                self._recon_f = f
                for index, (keyframe_dict, kf_desc_kpts, Rt) in enumerate(
                    zip(selected_keyframe_dicts, selected_desc_kpts, Rts)
                ):
                    start_time = get_time()
                    if self.args.use_colmap_poses:
                        Rt = keyframe_dict["info"]["Rt"]
                        f = keyframe_dict["info"]["focal"]
                    keyframe = Keyframe(
                        keyframe_dict["image"].cuda(),
                        keyframe_dict["info"],
                        kf_desc_kpts,
                        Rt,
                        index,
                        f,
                        self.dense_extractor,
                        self.triangulator,
                        self.args,
                    )
                    self.scene_model.add_keyframe(keyframe, keyframe_dict["vpr_feature"], f)
                    increment_runtime(runtimes["Add"], start_time)
                self._recon_f = f
                self._recon_bootstrap_keyframe_dicts = None
                if self.args.viewer_mode != "none":
                    self.viewer.reset_intrinsics("point_view")
                for index in range(len(selected_keyframe_dicts)):
                    start_time = get_time()
                    self.scene_model.keyframes[index].update_3dpts(self.scene_model.keyframes)
                    self.scene_model.ba_problem.append(self.scene_model.keyframes, self.scene_model.keyframes[index])
                    self.scene_model.add_new_gaussians(index)
                    increment_runtime(runtimes["Init"], start_time)
                start_time = get_time()
                # Run initial optimization on the bootstrap keyframes
                self.scene_model.optimization_loop(self.args.num_iterations)
                increment_runtime(runtimes["Opt"], start_time)
                # Move non-selected frames to the regular shelf for the incremental phase
                for d in remaining_dicts:
                    d["reason"] = "bootstrap_shelf"
                shelf_dicts += remaining_dicts
        self.update_pbar(pbar, metrics, runtimes, shelf_dicts, kf_stats, frameID)
        if frameID != len(self.dataset) - 1:
            return True
        return False

    @torch.no_grad()
    def _keypoint_valid_mask(self, image, dataset_mask):
        """Prepare the dataset mask for keypoint invalidation.

        Returns a ``[1, H, W]`` bool tensor (True = keep) or ``None`` when
        there is nothing to mask. ``image`` is the batched capture at detection
        time and provides the device for the returned mask.
        """
        valid = dataset_mask.to(image.device).bool() if dataset_mask is not None else None
        if valid is not None and valid.dim() == 4:
            assert valid.shape[0] == 1, f"Expected one mask batch, got {tuple(valid.shape)}"
            valid = valid[0]
        return valid

    @torch.no_grad()
    def _is_stationary_redundant(self, desc_kpts, info):
        """Reject near-stationary frames by matching against the last keyframe,
        skipping VPR (whose feature would be discarded anyway). The match is
        cached in self._recon_last_kf_match for reuse by the incremental path.
        Test and between-test frames are never rejected here.
        """
        if info["is_test"]:
            return False
        need_in_between = (
            info.get("next_is_test") and self.scene_model.keyframes[-1].is_test
        )
        if need_in_between:
            return False
        last_kf = self.scene_model.keyframes[-1]
        # rotation_idx 0: stationary frames have ~no in-plane rotation; otherwise
        # matches collapse and the frame falls through to the full VPR path.
        matches = self.matcher(desc_kpts, last_kf.desc_kpts, 0)
        self._recon_last_kf_match = (last_kf, matches)
        if len(matches.kpts) < self.args.min_init_matches_count:
            return False
        dist = torch.norm(matches.kpts - matches.kpts_other, dim=-1)
        return dist.median() <= self.min_displacement

    def _process_incremental_frame(self, image, info, desc_kpts, vpr_feature, frameID, pbar):
        shelf_dicts = self._recon_shelf_dicts
        runtimes = self._recon_runtimes
        metrics = self._recon_metrics
        kf_stats = self._recon_kf_stats
        f = self._recon_f

        ## Incremental phase: match with most similar keyframe in scene
        prev_kf, max_cossim, rotation_idx = self.scene_model.get_most_similar_keyframe(vpr_feature)
        self._recon_rotation_idx = rotation_idx
        prev_desc_kpts = prev_kf.desc_kpts
        # Reuse the stationary-check match if VPR picked the same last keyframe.
        cached = self._recon_last_kf_match
        if cached is not None and cached[0] is prev_kf and int(rotation_idx) == 0:
            curr_prev_matches = cached[1]
        else:
            curr_prev_matches = self.matcher(desc_kpts, prev_desc_kpts, rotation_idx)
        # Determine if we should add a keyframe based on the matches
        dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
        enough_matches = len(curr_prev_matches.kpts) >= self.args.min_init_matches_count
        enough_disp = dist.median() > self.min_displacement # Check for redundancy
        info["need_in_between"] = ( # Make sure there's always a training keyframe between two tests
            info.get("next_is_test") 
            and self.scene_model.keyframes[-1].is_test
        )
        should_add_keyframe = (
            (enough_disp or info["is_test"] or info["need_in_between"])
            and enough_matches
        )

        ## Try to use the best shelved image if their VPR matching is on par with last added frame
        if len(shelf_dicts) > 0 and self._recon_bootstrap_keyframe_dicts is None:
            shelf_cossim_max, max_index = self.get_shelf_best_match(shelf_dicts)
            if shelf_cossim_max > self._recon_last_max_valid_cossim:
                shelf_dict = self.try_unshelve(shelf_dicts, kf_stats, runtimes, f)
                # Reshelve if failed (try_unshelve pops the entry from shelf_dicts)
                if shelf_dict is not None:
                    shelf_dicts.append(shelf_dict)

        ## Incremental reconstruction of the current frame
        if should_add_keyframe:
            should_add_keyframe, reason = self.incremental_recons(
                image,
                info,
                desc_kpts,
                vpr_feature,
                kf_stats,
                runtimes,
                f,
                len(self.scene_model.keyframes),
            )
            if not should_add_keyframe:
                # No need to shelf if the frame has low parallax
                if reason == "no_parallax":
                    kf_stats["no_parallax"] += 1
                # Shelf if pose estimation failure
                else:
                    shelf_dicts.append(
                        {"image": image.cpu(), "info": info, "kpts": desc_kpts, "vpr_feature": vpr_feature,
                        "reason": reason})
            else:
                self._recon_last_max_valid_cossim = max_cossim
        elif not enough_matches:
            # If couldn't match, store in shelf for potential later use
            shelf_dicts.append(
                {"image": image.cpu(), "info": info, "kpts": desc_kpts, "vpr_feature": vpr_feature, "reason": "no_matches"})

        ## After the incremental reconstruction, try to use all the remaining shelved
        ## frames at the end of the sequence
        if frameID == len(self.dataset) - 1:
            self._flush_end_of_sequence_shelf(pbar, frameID)

        ## Display optimization progress and metrics
        self.update_pbar(pbar, metrics, runtimes, shelf_dicts, kf_stats, frameID)

    def _flush_end_of_sequence_shelf(self, pbar, frameID):
        """Register every remaining shelved frame at the end of the sequence.

        Runs several passes: a frame that still fails to register is kept for the
        next pass, and test frames are force-added after the first pass.
        """
        shelf_dicts = self._recon_shelf_dicts
        runtimes = self._recon_runtimes
        metrics = self._recon_metrics
        kf_stats = self._recon_kf_stats
        f = self._recon_f
        passes = 3
        for pass_id in range(passes):
            second_shelf_dicts = []
            # Try to add shelved frames at the end of the sequence
            while len(shelf_dicts) > 0:
                shelf_dict = self.try_unshelve(shelf_dicts, kf_stats, runtimes, f, pass_id > 0)
                if shelf_dict is not None:
                    second_shelf_dicts.append(shelf_dict)
                self.update_pbar(pbar, metrics, runtimes, shelf_dicts + second_shelf_dicts, kf_stats, frameID)
            shelf_dicts[:] = second_shelf_dicts

    def _finalize_reconstruction(self, reconstruction_time):
        self.scene_model.join_optimization_thread()

        # Set to inference mode so that the model can be rendered properly
        self.scene_model.enable_inference_mode()

        # Save the model and metrics
        print("Saving the reconstruction to:", self.model_path)
        metrics = self.scene_model.save(self.model_path, reconstruction_time, len(self.dataset))
        if self.args.cglf_scene_path:
            export_cglf_scene(
                scene_model=self.scene_model,
                reconstruction_path=self.model_path,
                input_image_paths=self.dataset.image_paths,
                output_path=self.args.cglf_scene_path,
                min_observations=2,
            )
        print(
            ", ".join(
                f"{metric}: {value:.3f}"
                if isinstance(value, float)
                else f"{metric}: {value}"
                for metric, value in metrics.items()
            )
            + f", Gaussians: {self.scene_model.n_active_gaussians}"
        )

        registered_test = sum(kf.info["is_test"] for kf in self.scene_model.keyframes)
        remaining_test = [sd for sd in self._recon_shelf_dicts if sd["info"]["is_test"]]
        total_test = sum(info["is_test"] for info in self.dataset.infos.values())
        if len(remaining_test) > 0 or registered_test + len(remaining_test) != total_test:
            print(
                f"Warning: {registered_test}/{total_test} test frames registered, "
                f"{len(remaining_test)} still shelved."
            )
            if len(remaining_test) > 0:
                print("Frames that failed to register (name: reason):")
                for sd in remaining_test:
                    print(f"  {sd['info']['name']}: {sd.get('reason', 'unknown')}")
            # If some test frames are neither registered nor shelved, they were dropped
            # somewhere in the pipeline without being tracked -- flag it explicitly.
            unaccounted = total_test - registered_test - len(remaining_test)
            if unaccounted != 0:
                print(f"Warning: {unaccounted} test frames are unaccounted for (dropped without being shelved).")

    def _run_finetuning(self, reconstruction_time):
        if len(self.args.save_at_finetune_epoch) > 0:
            finetune_epochs = max(self.args.save_at_finetune_epoch)
            torch.cuda.empty_cache()
            self.scene_model.inference_mode = False
            pbar = tqdm(range(0, finetune_epochs), desc="Fine tuning")
            for epoch in pbar:
                # Run one epoch of fine-tuning
                epoch_start_time = get_time()
                self.scene_model.finetune_epoch()
                epoch_time = get_time() - epoch_start_time
                reconstruction_time += epoch_time
                # Save the model and metrics
                if epoch + 1 in self.args.save_at_finetune_epoch:
                    torch.cuda.empty_cache()
                    self.scene_model.inference_mode = True
                    metrics = self.scene_model.save(
                        os.path.join(self.model_path, str(epoch + 1)), reconstruction_time
                    )
                    bar_postfix = []
                    for key, value in metrics.items():
                        bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                    pbar.set_postfix_str(",".join(bar_postfix))
                    self.scene_model.inference_mode = False
                    torch.cuda.empty_cache()

            # Set to inference mode so that the model can be rendered properly
            self.scene_model.inference_mode = True

    def run_reconstruction(self):
        self._init_reconstruction_state()

        ## Scene reconstruction
        print(f"Starting reconstruction for {self.args.source_path}")
        pbar = tqdm(range(0, len(self.dataset)))
        reconstruction_start_time = get_time()
        for frameID in pbar:
            start_time = get_time()

            ## Load, extract features and keypoints, and match
            image, info = self.dataset.getnext()
            desc_kpts = self.detector(image)
            # Invalidate keypoints that fall in masked-out regions (zeroed descriptors
            # never match).
            valid_mask = self._keypoint_valid_mask(image, info.get("mask"))
            if valid_mask is not None:
                desc_kpts.invalidate_masked(valid_mask.to(image.device))

            # Reset each frame so a stale match is never reused.
            self._recon_last_kf_match = None

            # Skip near-stationary frames before VPR. Incremental phase only, and
            # never the last frame so the end-of-sequence shelf flush still runs.
            if (
                self._recon_bootstrap_keyframe_dicts is None
                and len(self.scene_model.keyframes) > 0
                and frameID != len(self.dataset) - 1
                and self._is_stationary_redundant(desc_kpts, info)
            ):
                increment_runtime(self._recon_runtimes["Load"], start_time)
                self.update_pbar(
                    pbar,
                    self._recon_metrics,
                    self._recon_runtimes,
                    self._recon_shelf_dicts,
                    self._recon_kf_stats,
                    frameID,
                )
                continue

            vpr_feature = self.vpr_model(image)
            increment_runtime(self._recon_runtimes["Load"], start_time)

            ## Bootstrap phase: shelve all frames and try to find a good set to bootstrap from
            if self._recon_bootstrap_keyframe_dicts is not None:
                if self._process_bootstrap_frame(image, info, desc_kpts, vpr_feature, frameID, pbar):
                    continue
                if self._recon_bootstrap_keyframe_dicts is None:
                    self._flush_end_of_sequence_shelf(pbar, frameID)
                continue

            ## Incremental phase
            if desc_kpts.full_feats is not None:
                self._process_incremental_frame(image, info, desc_kpts, vpr_feature, frameID, pbar)

        reconstruction_time = get_time() - reconstruction_start_time
        self._finalize_reconstruction(reconstruction_time)
        self._run_finetuning(reconstruction_time)
    
if __name__ == "__main__":
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = get_args()
    # This ensures deterministic poses but slows down
    if args.deterministic_poses:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Gaussian backward is not deterministic.
        print("deterministic_poses is set, preventing pose refinement.")
        args.lr_poses = 0.0

    reconstructionTask = ReconstructionTask(args)
    reconstructionTask.run()
