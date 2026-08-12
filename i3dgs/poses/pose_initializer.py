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

import logging
import torch
import math
from argparse import Namespace
from poses.feature_detector import DescribedKeypoints
from poses.matcher import Matcher
from poses.mini_ba import MiniBA
from poses.triangulator import Triangulator
from utils import fov2focal, depth2points, sample, pts2px
from scene.keyframe import Keyframe
from poses.ransac import RANSACEstimator, EstimatorType
from poses.depth_anything3 import DepthAnything3

class PoseInitializer:
    """Fast pose initializer using MiniBA and the previous frames."""

    def __init__(
        self,
        width: int,
        height: int,
        triangulator: Triangulator,
        matcher: Matcher,
        max_pnp_error: float,
        ba_outlier_threshold: float,
        args: Namespace,
    ):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher

        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device="cuda")
        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers
        self.min_match_count = args.min_match_count
        self.baseline_thresh = args.baseline_thresh
        self.use_lightglue = args.use_lightglue
        self.scale_depth_target = args.scale_depth_target
        self.scale_depth_quantile = args.scale_depth_quantile

        # Initialize the focal length
        if args.init_focal > 0:
            self.f_init = args.init_focal
        elif args.init_fov > 0:
            self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.0

        self.num_prev_keyframes_window = args.num_prev_keyframes_miniba_incr
        self.n_kpts_pool = self.num_kpts
        self.miniba_bootstrap = MiniBA(
            args.num_keyframes_miniba_bootstrap,
            args.num_pts_miniba_bootstrap,
            args.num_keyframes_miniba_bootstrap,
            optimize_focal=not args.fix_focal,
            optimize_3Dpts=True,
            make_cuda_graph=True,
            outlier_mad_scale=5.0,
            outlier_threshold=20*max_pnp_error,
            iters=args.iters_miniba_bootstrap,
        )
        self.miniBA_incr = MiniBA(
            1,
            args.num_pts_miniba_incr,
            1,
            outlier_threshold=ba_outlier_threshold,
            optimize_focal=False,
            optimize_3Dpts=False,
            make_cuda_graph=True,
            iters=args.iters_miniba_incr,
            huber_delta=args.huber_delta_ratio * ba_outlier_threshold if args.huber_delta_ratio > 0.0 else 0.0,
        )

        self.PnPRANSAC = RANSACEstimator(
            args.pnpransac_samples, self.max_pnp_error, EstimatorType.P3P,
        )

        self.da3 = DepthAnything3(self.width, self.height, args.num_keyframes_miniba_bootstrap, args.da3_model_dir)

        # Pre-allocate padding buffers for initialize_incremental to reduce fragmentation
        self.xyz_padding = torch.zeros(self.num_pts_miniba_incr, 3, device="cuda")
        self.uvs_padding = -torch.ones(self.num_pts_miniba_incr, 2, device="cuda")

        # Pre-allocate build_problem buffers to avoid per-iteration GPU allocation.
        # Buffers index the full keypoint pool.
        n_cams_bs = args.num_keyframes_miniba_bootstrap
        npts_per_cam_bs = args.num_pts_miniba_bootstrap // n_cams_bs
        self._bp_idx_occ    = torch.zeros(self.n_kpts_pool, device="cuda", dtype=torch.int32)
        self._bp_sel_mask   = torch.zeros(self.n_kpts_pool, device="cuda", dtype=torch.bool)
        self._bp_aligned    = torch.arange(npts_per_cam_bs, device="cuda")
        self._bp_all_aligned = torch.zeros(self.n_kpts_pool, device="cuda", dtype=torch.int64)

    
    def build_problem(
        self,
        desc_kpts_list: list[DescribedKeypoints],
        npts: int,
        n_cams: int,
        min_n_matches: int,
        kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras.

        Match indices index the keyframe's flat keypoint pool, so all lookups here go through
        ``kpts_flat()`` on the ``DescribedKeypoints``.
        """
        npts_per_cam = npts // n_cams
        n_total = desc_kpts_list[0].num_kpts
        uvs = torch.zeros(npts, n_cams, 2, device="cuda") - 1
        cam_indices = torch.zeros(uvs[..., 0].shape, dtype=torch.int32, device="cuda")
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int32, device="cuda") - 1
        unused_kpts_mask = torch.ones(
            (n_cams, n_total), device="cuda", dtype=torch.bool
        )
        for k in range(n_cams):
            idx_occurrences = self._bp_idx_occ[:n_total].zero_()
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            selected_indices = torch.multinomial(
                idx_occurrences.float(), npts_per_cam, replacement=False
            )

            selected_mask = self._bp_sel_mask[:n_total].zero_()
            selected_mask[selected_indices] = True
            aligned_ids = self._bp_aligned[:npts_per_cam]
            all_aligned_ids = self._bp_all_aligned[:n_total].zero_()
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k * npts_per_cam : (k + 1) * npts_per_cam, :, :]
            cam_indices_k = cam_indices[k * npts_per_cam : (k + 1) * npts_per_cam, :]
            xyz_indices_k = xyz_indices[k * npts_per_cam : (k + 1) * npts_per_cam]
            for l in range(n_cams):
                if l == k:
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts_flat()[selected_indices]
                    cam_indices_k[:, l] = l
                    xyz_indices_k[:, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk]
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts_flat()[idxl]
                        cam_indices_k[set_idx, l] = l
                        xyz_indices_k[set_idx, l] = idxl

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        cam_indices[mask, :] = -1
        valid_obs = xyz_indices >= 0
        cam_indices[~valid_obs] = -1
        return uvs, cam_indices, xyz_indices

    def initialize_bootstrap(
        self, desc_kpts_list: list[DescribedKeypoints], keyframe_dicts: list[dict], ordered_bootstrap: bool
    ):
        """
        Estimate focal and initialize the poses of the frames corresponding to desc_kpts_list.
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        images_batch = torch.cat([kfd['image'].cuda() for kfd in keyframe_dicts])
        raw_output = self.da3(images_batch)

        f_init = raw_output["f"][None] if self.f_init <= 0 else torch.Tensor([self.f_init]).cuda()
        Rts_init = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts_init[:, :3, :3] = raw_output["extrinsics"][0, :, :3, :3]
        Rts_init[:, :3, 3] = raw_output["extrinsics"][0, :, :3, 3]

        ## Exhaustive matching
        for i in range(n_cams):
            m_window = min(n_cams, i + self.num_prev_keyframes_window + 1) if ordered_bootstrap else n_cams
            for j in range(i + 1, m_window):
                matches_i = self.matcher(
                    desc_kpts_list[i],
                    desc_kpts_list[j],
                    rotation_idx=0,
                    remove_outliers=True,
                    update_kpts_flag="inliers",
                    kID=i,
                    kID_other=j,
                    use_lightglue=self.use_lightglue,
                    min_match_count=self.min_match_count,
                )

        ## Check that the match graph is connected (no isolated cameras)
        adjacency = {i: set(desc_kpts_list[i].matches.keys()) for i in range(n_cams)}
        visited = {0}
        queue = [0]
        while queue:
            node = queue.pop()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(visited) < n_cams:
            return None, None, None

        ## Build the problem by organizing matches
        uvs, cam_indices, xyz_indices = self.build_problem(
            desc_kpts_list, npts, n_cams, 2, list(range(n_cams))
        )

        # Initialize the points
        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            depths = sample(raw_output["depth"][None, :, k], uvs[mask, k, :][None, None], self.width, self.height)
            xyz = depth2points(uvs[mask, k, :], depths.view(-1, 1), f_init, self.centre)
            R = raw_output["extrinsics"][0, k, :3, :3]
            t = raw_output["extrinsics"][0, k, :3, 3]
            xyz = (xyz - t[None]) @ R
            xyz_init[mask] = xyz

        ## Run miniBA, estimating 3D points, camera focal and poses
        Rts, f, xyz, r, r_init, mask, eff_iters = self.miniba_bootstrap(
            Rts_init, f_init, xyz_init, self.centre, uvs, cam_indices,
        )
        final_residual = (r * mask).abs().sum() / mask.sum()

        self.f = f

        ## Normalize scene scale
        xyz_cam = (Rts[:, :3, :3] @ xyz.T).permute(0, 2, 1) + Rts[:, :3, 3].unsqueeze(1)  # [n_cams, n_pts, 3]
        depths = xyz_cam[..., 2]  # [n_cams, n_pts]
        valid_depths = depths[depths > 0]
        scale = self.scale_depth_target / valid_depths.quantile(self.scale_depth_quantile)
        Rts[:, :3, 3] *= scale
        xyz = scale * xyz.clone()

        # Update 3D points in each keyframe's desc_kpts from the BA solution
        for k in range(n_cams):
            obs = xyz_indices[:, k] >= 0
            if not obs.any():
                continue
            pts_idx = obs.nonzero(as_tuple=True)[0]
            kpt_idx = xyz_indices[pts_idx, k]
            pts3d = xyz[pts_idx]
            depths = (pts3d @ Rts[k, :3, :3].T + Rts[k, :3, 3])[:, 2]
            valid = depths > 0
            desc_kpts_list[k].update_3D_pts(pts3d[valid], kpt_idx[valid])

        return Rts, f, final_residual

    @torch.no_grad()
    def _gather_matches(self, keyframes, curr_desc_kpts, force_add):
        xyz_raw, uvs_raw, cam_raw, masks, idx_raw = [], [], [], [], []
        for l_i, keyframe in enumerate(keyframes):
            matches = curr_desc_kpts.matches.get(keyframe.index, None)
            if matches is not None:
                idx_other = matches.idx_other
                masks.append(keyframe.desc_kpts.get_has_pt3d().view(-1).cuda()[idx_other])   # bool, no sync
                xyz_raw.append(keyframe.desc_kpts.get_pts3d().view(-1, 3)[idx_other])  # integer gather, no sync
                uvs_raw.append(matches.kpts)
                cam_raw.append(torch.full((idx_other.shape[0],), l_i,
                                          device="cuda", dtype=torch.int32))
                idx_raw.append(matches.idx)

        if not xyz_raw:
            if force_add:
                print("No matches at all, force adding keyframe failed.")
            return None, None, None, None

        # 1 sync: nonzero on the combined has-pt3d mask
        valid = torch.cat(masks).nonzero(as_tuple=True)[0]
        if valid.shape[0] == 0:
            if force_add:
                print("No matches at all, force adding keyframe failed.")
            return None, None, None, None

        xyz         = torch.cat(xyz_raw)[valid]
        uvs         = torch.cat(uvs_raw)[valid]
        cam_indices = torch.cat(cam_raw)[valid]
        kpt_idx     = torch.cat(idx_raw)[valid]

        # Subsample the points if there are too many
        if len(xyz) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(
                torch.ones_like(xyz[:, 0]), self.num_pts_miniba_incr, replacement=False
            )
            xyz         = xyz[selected_indices]
            uvs         = uvs[selected_indices]
            cam_indices = cam_indices[selected_indices]
            kpt_idx     = kpt_idx[selected_indices]

        return xyz, uvs, cam_indices, kpt_idx

    @torch.no_grad()
    def _estimate_pose(self, xyz, uvs, cam_indices, kpt_idx, keyframes, index, force_add):
        # Estimate an initial camera pose and inliers using PnP RANSAC
        # Check if we have sufficiently many inliers
        if len(xyz) < self.min_num_inliers and not force_add:
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None, None, None, None, "no_inliers"
        Rt, inliers = self.PnPRANSAC(
            uvs, xyz, self.f, self.centre, confs=torch.ones_like(xyz[:, 0])
        )
        if torch.isnan(Rt).any():
            logging.warning("PnP returned NaN Rt")

        xyz         = xyz[inliers]
        uvs         = uvs[inliers]
        cam_indices = cam_indices[inliers]
        kpt_idx     = kpt_idx[inliers]
        # Subsample the points if there are too many.
        # Track n_real_ba (Python int) to avoid a boolean-index sync in the parallax check.
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(
                torch.rand_like(xyz[..., 0]),
                self.num_pts_miniba_incr,
                dim=0,
                largest=False,
            )[1]
            xyz_ba    = xyz[selected_indices]
            uvs_ba    = uvs[selected_indices]
            cam_indices_ba = torch.zeros(self.num_pts_miniba_incr, device="cuda", dtype=torch.int32)
        elif len(xyz) < self.num_pts_miniba_incr:
            # Use pre-allocated padding buffers to avoid fragmentation
            n_padding = self.num_pts_miniba_incr - len(xyz)
            xyz_ba    = torch.cat([xyz, self.xyz_padding[:n_padding]], dim=0)
            uvs_ba    = torch.cat([uvs, self.uvs_padding[:n_padding]], dim=0)
            cam_indices_ba = torch.cat([
                torch.zeros(len(xyz), device="cuda", dtype=torch.int32),
                torch.full((n_padding,), -1, device="cuda", dtype=torch.int32),
            ], dim=0)

        # Run the initialization
        Rts = torch.eye(4, device="cuda")[None]
        Rts[0, :3, :4] = Rt
        Rts, _, _, r, r_init, mask, eff_iters = self.miniBA_incr(
            Rts, self.f, xyz_ba, self.centre, uvs_ba.view(-1, 1, 2),
            cam_indices=cam_indices_ba.view(-1, 1),
        )
        Rt = Rts[0]

        return Rt, xyz, cam_indices, kpt_idx, mask

    @torch.no_grad()
    def initialize_incremental(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        force_add: bool,
        is_test: bool,
    ):
        """
        Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
        """
        # Match the current frame with previous keyframes
        # Gather with integer indexing (no sync), filter with a single nonzero() at the end.
        xyz, uvs, cam_indices, kpt_idx = self._gather_matches(keyframes, curr_desc_kpts, force_add)

        if xyz is None:
            return None, "no_inliers"

        result = self._estimate_pose(xyz, uvs, cam_indices, kpt_idx, keyframes, index, force_add)
        Rt, xyz, cam_indices, kpt_idx, mask = result

        if Rt is None:
            return None, mask  # mask is the error string here

        # Check if there is enough parallax
        if not is_test and not force_add and len(cam_indices) > 0:
            centre = -Rt[:3, :3].T @ Rt[:3, 3]
            counts = torch.bincount(cam_indices)
            most_present_index = torch.argmax(counts)
            present_indices = torch.where(counts > 0)[0]
            other_centres = torch.stack([keyframes[i].get_centre() for i in present_indices.tolist()])
            baseline = torch.linalg.vector_norm(other_centres - centre, dim=1).max()
            xyz_cam = xyz @ Rt[:3, :3].T + Rt[None, :3, 3]
            depths = xyz_cam[..., -1]
            depths = depths[depths > 0]
            if depths.numel() == 0:
                # No valid depths -- skip the parallax gate this round.
                pass
            else:
                depth_q1 = depths.quantile(self.scale_depth_quantile)
                parallax_thresh = self.baseline_thresh * depth_q1 / self.f
                if (
                    baseline < parallax_thresh
                    and not is_test
                    and not keyframes[most_present_index].is_test
                ):
                    for keyframe in keyframes:
                        keyframe.desc_kpts.matches.pop(index, None)
                    return None, "no_parallax"

        # Check if we have sufficiently many inliers
        if force_add or (mask.sum() > self.min_num_inliers):
            xyz_cam = xyz @ Rt[:3, :3].T + Rt[None, :3, 3]
            valid_depth = xyz_cam[..., -1] > 0
            curr_desc_kpts.update_3D_pts(xyz[valid_depth], kpt_idx[valid_depth])
            return Rt, False
        else:
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None, "no_inliers"

