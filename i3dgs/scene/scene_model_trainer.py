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

import threading
from typing import TYPE_CHECKING, List, Optional
import gc

import torch
import torch.nn.functional as F
import numpy as np

from fused_ssim import fused_ssim
from scene.mono_depth import align_depth
from utils import (
    RGB2SH,
    depth2points,
    get_lapla_norm,
    inverse_sigmoid,
    make_torch_sampler,
    sample,
)
from joblib import Parallel, delayed
from scene.keyframe import Keyframe

if TYPE_CHECKING:
    from argparse import Namespace
    from scene.hierarchy import HierarchyStructure
    from scene.mono_depth import MonoDepthEstimator
    from scene.optimizers import BaseAdam
    from poses.guided_mvs import GuidedMVS


class GaussianTrainerMixin:
    """Mixin providing training, optimization, and Gaussian initialization for SceneModel."""

    args: "Namespace"
    keyframes: List[Keyframe]
    hierarchy: "HierarchyStructure"
    depth_estimator: "MonoDepthEstimator"
    guided_mvs: "GuidedMVS"
    gaussian_optimizer: Optional["BaseAdam"]
    width: int
    height: int
    f: torch.Tensor
    centre: torch.Tensor
    lock: threading.Lock
    optimization_thread: Optional[threading.Thread]
    interrupt_optimization: bool
    training_proba: torch.Tensor
    use_last_frame_proba: float
    last_trained_id: int
    lambda_dssim: float
    init_proba_scaler: float
    max_active_keyframes: int
    max_scale_screen_size: float
    max_sh_degree: int
    approx_cam_centres: Optional[torch.Tensor]
    disc_kernel: torch.Tensor
    uv: torch.Tensor
    xyz: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    children: torch.Tensor
    kf_id: torch.Tensor
    id_in_hierarchy: torch.Tensor

    def _init_pixel_helpers(self, width: int, height: int):
        """Build disc_kernel and uv grid for the given resolution."""
        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()

        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), torch.arange(0, height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .cuda()
        )
    def optimization_step(self, keyframe_id=None, finetuning=False):
        if len(self.xyz) == 0:
            return
        # Select which keyframe to train on
        # We train on the latest keyframe with self.use_last_frame_proba probability or a random keyframe otherwise
        last_kf = self.keyframes[-1]
        if keyframe_id is None:
            if (
                np.random.rand() > self.use_last_frame_proba
                or self.last_trained_id == -1
                or finetuning
                # A test frame only trains its own pose; skip it once frozen.
                or (last_kf.is_test and not last_kf.optimize_pose)
            ):
                keyframe_id = torch.multinomial(self.training_proba, 1, generator=self.gs_rng).item()
            else:
                keyframe_id = -1
        keyframe = self.keyframes[keyframe_id]
        is_test = keyframe.is_test

        gt_image = keyframe.image.to(device="cuda", non_blocking=True)
        keyframe_mask = keyframe.mask
        if keyframe_mask is not None:
            mask = keyframe_mask.to(device="cuda", non_blocking=True)

        # render->backward->step must be atomic w.r.t. main-thread scene
        # mutations while autograd-tracked scene tensors are live.
        with self._scene_mutation_lock():
            keyframe.zero_grad()
            if self.gaussian_optimizer is not None:
                self.gaussian_optimizer.zero_grad()

            render_pkg = self.render_from_id(keyframe_id, bg=torch.rand(3, device="cuda", generator=self.gs_rng))
            image = render_pkg["render"]

            if keyframe_mask is not None:
                image = image * mask
                gt_image = gt_image * mask

            l1_loss = (image - gt_image).abs().mean()
            ssim_loss = 1 - fused_ssim(image[None], gt_image[None])
            loss = (
                self.lambda_dssim * ssim_loss
                + (1 - self.lambda_dssim) * l1_loss
            )

            loss.backward()
            keyframe.step()  # pose / colour correction
            # Test frames refine only their own pose, not the shared scene.
            if not is_test:
                self.gaussian_optimizer.step(
                    render_pkg["visibility_filter"], self.hierarchy.active_gaussians_count
                )

        self.last_trained_id = keyframe_id

    def optimization_loop(self, n_iters: int):
        """
        Runs at least n_iters optimization steps.
        """
        self.interrupt_optimization = False
        i = 0
        while i < n_iters:
            self.optimization_step(finetuning=i>n_iters)
            i += 1
        return

    def join_optimization_thread(self):
        """
        Interrupts the optimization loop and waits for the thread to finish.
        """
        if self.optimization_thread is not None:
            self.interrupt_optimization = True
            self.optimization_thread.join()
            self.optimization_thread = None

    def optimize_async(self, n_iters: int):
        """
        Starts an optimization thread that runs at least n_iters optimization steps.
        """
        self.join_optimization_thread()
        self.optimization_thread = threading.Thread(
            target=self.optimization_loop, args=(n_iters,)
        )
        self.optimization_thread.start()

    def finetune_epoch(self):
        """
        Go through all keyframes and optimize them.
        This is used for finetuning after the initial training.
        """
        for _ in range(len(self.keyframes)):
            self.optimization_step(finetuning=True)

    @torch.no_grad()
    def check_and_create_hierarchy(self):
        n_merged = self.hierarchy.check_and_create_hierarchy(self.keyframes[-1])
        if n_merged > 0 and getattr(self.args, "lr_poses", 0) > 0:
            # Refining poses against the coarse representation degrades quality. Plain
            self.args.lr_poses = 0.0
            for kf in self.keyframes:
                kf.freeze_pose()
        if n_merged > 0 or len(self.keyframes) % self.args.training_proba_update_freq == 0:
            self.update_training_proba()

    @torch.no_grad()
    def update_training_proba(self):
        # Get the contribution of each keyframe to the current leaves (i.e. how many Gaussians each spawn)
        leaf_mask = (self.children == -1).all(dim=-1)
        leaf_kf_id = self.kf_id[leaf_mask]
        pts_per_kf = torch.bincount(leaf_kf_id, minlength=len(self.keyframes))
        training_proba = pts_per_kf / pts_per_kf.max()
        training_proba.nan_to_num_()

        # The 50 closest keyframes to the last are always used for training
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - self.keyframes[-1].approx_centre[None], dim=-1
        )
        closest_ids = dists.argsort()[:self.args.num_closest_active_keyframes]

        # No more than self.max_active_keyframes are used for training
        min_proba = torch.topk(training_proba, min(self.max_active_keyframes, len(training_proba)))[0][-1]

        # Prevent low proba keyframes from being used and send them to CPU
        thresh = max(self.args.min_training_proba, min_proba)
        closest_ids_set = set(closest_ids.tolist())

        optimize_test_poses = getattr(self.args, "lr_poses", 0) > 0

        def process_keyframe(kf: Keyframe):
            # Test frames are handled in a second pass (proba from neighbours).
            if kf.is_test:
                if not optimize_test_poses:
                    training_proba[kf.index] = 0.0
                    kf.to("cpu", only_train=True)
                return
            if kf.index in closest_ids_set:
                training_proba[kf.index] = 1.0
            if training_proba[kf.index] < thresh and kf.index not in closest_ids_set:
                training_proba[kf.index] = 0.0
                kf.to("cpu")
            else:
                kf.to("cuda", only_train=True)

        Parallel(n_jobs=-1, backend="threading")(
            delayed(process_keyframe)(kf) for kf in self.keyframes
        )

        # Keep test frames active with proba = max of their two neighbours.
        if optimize_test_poses:
            n = len(self.keyframes)
            for kf in self.keyframes:
                if not kf.is_test:
                    continue
                i = kf.index
                neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
                proba = max((training_proba[j].item() for j in neighbours),
                            default=training_proba[i].item())
                training_proba[i] = proba
                if proba > 0:
                    kf.to("cuda", only_train=True)
                else:
                    kf.to("cpu", only_train=True)

        # Zero the bootstrapping frames as they add lots of Gaussians
        if min(training_proba[:self.args.num_keyframes_miniba_bootstrap*2]) < thresh:
            training_proba[:self.args.num_keyframes_miniba_bootstrap] = 0.0

        self.training_proba = training_proba


    @torch.no_grad()
    def _compute_init_probability(self, keyframe, keyframe_id):
        img = keyframe.image
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        init_proba = get_lapla_norm(img, self.disc_kernel) # eq. 1

        if keyframe.mask is not None:
            init_proba *= keyframe.mask[0]

        ## Compute the penalty based on the rendering from the new keyframe's point of view
        penalty = 0
        rendered_depth = None
        main_gaussians_map = None
        if self.xyz.shape[0] > 0:
            render_pkg = self.render_from_id(keyframe_id)
            render = render_pkg["render"]
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
            # Per-pixel id of the Gaussian that contributes most (used later to
            # remove primitives that are coarser than the incoming geometry).
            main_gaussians_map = render_pkg["mainGaussID"]
            penalty = get_lapla_norm(render, self.disc_kernel)

        ## Define which pixels should become Gaussians
        init_proba *= self.init_proba_scaler
        init_proba.clamp_min_(1 / (0.5 * self.max_scale_screen_size)**2)
        penalty *= self.init_proba_scaler
        sample_mask = torch.rand_like(init_proba) < init_proba - penalty # eq. 3

        return init_proba, rendered_depth, sample_mask, img, main_gaussians_map

    @torch.no_grad()
    def _estimate_gaussian_depths(self, keyframe, keyframe_id, sample_mask, mono_idepth, mono_depth_conf):
        sampled_uv = self.uv[sample_mask]
        ## Initialize positions
        # Get the samples' depth with guided stereo matching
        prev_KFs = [self.keyframes[id] for id in self.keyframes[keyframe_id].desc_kpts.matches][:self.guided_mvs.n_cams + 1]
        for i, prev_keyframe in enumerate(prev_KFs):
            if keyframe.index == prev_keyframe.index:
                prev_KFs.pop(i)
                break
        depth, accurate_mask = self.guided_mvs(sampled_uv, keyframe, prev_KFs, mono_idepth)
        conf = sample(
            mono_depth_conf,
            sampled_uv.view(1, 1, -1, 2),
            self.width, self.height
        )[0, 0, 0]
        valid_mask = (conf > self.args.depth_conf_threshold) * (depth > 1e-6)
        sample_mask[sample_mask.clone()] = valid_mask
        depth = depth[valid_mask]
        sampled_uv = sampled_uv[valid_mask]
        accurate_mask = accurate_mask[valid_mask]

        return sampled_uv, depth, accurate_mask, sample_mask

    @torch.no_grad()
    def _check_occlusions(self, sampled_uv, depth, accurate_mask, sample_mask, rendered_depth):
        if rendered_depth is not None:
            valid_mask = depth < self.args.occlusion_depth_factor * rendered_depth[sample_mask]
            sample_mask[sample_mask.clone()] = valid_mask
            depth = depth[valid_mask]
            sampled_uv = sampled_uv[valid_mask]
            accurate_mask = accurate_mask[valid_mask]

        return sampled_uv, depth, accurate_mask, sample_mask

    def _make_dummy_ext_tensor(self):
        """Empty extension dict so ``add_and_prune`` can prune without appending."""
        return {
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.log_scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
            "kf_id": self.kf_id[:0].detach(),
            "children": self.children[:0].detach(),
            "parent": self.parent[:0].detach(),
            "id_in_hierarchy": self.id_in_hierarchy[:0].detach(),
        }

    @torch.no_grad()
    def _remove_coarse_gaussians(self, keyframe, keyframe_id, main_gaussians_map, accurate_mask, sample_mask):
        """Remove Gaussians that are coarser than the newly sampled geometry.

        A Gaussian is considered coarse (and removed) when it is the main
        contributor for many of the accurately-sampled new points: the incoming
        finer primitives will represent that region better. Recently-added
        Gaussians and non-leaf hierarchy nodes are always kept. Returns the
        rendered depth from ``keyframe_id`` after the removal.
        """
        # Restrict to pixels whose new point was accurately triangulated.
        accurate_sample_mask = sample_mask.clone()
        accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
        selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]
        ids, counts = torch.unique(
            selected_main_gaussians[selected_main_gaussians >= 0],
            return_counts=True,
        )
        valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
        valid_gs_mask[ids] = (counts < self.args.coarse_removal_count) | (
            (self.kf_id[ids] - keyframe.index).abs() < self.args.coarse_removal_kf_window
        )
        # Never remove non-leaf Gaussians (they are referenced by the hierarchy).
        valid_gs_mask[self.id_in_hierarchy != -1] = True
        with self.lock:
            self.hierarchy.add_and_prune(self._make_dummy_ext_tensor(), valid_gs_mask)
        render_pkg = self.render_from_id(keyframe_id)
        return 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

    @torch.no_grad()
    def _init_gaussian_attributes(self, keyframe, sampled_uv, depth, accurate_mask, sample_mask, init_proba, img):
        R = keyframe.get_R()
        t = keyframe.get_t()
        new_pts = depth2points(sampled_uv, depth.unsqueeze(-1), self.f, self.centre)
        new_pts = (new_pts - t) @ R
        has_pt3d = keyframe.desc_kpts.get_has_pt3d()
        match_pts = keyframe.desc_kpts.get_pts3d()[has_pt3d]
        new_pts = torch.cat([new_pts, match_pts], dim=0)

        ## Initialize Colour
        f_dc = img[:, sample_mask]
        match_sampler = keyframe.desc_kpts.kpts[has_pt3d]
        match_sampler = make_torch_sampler(match_sampler, self.width, self.height)
        match_colors = F.grid_sample(
            img[None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(3, -1)
        f_dc = torch.cat([f_dc, match_colors], dim=1)
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))

        ## Initialize Scales
        sampled_init_proba = init_proba[sample_mask]
        match_init_proba = F.grid_sample(
            init_proba[None, None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(-1)
        sampled_init_proba = torch.cat([sampled_init_proba, match_init_proba], dim=0)

        # Expected distance to the nearest neighbour (eq. 4)
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.max_scale_screen_size)
        # Scale by the distance to the camera centre
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(
            new_pts - keyframe.approx_centre[None], dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

        ## Initialize opacities
        opacities = torch.ones(f_dc.shape[0], 1, device="cuda")
        # Lower inital opacity depending for innacurate points
        opacities[: sampled_uv.shape[0]] *= (
            self.args.init_opacity_accurate * accurate_mask[..., None] + self.args.init_opacity_inaccurate * ~accurate_mask[..., None]
        )
        # High opacity for triangulated Gaussians
        opacities[sampled_uv.shape[0] :] *= self.args.init_opacity_triangulated
        opacities = inverse_sigmoid(opacities)

        ## Initialize SH, rotations as identity
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device="cuda",
        )
        rots = torch.zeros(f_dc.shape[0], 4, device="cuda")
        rots[:, 0] = 1

        extension_tensors = {
            "xyz": new_pts,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
        }
        return new_pts, extension_tensors

    @torch.no_grad()
    def _prune_and_filter_gaussians(self, keyframe, new_pts, extension_tensors):
        ## Get which Gaussians should be pruned
        if self.xyz.shape[0] > 0:
            # Only keep Gaussians with non neglectible opacity
            valid_gs_mask = self.opacity[:, 0] > self.args.prune_opacity_threshold

            # Discard huge Gaussians
            dist = torch.linalg.vector_norm(
                self.xyz - keyframe.approx_centre[None], dim=-1
            )
            screen_size = self.f * self.scaling.max(dim=-1)[0] / dist
            valid_gs_mask *= screen_size < self.max_scale_screen_size
            # Keep non-leaf Gaussians
            valid_gs_mask[self.id_in_hierarchy != -1] = True

        else:
            valid_gs_mask = torch.ones(0, device="cuda", dtype=torch.bool)

        ## Make sure they don't bother another part of the trajectory
        dists_w_kf = torch.linalg.vector_norm(
            new_pts[:, None] - self.approx_cam_centres[None], dim=-1)
        dists_w_current = dists_w_kf[:, keyframe.index]
        min_dist_w_other = dists_w_kf.min(dim=-1)[0]
        dist_mask = min_dist_w_other > 0.1 * dists_w_current

        new_pts = new_pts[dist_mask]
        for k in extension_tensors:
            extension_tensors[k] = extension_tensors[k][dist_mask]

        ## Append the new Gaussians
        extension_tensors["kf_id"] = torch.full(
            (new_pts.shape[0],), keyframe.index, device="cuda", dtype=torch.int32
        )
        extension_tensors["children"] = torch.full(
            (new_pts.shape[0], self.args.num_neighbors_for_hierarchy), -1, device="cuda", dtype=torch.int32
        )
        extension_tensors["parent"] = torch.full(
            (new_pts.shape[0],), -1, device="cuda", dtype=torch.int32
        )
        extension_tensors["id_in_hierarchy"] = torch.full(
            (new_pts.shape[0],), -1, device="cuda", dtype=torch.int32
        )
        with self.lock:
            self.hierarchy.add_and_prune(extension_tensors, valid_gs_mask)

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        """Use the given keyframe to add new Gaussians to the scene model."""
        if self.args.skip_gs:
            return
        keyframe = self.keyframes[keyframe_id]

        # Skip if the keyframe is a test keyframe
        if keyframe.info["is_test"]:
            return

        mono_idepth, mono_depth_conf = self.depth_estimator(keyframe.image[None])
        init_proba, rendered_depth, sample_mask, img, main_gaussians_map = self._compute_init_probability(
            keyframe, keyframe_id)

        has_pt3d = keyframe.desc_kpts.get_has_pt3d()
        if has_pt3d.sum() > self.args.min_pts3d_for_depth_align:
            mono_idepth = align_depth(
                mono_idepth, keyframe, self.width, self.height,
                grid_size=self.args.depth_grid_size,
                outlier_mult=self.args.depth_align_outlier_mult,
                grid_outlier_mult=self.args.depth_grid_outlier_mult,
                min_pts_per_block=self.args.depth_grid_min_pts_per_block,
                min_kpts_per_block=self.args.depth_grid_min_kpts_per_block,
                scale_tolerance=self.args.depth_grid_scale_tolerance)
        else:
            mono_depth_conf = torch.zeros_like(mono_depth_conf)

        sampled_uv, depth, accurate_mask, sample_mask = self._estimate_gaussian_depths(
            keyframe, keyframe_id, sample_mask, mono_idepth, mono_depth_conf)

        # Remove Gaussians that are coarser than the newly sampled points, then
        # re-render to get an up-to-date depth map for the occlusion test.
        if self.xyz.shape[0] > 0 and main_gaussians_map is not None and self.args.coarse_removal_count > 0:
            rendered_depth = self._remove_coarse_gaussians(
                keyframe, keyframe_id, main_gaussians_map, accurate_mask, sample_mask)

        sampled_uv, depth, accurate_mask, sample_mask = self._check_occlusions(
            sampled_uv, depth, accurate_mask, sample_mask, rendered_depth)

        new_pts, extension_tensors = self._init_gaussian_attributes(
            keyframe, sampled_uv, depth, accurate_mask, sample_mask, init_proba, img)

        self._prune_and_filter_gaussians(keyframe, new_pts, extension_tensors)

        self.max_gpu_allocated = max(getattr(self, 'max_gpu_allocated', 0), torch.cuda.max_memory_allocated())
        self.max_n_active_gaussians = max(getattr(self, 'max_n_active_gaussians', 0), self.hierarchy.n_active_gaussians)

        # Free GPU memory back to the driver once the device is running low. We use
        # the driver's free/total figures (torch.cuda.mem_get_info) rather than
        # PyTorch's own allocated/reserved counters, so this also accounts for the
        # CUDA context, cuBLAS workspaces, the rasterizer's allocations, and any
        # other processes sharing the device.
        free, total = torch.cuda.mem_get_info()
        used_fraction = 1 - free / total
        if used_fraction > self.args.gpu_mem_clear_fraction:
            print(f"GPU memory usage at {used_fraction * 100:.0f}% "
                  f"({(total - free) / 1024**3:.1f}/{total / 1024**3:.1f} GB), freeing cache.")
            gc.collect()
            torch.cuda.empty_cache()
