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

import math
import os
import contextlib
import warnings
from typing import TYPE_CHECKING, List, Optional

import torch
import numpy as np

from fused_ssim import fused_ssim
from utils import (
    psnr,
    rotation_distance,
    align_poses,
    batch_write_imgs,
)

if TYPE_CHECKING:
    from scene.hierarchy import HierarchyStructure
    from scene.keyframe import Keyframe


class EvaluationMixin:
    """Mixin providing evaluation, test rendering, and exposure harmonization for SceneModel."""

    keyframes: List["Keyframe"]
    hierarchy: "HierarchyStructure"
    inference_mode: bool
    render_tau: float
    lpips: Optional[object]

    def _ensure_lpips(self):
        if self.lpips is not None:
            return
        try:
            import lpips
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                warnings.filterwarnings("ignore")
                self.lpips = lpips.LPIPS(net="vgg").cuda()
        except Exception:
            self.lpips = None

    @torch.no_grad()
    def harmonize_test_exposure(self):
        """Sets each test keyframe's colour correction to the average of its nearest train neighbours."""
        for index, keyframe in enumerate(self.keyframes):
            if not keyframe.info["is_test"]:
                continue
            prev_kf = next(
                (kf for kf in reversed(self.keyframes[:index]) if not kf.info["is_test"]),
                None,
            )
            next_kf = next(
                (kf for kf in self.keyframes[index + 1:] if not kf.info["is_test"]),
                None,
            )
            neighbours = [kf for kf in (prev_kf, next_kf) if kf is not None]
            if not neighbours:
                continue
            keyframe.colour_corr.copy_(
                sum(kf.colour_corr for kf in neighbours) / len(neighbours)
            )

    @torch.no_grad()
    def evaluate(self, eval_poses=False, with_LPIPS=False, all=False, out_dir=""):
        # Make sure test keyframes have similar exposure matrices compared to their neighbors
        self.harmonize_test_exposure()

        # Compute image quality metrics
        metrics = {"PSNR": 0, "SSIM": 0}
        if with_LPIPS:
            self._ensure_lpips()
            metrics["LPIPS"] = 0
        n_test_frames = 0
        start_index = 0
        imgs = []
        imgs_paths = []
        for keyframe in self.keyframes[start_index:]:
            if keyframe.info["is_test"] and not self.args.skip_gs:
                if not all and abs(keyframe.index - len(self.keyframes)) > 250:
                    continue
                if self.inference_mode:
                    self.hierarchy.update_cut_for_test(keyframe.get_centre(), self.render_tau)
                gt_image = keyframe.image.cuda()
                render_pkg = self.render_from_id(keyframe.index)
                image = render_pkg["render"]
                mask = keyframe.mask.cuda() if keyframe.mask is not None else torch.ones_like(image[:1] > 0)
                mask = mask.expand_as(image)
                image = image * mask
                gt_image = gt_image * mask
                metrics["PSNR"] += psnr(image[mask], gt_image[mask])
                metrics["SSIM"] += fused_ssim(
                    image[None], gt_image[None], train=False
                ).item()
                if with_LPIPS and self.lpips is not None:
                    metrics["LPIPS"] += self.lpips(image[None], gt_image[None]).item()
                if out_dir != "":
                    imgs.append(image.cpu())
                    imgs_paths.append(os.path.join(out_dir, keyframe.info["name"]))
                n_test_frames += 1

        if n_test_frames > 0:
            for metric in metrics:
                metrics[metric] /= n_test_frames
        else:
            metrics = {}

        if out_dir != "" and not self.args.skip_gs:
            batch_write_imgs(imgs, imgs_paths)

        # Compute pose errors
        if eval_poses:
            Rts = self.get_Rts()[self.gt_Rts_mask]
            gt_Rts = self.get_gt_Rts(align=False)
            if len(Rts) == len(gt_Rts) and len(Rts) > 0:
                Rts_aligned = torch.linalg.inv(align_poses(Rts, gt_Rts))
                gt_Rts = torch.linalg.inv(gt_Rts)

                R_error = rotation_distance(Rts_aligned[:, :3, :3], gt_Rts[:, :3, :3])
                t_error = (Rts_aligned[:, :3, 3] - gt_Rts[:, :3, 3]).norm(dim=-1)

                cam_centers = gt_Rts[:, :3, 3]
                avg_cam_center = torch.mean(cam_centers, axis=0, keepdims=True)
                dist = torch.linalg.norm(cam_centers - avg_cam_center, axis=1, keepdims=True)
                diagonal = torch.quantile(dist, 0.9)

                metrics["R°"] = R_error.mean().item() * 180 / math.pi
                metrics["t"] = 1000 * (t_error.mean() / diagonal).item()

        return metrics

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        self.harmonize_test_exposure()
        imgs = []
        imgs_paths = []
        for kf in self.keyframes:
            if not kf.info["is_test"]:
                continue
            imgs.append(self.render_from_id(kf.index)["render"])
            imgs_paths.append(os.path.join(out_dir, kf.info["name"]))
        batch_write_imgs(imgs, imgs_paths)
