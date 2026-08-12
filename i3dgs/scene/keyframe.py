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

from __future__ import annotations
from argparse import Namespace
import torch
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.optimizers import BaseAdam
from utils import sample, make_torch_sampler, depth2points, sixD2mtx
from dataloaders.read_write_model import Camera, BaseImage, rotmat2qvec


class Keyframe:
    """
    A keyframe / capture.

    Stored tensors:
        image:                [3, H, W]
        mask:                 [1, H, W] | None  (bool)
        feat_map:             [H', W', C] | None
        mono_idepth:          [1, H, W] | None     (set later)
        mono_depth_conf:      [1, H, W] | None
        mono_idepth_dx:       [1, H, W-1] | None
        mono_idepth_dy:       [1, H-1, W] | None
        Rt:                   [4, 4] (optimised)
        f:                    scalar focal tensor
        centre:               [2] (principal point)
    """

    def __init__(
        self,
        image: torch.Tensor,
        info: dict,
        desc_kpts: DescribedKeypoints,
        Rt: torch.Tensor,
        index: int,
        f: torch.Tensor,
        feat_extractor: DenseExtractor,
        triangulator: Triangulator,
        args: Namespace,
        inference_mode: bool = False,
    ):
        if image.dim() == 4:
            assert image.shape[0] == 1, f"Keyframe expects one image, got {tuple(image.shape)}"
            image = image[0]
        assert image.dim() == 3, f"Keyframe expects image shaped [3,H,W], got {tuple(image.shape)}"
        self.image = image
        self.mask = None
        self.feat_map = None

        if not inference_mode:
            self.feat_map = feat_extractor(self.image)
            self.width = self.image.shape[2]
            self.height = self.image.shape[1]
            self.centre = torch.tensor(
                [(self.width - 1) / 2, (self.height - 1) / 2],
                device=self.image.device,
            )
            self.f = f
            self.triangulator = triangulator

            mask = info.pop("mask", None)
            if mask is not None:
                if mask.dim() == 4:
                    assert mask.shape[:2] == (1, 1), f"mask must be [1,H,W], got {tuple(mask.shape)}"
                    mask = mask[0]
                assert mask.dim() == 3 and mask.shape[0] == 1, (
                    f"mask must be [1,H,W], got {tuple(mask.shape)}"
                )
                mask = mask.to(self.image.device)
                if mask.dtype != torch.bool:
                    mask = mask > (1 - 1e-3)
                self.mask = mask
            else:
                self.mask = None

        self.index = index
        self.last_opt = -1

        self.desc_kpts = desc_kpts
        self.info = info
        self.is_test = info["is_test"]

        # When refining poses we store them as optimisable
        # 6D-rotation + translation params so the rasterizer's viewmatrix gradient
        # reaches the camera; otherwise a plain tensor (zero overhead).
        self.optimize_pose = (
            not inference_mode
            and getattr(args, "lr_poses", 0) > 0
        )
        if self.optimize_pose:
            self.rW2C = torch.nn.Parameter(Rt[:3, :2].clone())
            self.tW2C = torch.nn.Parameter(Rt[:3, 3].clone())
        else:
            self.Rt = Rt.clone()
        colour_corr = torch.eye(3, 4, device="cuda")
        self.colour_corr = torch.nn.Parameter(colour_corr)

        # Monocular depth (set later by add_new_gaussians).
        self.mono_idepth = None
        self.mono_depth_conf = None
        self.mono_idepth_dx = None
        self.mono_idepth_dy = None

        if not inference_mode:
            params = {}
            if not info["is_test"] and args.lr_colour_corr > 0:
                params["colour_corr"] = {"val": self.colour_corr, "lr": args.lr_colour_corr}
            # Refine every keyframe's pose, test frames included (they refine
            # their own pose but never step the shared scene).
            if self.optimize_pose:
                params["rW2C"] = {"val": self.rW2C, "lr": args.lr_poses}
                params["tW2C"] = {"val": self.tW2C, "lr": args.lr_poses}

            self.optimizer = BaseAdam(params, betas=(args.exposure_adam_beta1, args.exposure_adam_beta2))
            self.num_steps = 0

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    @torch.no_grad()
    def freeze_pose(self):
        """
        Stop optimizing this keyframe's pose with the Gaussians
        """
        if not self.optimize_pose:
            return
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(self.rW2C)
        Rt[:3, 3] = self.tW2C
        self.Rt = Rt
        self.optimize_pose = False

    @torch.no_grad()
    def to(self, device: str, only_train=False):
        if self.device.type == device:
            return
        self.image = self.image.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        if self.mono_idepth is not None:
            self.mono_idepth = self.mono_idepth.to(device)
        if self.mono_depth_conf is not None:
            self.mono_depth_conf = self.mono_depth_conf.to(device)
        if self.mono_idepth_dx is not None:
            self.mono_idepth_dx = self.mono_idepth_dx.to(device)
        if self.mono_idepth_dy is not None:
            self.mono_idepth_dy = self.mono_idepth_dy.to(device)
        if not only_train:
            if self.feat_map is not None:
                self.feat_map = self.feat_map.to(device)
            if self.desc_kpts is not None:
                self.desc_kpts.to(device)

    @property
    def device(self):
        return self.image.device

    def get_R(self):
        if self.optimize_pose:
            return sixD2mtx(self.rW2C)
        return self.Rt[:3, :3]

    def get_t(self):
        if self.optimize_pose:
            return self.tW2C
        return self.Rt[:3, 3]

    def get_Rt(self):
        if not self.optimize_pose:
            return self.Rt
        # Differentiable 4x4 so gradients reach rW2C / tW2C.
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def get_K(self):
        K = torch.eye(3, device=self.device)
        f = self.f if self.f.numel() == 1 else self.f[:1]
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = self.centre[0]
        K[1, 2] = self.centre[1]
        return K

    def set_Rt(self, Rt: torch.Tensor):
        with torch.no_grad():
            if self.optimize_pose:
                self.rW2C.data.copy_(Rt[:3, :2])
                self.tW2C.data.copy_(Rt[:3, 3])
            else:
                self.Rt.copy_(Rt)
            self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    @torch.no_grad()
    def get_centre(self, approx=False):
        if approx:
            return self.approx_centre
        else:
            return -self.get_R().T @ self.get_t()

    @torch.no_grad()
    def update_3dpts(self, all_keyframes: list[Keyframe]):
        """
        Assign a 3D point to each keypoint in the keyframe based on triangulation and the latest rendered depth.
        """
        ## Triangulation
        # Select keyframes to triangulate with based on their locations
        uv, uvs_others, chosen_kfs_ids = self.triangulator.prepare_matches(
            self.desc_kpts
        )

        Rt = self.get_Rt()
        Rts_others = torch.stack(
            [all_keyframes[index].get_Rt() for index in chosen_kfs_ids],
            dim=0,
        )
        # Compute a median depth prior from valid 3D points
        pts = self.desc_kpts.get_pts3d()[self.desc_kpts.get_has_pt3d()]
        depths_prior = (pts @ Rt[:3, :3].T + Rt[None, :3, 3])[:, 2]
        depths_prior = depths_prior[depths_prior > 0]
        if len(depths_prior) == 0:
            return
        depth_q1 = depths_prior[depths_prior > 0].quantile(0.25).item()
        new_pts, depth, best_dis, valid_matches = self.triangulator(
            uv, uvs_others, Rt, Rts_others, self.f, self.centre, depth_q1,
        )
        # Compute a median depth prior from valid 3D points
        pts = torch.cat([
            self.desc_kpts.get_pts3d()[self.desc_kpts.get_has_pt3d()],
            new_pts[best_dis>0],
        ])
        depths_prior = (pts @ Rt[:3, :3].T + Rt[None, :3, 3])[:, 2]
        depth_q1 = depths_prior[depths_prior > 0].quantile(0.25).item()
        valid_matches = best_dis > (self.triangulator.min_dis / depth_q1)

        self.desc_kpts.update_3D_pts(
            new_pts[valid_matches], valid_matches
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        self.num_steps += 1

    def to_json(self):
        info = {
            "is_test": self.info["is_test"],
        }
        if "name" in self.info:
            info["name"] = self.info["name"]
        if "Rt" in self.info:
            info["gt_Rt"] = self.info["Rt"].cpu().numpy().tolist()

        return {
            "info": info,
            "Rt": self.get_Rt().detach().cpu().numpy().tolist(),
            "f": self.f.item(),
        }

    @classmethod
    def from_json(cls, config, index, height, width):
        info = dict(config["info"])
        if "gt_Rt" in info:
            info["Rt"] = torch.tensor(info["gt_Rt"], device="cuda")

        f = config.get("f", None)
        f = torch.tensor([float(f)], device="cuda") if f is not None else None
        keyframe = cls(
            image=torch.empty((1, 3, 0, 0), device="cuda"),
            info=info,
            desc_kpts=None,
            Rt=torch.tensor(config["Rt"]).cuda(),
            index=index,
            f=f,
            triangulator=None,
            feat_extractor=None,
            args=None,
            inference_mode=True,
        )
        keyframe.height = height
        keyframe.width = width
        keyframe.centre = torch.tensor(
            [(width - 1) / 2, (height - 1) / 2], device="cuda"
        )
        keyframe.f = f
        return keyframe

    def to_colmap(self, id, first_colmap_id=None):
        """Convert the keyframe to COLMAP entries.

        Returns single-camera COLMAP entries as one-element lists.
        """
        colmap_id = id if first_colmap_id is None else first_colmap_id
        base_name = self.info.get("name", str(id))
        camera = Camera(
            id=colmap_id,
            model="SIMPLE_PINHOLE",
            width=self.width,
            height=self.height,
            params=[self.f.item(), self.centre[0].item(), self.centre[1].item()],
        )
        image = BaseImage(
            id=colmap_id,
            name=base_name,
            camera_id=colmap_id,
            qvec=-rotmat2qvec(self.get_R().cpu().detach().numpy()),
            tvec=self.get_t().flatten().cpu().detach().numpy(),
            xys=[],
            point3D_ids=[],
        )

        return [camera], [image]
