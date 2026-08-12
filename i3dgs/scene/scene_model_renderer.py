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
import math
import threading
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils import focal2fov, fov2focal, getProjectionMatrix

if TYPE_CHECKING:
    from argparse import Namespace
    from scene.hierarchy import HierarchyStructure
    from scene.keyframe import Keyframe

logger = logging.getLogger(__name__)


class RenderingMixin:
    """Mixin providing rendering capabilities for SceneModel."""

    args: "Namespace"
    keyframes: List["Keyframe"]
    hierarchy: "HierarchyStructure"
    width: int
    height: int
    f: torch.Tensor
    lock: threading.Lock
    active_sh_degree: int
    max_scale_screen_size: float
    render_tau: float
    inference_mode: bool
    autograd_dummy: torch.Tensor
    xyz: torch.Tensor
    log_scaling: torch.Tensor
    opacity: torch.Tensor
    f_dc: torch.Tensor
    f_rest: torch.Tensor

    def render_from_id(
        self,
        keyframe_id,
        scaling_modifier=1,
        bg=torch.zeros(3, device="cuda"),
        compensate_exposure=True,
    ):
        """
        Render the scene from a given keyframe id.
        Applies the exposure matrix of the keyframe to the rendered image.
        """
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt().T
        width, height = self.width, self.height
        render_pkg = self.render(
            width, height, view_matrix, scaling_modifier, bg)
        if compensate_exposure:
            render_pkg["render"] = self.apply_compensation(keyframe, render_pkg["render"])
        return render_pkg

    def apply_compensation(self, keyframe, image):
        _, height, width = image.shape
        image = (
            keyframe.colour_corr[:3, :3] @ image.view(3, -1)
        ) + keyframe.colour_corr[:3, 3, None]
        return image.clamp(0, 1).view(3, height, width)

    def _prepare_render_params(self, top_view, scaling_modifier):
        if self.xyz.shape[0] == 0:
            return None
        # Set constant scaling and opacity to visualize the Gaussians' positions in the top view
        if top_view:
            log_scaling = torch.full_like(self.log_scaling, torch.log(torch.tensor(scaling_modifier)).item())
            opacity = torch.ones_like(self.opacity)
        else:
            log_scaling = self.log_scaling
            opacity = self.hierarchy.active_gaussians["opacity"]["val"][:self.hierarchy.active_gaussians_count]
        f_dc = self.f_dc
        f_rest = self.f_rest
        xyz = self.xyz
        rotation = self.hierarchy.active_gaussians["rotation"]["val"][:self.hierarchy.active_gaussians_count]

        ## Fewer GS for top view
        max_gs_topview = 200_000
        step = len(xyz) // max_gs_topview
        if top_view and step > 1:
            xyz = xyz[::step].contiguous()
            opacity = opacity[::step].contiguous()
            f_dc = f_dc[::step].contiguous()
            f_rest = f_rest[::step].contiguous()
            log_scaling = log_scaling[::step].contiguous()
            rotation = rotation[::step].contiguous()

        return xyz, opacity, f_dc, f_rest, log_scaling, rotation

    def _render(self, xyz, opacity_raw, f_dc, f_rest, log_scaling, rotation_raw,
                       screenspace_points, width, height, f, view_matrix, cam_centre, bg):
        """Render via the diff-gaussian-rasterization backend.

        Applies activations (sigmoid/exp/normalize) so autograd accumulates gradients
        into the raw active_gaussians val tensors, where SparseGaussianAdam reads them.
        """
        opacity = torch.sigmoid(opacity_raw)
        scales = torch.exp(log_scaling)
        rotations = F.normalize(rotation_raw, dim=-1)

        if f == self.f.item():
            tanfovx, tanfovy = self.tanfovx, self.tanfovy
            projection_matrix = self.projection_matrix
        else:
            fov_x = focal2fov(float(f), int(width))
            fov_y = focal2fov(float(f), int(height))
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix(
                    znear=getattr(self.args, "render_near_plane", 0.01),
                    zfar=getattr(self.args, "render_far_plane", 10000.0),
                    fovX=fov_x, fovY=fov_y,
                )
                .transpose(0, 1)
                .cuda()
            )

        raster_settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=1.0,
            projmatrix=projection_matrix,
            sh_degree=self.active_sh_degree,
            campos=cam_centre,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings)
        color, invdepth, mainGaussID, radii = rasterizer(
            xyz, screenspace_points, opacity, f_dc, f_rest, scales, rotations, view_matrix,
        )
        return color, invdepth, mainGaussID, radii

    def render(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        scaling_modifier: float,
        bg: torch.Tensor = torch.zeros(3, device="cuda"),
        top_view: bool = False,
        fov_x: float = None,
        fov_y: float = None,
    ):
        cam_centre = -(view_matrix[:3, :3] @ view_matrix[3, :3])

        if fov_x is None and fov_y is None:
            f = self.f.item()
        elif fov_x is not None and fov_y is not None:
            f = (fov2focal(fov_x, width) + fov2focal(fov_y, height)) / 2
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")

        if self.inference_mode and not top_view:
            self.hierarchy.update_cut(cam_centre, self.render_tau)

        with self.lock:
            params = self._prepare_render_params(top_view, scaling_modifier)
            if params is not None:
                xyz, opacity, f_dc, f_rest, log_scaling, rotation = params

                screenspace_points = torch.zeros_like(xyz, requires_grad=True)

                color, invdepth, mainGaussID, radii = self._render(
                    xyz, opacity, f_dc, f_rest, log_scaling, rotation,
                    screenspace_points, width, height, f, view_matrix,
                    cam_centre, bg)
            else:
                # If no Gaussians are present, return empty tensors
                color = torch.zeros(3, height, width, device="cuda")
                invdepth = torch.zeros(1, height, width, device="cuda")
                # The rasterizer emits a [2, H, W] map (best / second-best contributor).
                mainGaussID = torch.zeros(
                    2, height, width, device="cuda", dtype=torch.int32
                )
                radii = torch.zeros(len(self.xyz), device="cuda")
                screenspace_points = torch.zeros_like(self.xyz, requires_grad=True)

        return {
            "render": color,
            "invdepth": invdepth,
            "mainGaussID": mainGaussID,
            "radii": radii,
            "visibility_filter": radii > 0,
            "screenspace_points": screenspace_points,
        }
