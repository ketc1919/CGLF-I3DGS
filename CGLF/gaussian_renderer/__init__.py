#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def repeat_rows(x: torch.Tensor, k: int) -> torch.Tensor:
    return x.unsqueeze(1).expand(-1, k, -1).reshape(-1, x.shape[-1])

def _decode_gaussians_from_scaffold(
    viewpoint_camera,
    pc: GaussianModel,
    anchor: torch.Tensor,
    feat: torch.Tensor,
    grid_offsets: torch.Tensor,
    grid_scaling: torch.Tensor,
    is_training: bool = False,
    camera_uid: int = None,
):
    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist.clamp_min(1e-6)

    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)
        feat_u = feat.unsqueeze(dim=-1)
        feat = (
            feat_u[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
            + feat_u[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
            + feat_u[:, ::1, :1] * bank_weight[:, :, 2:]
        ).squeeze(dim=-1)

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)

    if pc.appearance_dim > 0:
        if camera_uid is None:
            camera_uid = viewpoint_camera.uid
        num_embeddings = getattr(pc.get_appearance, "num_embeddings", None)
        if num_embeddings is None:
            weight = getattr(pc.get_appearance, "weight", None)
            if weight is not None:
                num_embeddings = int(weight.shape[0])
        if num_embeddings is None:
            camera_uid = 0
            num_embeddings = 1
        if num_embeddings > 0:
            camera_uid = int(camera_uid) % int(num_embeddings)
        camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device) * camera_uid
        appearance = pc.get_appearance(camera_indicies)
    else:
        appearance = None

    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view)
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity > 0.0).view(-1)
    opacity = neural_opacity[mask]

    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])

    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])

    offsets = grid_offsets.view([-1, 3])
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat_rows(concatenated, pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets
    offset_norm = offsets.norm(dim=-1, keepdim=True)

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, offset_norm
    return xyz, color, opacity, scaling, rot


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    anchor = pc.get_anchor[visible_mask]
    feat = pc.get_anchor_feat_with_field(anchor, visible_mask, use_stage2=pc.use_stage2_sr, deterministic=not is_training)
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    return _decode_gaussians_from_scaffold(
        viewpoint_camera,
        pc,
        anchor,
        feat,
        grid_offsets,
        grid_scaling,
        is_training=is_training,
        camera_uid=viewpoint_camera.uid,
    )


def generate_neural_gaussians_with_feat_override(
    viewpoint_camera,
    pc: GaussianModel,
    feat_override: torch.Tensor,
    visible_mask=None,
    is_training=False,
):
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    return _decode_gaussians_from_scaffold(
        viewpoint_camera,
        pc,
        anchor,
        feat_override,
        grid_offsets,
        grid_scaling,
        is_training=is_training,
        camera_uid=viewpoint_camera.uid,
    )


def render_with_dense_scaffold(
    viewpoint_camera,
    pc: GaussianModel,
    anchor_xyz: torch.Tensor,
    anchor_feat: torch.Tensor,
    parent_scaling: torch.Tensor,
    parent_offsets: torch.Tensor,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    retain_grad: bool = False,
):
    is_training = pc.get_color_mlp.training
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, offset_norm = _decode_gaussians_from_scaffold(
            viewpoint_camera,
            pc,
            anchor_xyz,
            anchor_feat,
            parent_offsets,
            parent_scaling,
            is_training=True,
            camera_uid=viewpoint_camera.uid,
        )
    else:
        xyz, color, opacity, scaling, rot = _decode_gaussians_from_scaffold(
            viewpoint_camera,
            pc,
            anchor_xyz,
            anchor_feat,
            parent_offsets,
            parent_scaling,
            is_training=False,
            camera_uid=viewpoint_camera.uid,
        )

    screenspace_points = torch.zeros_like(xyz, dtype=anchor_xyz.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            "offset_norm": offset_norm,
        }
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def render_with_appended_dense_scaffold(
    viewpoint_camera,
    pc: GaussianModel,
    visible_mask: torch.Tensor,
    dense_anchor_xyz: torch.Tensor,
    dense_anchor_feat: torch.Tensor,
    dense_scaling: torch.Tensor,
    dense_offsets: torch.Tensor,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    retain_grad: bool = False,
):
    with torch.no_grad():
        base_xyz, base_color, base_opacity, base_scaling, base_rot = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=False
        )

    is_training = pc.get_color_mlp.training
    if is_training:
        dense_xyz, dense_color, dense_opacity, dense_scaling_out, dense_rot, neural_opacity, mask, offset_norm = _decode_gaussians_from_scaffold(
            viewpoint_camera,
            pc,
            dense_anchor_xyz,
            dense_anchor_feat,
            dense_offsets,
            dense_scaling,
            is_training=True,
            camera_uid=viewpoint_camera.uid,
        )
    else:
        dense_xyz, dense_color, dense_opacity, dense_scaling_out, dense_rot = _decode_gaussians_from_scaffold(
            viewpoint_camera,
            pc,
            dense_anchor_xyz,
            dense_anchor_feat,
            dense_offsets,
            dense_scaling,
            is_training=False,
            camera_uid=viewpoint_camera.uid,
        )

    xyz = torch.cat([base_xyz.detach(), dense_xyz], dim=0)
    color = torch.cat([base_color.detach(), dense_color], dim=0)
    opacity = torch.cat([base_opacity.detach(), dense_opacity], dim=0)
    scaling = torch.cat([base_scaling.detach(), dense_scaling_out], dim=0)
    rot = torch.cat([base_rot.detach(), dense_rot], dim=0)

    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    if is_training:
        out["selection_mask"] = mask
        out["neural_opacity"] = neural_opacity
        out["scaling"] = dense_scaling_out
        out["offset_norm"] = offset_norm
    return out

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, offset_norm = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        out = {"render": rendered_image,
               "viewspace_points": screenspace_points,
               "visibility_filter" : radii > 0,
               "radii": radii,
               "selection_mask": mask,
               "neural_opacity": neural_opacity,
               "scaling": scaling,
               "offset_norm": offset_norm,
               }
        if pc.use_stage2_sr:
            anchor_unc = pc.get_anchor_uncertainty(visible_mask)
            unc_masked = anchor_unc.repeat_interleave(pc.n_offsets, dim=0)[mask]
            unc_colors = unc_masked.repeat(1, 3)
            uncertainty_image, _ = rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=unc_colors,
                opacities=opacity,
                scales=scaling,
                rotations=rot,
                cov3D_precomp=None,
            )
            out["uncertainty"] = uncertainty_image[:1]
        return out
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def render_with_feat_override(
    viewpoint_camera,
    pc: GaussianModel,
    feat_override: torch.Tensor,
    uncertainty_override: torch.Tensor = None,
    pipe=None,
    bg_color: torch.Tensor = None,
    scaling_modifier: float = 1.0,
    visible_mask=None,
    retain_grad: bool = False,
):
    is_training = pc.get_color_mlp.training
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, offset_norm = generate_neural_gaussians_with_feat_override(
            viewpoint_camera, pc, feat_override, visible_mask=visible_mask, is_training=True
        )
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians_with_feat_override(
            viewpoint_camera, pc, feat_override, visible_mask=visible_mask, is_training=False
        )

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    if is_training:
        out = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            "offset_norm": offset_norm,
        }
        if uncertainty_override is not None:
            unc_masked = uncertainty_override.repeat_interleave(pc.n_offsets, dim=0)[mask]
            unc_colors = unc_masked.repeat(1, 3)
            uncertainty_image, _ = rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=unc_colors,
                opacities=opacity,
                scales=scaling,
                rotations=rot,
                cov3D_precomp=None,
            )
            out["uncertainty"] = uncertainty_image[:1]
        return out
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if hasattr(rasterizer, "visible_filter"):
        radii_pure = rasterizer.visible_filter(
            means3D=means3D,
            scales=scales[:, :3],
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        return radii_pure > 0

    # Fallback for older rasterizers that only expose frustum culling.
    if hasattr(rasterizer, "markVisible"):
        return rasterizer.markVisible(means3D)

    return torch.ones(means3D.shape[0], dtype=torch.bool, device=means3D.device)
