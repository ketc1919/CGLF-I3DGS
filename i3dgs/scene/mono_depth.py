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

import warnings
import logging
import torch
import os
import sys
import urllib.request
import torch.nn.functional as F
from typing import TYPE_CHECKING

from poses.feature_detector import DescribedKeypoints
from utils import sample

if TYPE_CHECKING:
    from scene.keyframe import Keyframe


size = 518
encoder = "vits"


class FixedSizeDepthAnythingV2(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_size: int):
        super().__init__()
        self.model = model
        self.patch_h = input_size // 14
        self.patch_w = input_size // 14

    def forward(self, x: torch.Tensor):
        features = self.model.pretrained.get_intermediate_layers(
            x,
            self.model.intermediate_layer_idx[self.model.encoder],
            return_class_token=True,
        )
        depth = self.model.depth_head(features, self.patch_h, self.patch_w)
        depth = F.relu(depth)
        return depth.squeeze(1)


class MonoDepthInternal(torch.nn.Module):
    def __init__(self, edge_conf_variance: float = 0.2):
        super(MonoDepthInternal, self).__init__()

        sys.path.append("submodules/Depth-Anything-V2")
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
        from depth_anything_v2.dpt import DepthAnythingV2

        self.register_buffer("edge_conf_var", torch.tensor(edge_conf_variance, device="cuda", dtype=torch.half))
        model_path = f"models/depth_anything_v2_{encoder}.pth"
        if not os.path.exists(model_path):
            print(f"Downloading Depth-Anything-V2 model for {encoder}, may take a few minutes...")
            model_sizes = {
                "vits": "Small",
                "vitb": "Base",
                "vitl": "Large",
                "vitg": "Giant",
            }
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_sizes[encoder]}/resolve/main/depth_anything_v2_{encoder}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model = FixedSizeDepthAnythingV2(model.to("cuda").half().eval(), size).eval()
        self.sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )
        self.sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, image: torch.Tensor):
        # Batched: image is ``[B, 3, H, W]``; outputs are ``[B, 1, size, size]``.
        B = image.shape[0]
        img = torch.nn.functional.interpolate(
            image.half(), (size, size), mode="bilinear", align_corners=True
        )
        depth = self.model(img).unsqueeze(1)  # [B, 1, size, size]
        # Per-batch normalisation; a global median is wrong when unrelated
        # frames share a batch.
        flat = depth.reshape(B, -1)
        t = flat.median(dim=1, keepdim=True).values[:, :, None, None]
        centered = depth - t
        s = centered.abs().reshape(B, -1).mean(dim=1, keepdim=True)[:, :, None, None].clamp_min(1e-6)
        ndepth = centered / s

        grad_x = F.conv2d(ndepth, self.sobel_x, padding=1)
        grad_y = F.conv2d(ndepth, self.sobel_y, padding=1)
        # Stack along the channel axis (per-batch), then squash.
        edges = torch.cat((grad_x, grad_y), dim=1)            # [B, 2, size, size]
        edges_sq_norm = (edges**2).sum(1, keepdim=True)       # [B, 1, size, size]
        confidence = torch.exp(-edges_sq_norm / self.edge_conf_var)
        return depth.float(), confidence.float()


def get_t_s(d):
    t = d.median()
    s = (d - t).abs().mean()
    return t, s


def align_samples(tri_idepth: torch.Tensor, mono_idepth: torch.Tensor):
    scale, offset = align_samples_batched(tri_idepth[None], mono_idepth[None])
    scale, offset = scale[0], offset[0]
    return mono_idepth * scale + offset, scale, offset


def robust_align(tri_idepth: torch.Tensor, mono_idepth: torch.Tensor, outlier_mult: float = 5.0):
    mono_idepth_aligned, _, _ = align_samples(tri_idepth, mono_idepth)
    err = (mono_idepth_aligned - tri_idepth).abs()
    m_err = err.median()
    err_dev = (err - m_err).abs().median()
    valid = err < m_err + outlier_mult * err_dev
    return align_samples(tri_idepth[valid], mono_idepth[valid])


def nanmedian(x: torch.Tensor, dim: int):
    out = torch.nanmedian(x, dim=dim)
    return out.values if hasattr(out, "values") else out[0]


def align_samples_batched(tri: torch.Tensor, mono: torch.Tensor, eps=1e-12):
    t_tri  = nanmedian(tri,  dim=1)
    t_mono = nanmedian(mono, dim=1)
    tri_c  = tri  - t_tri[:, None]
    mono_c = mono - t_mono[:, None]
    s_tri = tri_c.abs().nanmean(dim=1)
    s_mono = mono_c.abs().nanmean(dim=1).clamp_min(eps)
    scale  = s_tri / s_mono
    offset = t_tri - t_mono * scale
    return scale, offset


def robust_align_batched(tri: torch.Tensor, mono: torch.Tensor, k=5.0):
    scale, offset = align_samples_batched(tri, mono)
    mono_aligned = mono * scale[:, None] + offset[:, None]
    err = (mono_aligned - tri).abs()
    med_err = nanmedian(err, dim=1)
    keep = err < (k * med_err[:, None])
    tri2  = tri.masked_fill(~keep, float("nan"))
    mono2 = mono.masked_fill(~keep, float("nan"))
    return align_samples_batched(tri2, mono2)


def pack_by_cell(cell_id: torch.Tensor, tri: torch.Tensor, mono: torch.Tensor, C: int):
    device = tri.device
    K = tri.numel()
    perm = cell_id.argsort()
    cell_sorted = cell_id[perm]
    tri_sorted  = tri[perm]
    mono_sorted = mono[perm]
    counts = torch.bincount(cell_sorted, minlength=C)
    M = int(counts.max().item()) if K > 0 else 0
    tri_blk  = torch.full((C, M), float("nan"), device=device, dtype=tri.dtype)
    mono_blk = torch.full((C, M), float("nan"), device=device, dtype=mono.dtype)
    if M == 0:
        return tri_blk, mono_blk, counts
    starts = torch.zeros(C + 1, device=device, dtype=torch.long)
    starts[1:] = torch.cumsum(counts, dim=0)
    idx_global = torch.arange(K, device=device)
    pos_in_cell = idx_global - starts[cell_sorted]
    tri_blk[cell_sorted, pos_in_cell]  = tri_sorted
    mono_blk[cell_sorted, pos_in_cell] = mono_sorted
    return tri_blk, mono_blk, counts


def fill_from_coarser(scales: torch.Tensor, offsets: torch.Tensor):
    """
    Fill nans using coarser parents
    """
    G = scales.shape[0]
    assert scales.shape == (G, G)

    s = scales
    o = offsets

    s_pyr = [s]
    o_pyr = [o]
    g = G
    while g > 1:
        g2 = g // 2
        block = torch.stack([
            s_pyr[-1][0::2, 0::2],
            s_pyr[-1][1::2, 0::2],
            s_pyr[-1][0::2, 1::2],
            s_pyr[-1][1::2, 1::2],
        ], dim=0)
        s_coarse = torch.nanmean(block, dim=0)

        blocko = torch.stack([
            o_pyr[-1][0::2, 0::2],
            o_pyr[-1][1::2, 0::2],
            o_pyr[-1][0::2, 1::2],
            o_pyr[-1][1::2, 1::2],
        ], dim=0)
        o_coarse = torch.nanmean(blocko, dim=0)

        s_pyr.append(s_coarse)
        o_pyr.append(o_coarse)
        g = g2

    for lvl in range(len(s_pyr) - 1, 0, -1):
        fine = s_pyr[lvl - 1]
        coarse = s_pyr[lvl]
        # upsample coarse to fine with repeat
        up = coarse.repeat_interleave(2, 0).repeat_interleave(2, 1)
        fine = torch.where(torch.isfinite(fine), fine, up)
        s_pyr[lvl - 1] = fine

        fineo = o_pyr[lvl - 1]
        coarseo = o_pyr[lvl]
        upo = coarseo.repeat_interleave(2, 0).repeat_interleave(2, 1)
        fineo = torch.where(torch.isfinite(fineo), fineo, upo)
        o_pyr[lvl - 1] = fineo

    return s_pyr[0], o_pyr[0]


def align_depth_grid(mono_idepth: torch.Tensor, kpts, width: int, height: int, sampled_mono_idepth: torch.Tensor, tri_idepth: torch.Tensor, grid_size: int, global_scale, global_offset,
                     min_pts_per_block: int = 10, min_kpts_per_block: int = 30, scale_tolerance: float = 0.5):
    depth_h, depth_w = mono_idepth.shape[-2], mono_idepth.shape[-1]
    sx = (depth_w - 1) / (width - 1)
    sy = (depth_h - 1) / (height - 1)
    x = kpts[:, 0] * sx
    y = kpts[:, 1] * sy
    gx = torch.clamp((x * grid_size / depth_w).long(), 0, grid_size - 1)
    gy = torch.clamp((y * grid_size / depth_h).long(), 0, grid_size - 1)
    cell_id = gy * grid_size + gx
    C = grid_size * grid_size

    tri_blk, mono_blk, counts = pack_by_cell(cell_id, tri_idepth, sampled_mono_idepth, C)

    scale, offset = robust_align_batched(tri_blk, mono_blk, k=5.0)
    scales  = scale.view(grid_size, grid_size)
    offsets = offset.view(grid_size, grid_size)

    # Fill from coarser blocks
    invalid = counts.view(grid_size, grid_size) < min_pts_per_block
    scales  = scales.masked_fill(invalid, float("nan"))
    offsets = offsets.masked_fill(invalid, float("nan"))
    scales, offsets = fill_from_coarser(scales, offsets)
    
    # Fill from neighbour blocks
    valid_down = counts.view(grid_size, grid_size) > min_kpts_per_block
    valid_down_s = ((scales - global_scale).abs()/global_scale < scale_tolerance) & (scales > 0)
    valid_down &= valid_down_s
    
    valid_down_f = valid_down.to(offsets.dtype)
    
    scales_masked = scales * valid_down_f
    offsets_masked = offsets * valid_down_f

    kernel = torch.tensor([[0.0, 1.0, 0.0],
                             [1.0, 1.0, 1.0],
                             [0.0, 1.0, 0.0]]).view(1, 1, 3, 3).cuda()
    n_count_s = F.conv2d(valid_down_f[None, None], kernel, padding=1)

    n_sum_s = F.conv2d(scales_masked[None, None], kernel, padding=1)
    n_avg_s = n_sum_s / n_count_s
    n_avg_s[n_count_s == 0] = global_scale
    scales_fixed = torch.where(valid_down, scales, n_avg_s[0, 0])

    n_sum_o = F.conv2d(offsets_masked[None, None], kernel, padding=1)
    n_avg_o = n_sum_o / n_count_s
    n_avg_o[n_count_s == 0] = global_offset
    offsets_fixed = torch.where(valid_down, offsets, n_avg_o[0, 0])

    scale_map_fixed  = F.interpolate(scales_fixed[None, None], size=mono_idepth.shape[-2:], mode="bilinear", align_corners=True)
    offset_map_fixed = F.interpolate(offsets_fixed[None, None], size=mono_idepth.shape[-2:], mode="bilinear", align_corners=True)

    mono_idepth_aligned = mono_idepth * scale_map_fixed + offset_map_fixed

    return mono_idepth_aligned
    

def align_depth(mono_idepth: torch.Tensor, keyframe: 'Keyframe', width: int, height: int, grid_size: int,
                outlier_mult: float = 5.0, grid_outlier_mult: float = 7.0,
                min_pts_per_block: int = 10, min_kpts_per_block: int = 30, scale_tolerance: float = 0.5):
    """Aligns the mono depth map with the triangulated depth from keypoints by finding the best scale and offset."""
    desc_kpts: DescribedKeypoints = keyframe.desc_kpts
    has_pt3d = desc_kpts.get_has_pt3d()
    pts3d = desc_kpts.get_pts3d()
    kpts = desc_kpts.kpts
    R = keyframe.get_R()
    t = keyframe.get_t()
    tri_depth = (pts3d[has_pt3d] @ R.T + t)[..., -1]
    tri_idepth = 1 / tri_depth
    sampled_mono_idepth = sample(mono_idepth, kpts[has_pt3d].view(1, 1, -1, 2), width, height)[0, 0, 0]

    if grid_size > 0:
        mono_idepth_aligned, global_scale, global_offset = align_samples(tri_idepth, sampled_mono_idepth)
        err = (mono_idepth_aligned - tri_idepth).abs()
        m_err = err.median()
        err_dev = (err - m_err).abs().median()
        valid = err < m_err + grid_outlier_mult * err_dev
        return align_depth_grid(mono_idepth, kpts[has_pt3d][valid], width, height, sampled_mono_idepth[valid], tri_idepth[valid], grid_size, global_scale, global_offset,
                                min_pts_per_block=min_pts_per_block, min_kpts_per_block=min_kpts_per_block, scale_tolerance=scale_tolerance)

    _, scale, offset = robust_align(tri_idepth, sampled_mono_idepth, outlier_mult=outlier_mult)
    return mono_idepth * scale + offset


class MonoDepthEstimator:
    @torch.no_grad()
    def __init__(self, width: int, height: int, edge_conf_variance: float = 0.2, batch: int = 1,
                 use_cuda_graph: bool = True):
        """``batch`` matches the input image batch size."""
        self.width = width
        self.height = height
        self.batch = batch
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available() and not torch.version.hip
        dummy = torch.zeros(batch, 3, height, width).cuda()

        warnings.filterwarnings("ignore")
        logging.getLogger("dinov2").setLevel(logging.ERROR)
        self.model = MonoDepthInternal(edge_conf_variance=edge_conf_variance).eval()
        if self.use_cuda_graph:
            try:
                self.model = torch.cuda.make_graphed_callables(self.model, [dummy])
            except Exception as exc:
                print(f"MonoDepth CUDA graph capture disabled: {exc}")
                self.model(dummy)  # warmup
        else:
            self.model(dummy)  # warmup

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Accepts ``[3, H, W]`` (only valid when ``batch == 1``) or ``[batch, 3, H, W]``.

        Returns ``(depth, conf)``; each is ``[1, H', W']`` for the squeezed
        single-image input, or ``[batch, 1, H', W']`` for batched input.
        """
        squeeze = image.dim() == 3
        if squeeze:
            assert self.batch == 1, (
                f"MonoDepthEstimator configured for batch={self.batch}; got single [3,H,W]"
            )
            image = image[None]
        assert image.shape[0] == self.batch, (
            f"MonoDepthEstimator expects batch={self.batch}, got {image.shape[0]}"
        )
        depth, conf = self.model(image)
        if squeeze:
            return depth[0].clone(), conf[0].clone()
        return depth.clone(), conf.clone()
