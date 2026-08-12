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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import types
from typing import Dict

from poses.matcher import Matches

class DescribedKeypoints():
    """
    2D keypoints, descriptors, matches and triangulated 3D positions.

    Fields:
        kpts:              [N, 2]
        feats:             [N, C]  (upright descriptor)
        full_feats:        [Norient, N, C]
        has_pt3d:          [N]   bool
        pts3d:             [N, 3]
        id_in_ba_problem:  [N]   int
    """
    @torch.no_grad()
    def __init__(self, kpts, feats):
        # kpts:  [N, 2]
        # feats: [Norient, N, C]  (Norient = 1 or 4 with rotated descriptors)
        assert kpts.dim() == 2 and kpts.shape[-1] == 2, f"kpts must be [N,2], got {tuple(kpts.shape)}"
        assert feats.dim() == 3, f"feats must be [Norient,N,C], got {tuple(feats.shape)}"
        N = kpts.shape[0]
        self.kpts = kpts
        self.feats = feats[0]             # [N, C] upright
        self.full_feats = feats           # [Norient, N, C]
        self.has_pt3d = torch.zeros(N, dtype=torch.bool, device=kpts.device)
        self.pts3d = torch.zeros((N, 3), dtype=torch.float, device=kpts.device)
        self.id_in_ba_problem = torch.full((N,), -1, dtype=torch.int, device=kpts.device)
        self.matches: Dict[int, Matches] = {}
        self.ba_problem = None

    @property
    def num_kpts(self):
        return self.kpts.shape[0]

    def kpts_flat(self):
        return self.kpts

    def feats_flat(self):
        return self.feats

    def full_feats_flat(self):
        return self.full_feats

    def has_pt3d_flat(self):
        return self.has_pt3d

    def pts3d_flat(self):
        return self.pts3d

    def id_in_ba_problem_flat(self):
        return self.id_in_ba_problem

    @torch.no_grad()
    def update_matches(self, other_id, matches, swap=False):
        if swap:
            matches = Matches(matches.kpts_other, matches.kpts, matches.idx_other, matches.idx)
        self.matches[other_id] = matches

    @torch.no_grad()
    def invalidate_masked(self, masks):
        """Zero descriptors of keypoints whose pixel falls in a masked-out region.

        ``masks`` is a ``[1, H, W]`` (or ``[H, W]``) bool tensor where
        ``True`` marks a valid pixel. Keypoints in invalid pixels get their
        ``feats`` and ``full_feats`` zeroed, so every cosine similarity
        against them is 0 and they are never picked as a match.
        """
        if masks is None:
            return
        if masks.dim() == 3:
            assert masks.shape[0] == 1, f"masks must be [1,H,W] or [H,W], got {tuple(masks.shape)}"
            masks = masks[0]
        assert masks.dim() == 2, f"masks must be [1,H,W] or [H,W], got {tuple(masks.shape)}"
        H, W = masks.shape[-2:]
        x = self.kpts[..., 0].round().long().clamp(0, W - 1)
        y = self.kpts[..., 1].round().long().clamp(0, H - 1)
        valid = masks[y, x].to(self.feats.dtype)
        self.feats.mul_(valid[..., None])
        self.full_feats.mul_(valid[None, ..., None])

    @torch.no_grad()
    def update_3D_pts(self, pts3D, idx):
        self.has_pt3d[idx] = True
        self.pts3d[idx] = pts3D

    @torch.no_grad()
    def get_pts3d(self):
        if self.ba_problem is not None:
            return self.ba_problem.landmarks[self.id_in_ba_problem]
        else:
            return self.pts3d

    @torch.no_grad()
    def get_has_pt3d(self):
        if self.ba_problem is not None:
            return self.id_in_ba_problem >= 0
        else:
            return self.has_pt3d

    @torch.no_grad()
    def to(self, device):
        self.kpts = self.kpts.to(device)
        self.feats = self.feats.to(device)
        if self.full_feats is not None:
            self.full_feats = self.full_feats.to(device)
        if self.has_pt3d is not None:
            self.has_pt3d = self.has_pt3d.to(device)
            self.pts3d = self.pts3d.to(device)
        if self.id_in_ba_problem is not None:
            self.id_in_ba_problem = self.id_in_ba_problem.to(device)
        for key in self.matches:
            self.matches[key].kpts = self.matches[key].kpts.to(device)
            self.matches[key].kpts_other = self.matches[key].kpts_other.to(device)
            self.matches[key].idx = self.matches[key].idx.to(device)
            self.matches[key].idx_other = self.matches[key].idx_other.to(device)

# From https://github.com/verlab/accelerated_features
# XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
# (a copy is provided in licenses/Apache-2.0.txt)
class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """
    def __init__(self, mode = 'bicubic', align_corners = False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)

class Detector():
    @staticmethod
    def _detect_and_compute(x, extractor, use_rotated_descriptors):
        # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
        # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license.
        # Batched: x can be [B, 3, H, W] with B >= 1. The rotated-descriptor path requires B == 1.
        top_k = extractor.top_k
        detection_threshold = extractor.detection_threshold
        x, rh1, rw1 = extractor.preprocess_tensor(x)

        B, _, _H1, _W1 = x.shape

        if use_rotated_descriptors:
            assert B == 1, "Rotated descriptors require batch size 1"
            x1 = torch.rot90(x, k=1, dims=(2, 3))
            x2 = torch.rot90(x, k=2, dims=(2, 3))
            x3 = torch.rot90(x, k=3, dims=(2, 3))
            x_cat02 = torch.cat([x, x2], dim=0)
            x_cat13 = torch.cat([x1, x3], dim=0)
            M, K1, H1 = extractor.net(x_cat02)
            M = F.normalize(M, dim=1)
            M0 = M[0][None]
            M2 = M[1][None]

            M, _, _ = extractor.net(x_cat13)
            M = F.normalize(M, dim=1)
            M1 = M[0][None]
            M3 = M[1][None]

            # Use the first orientation for keypoint detection (rotated path returns
            # interleaved features, so we still slice the upright one out of the cat).
            K1 = K1[0][None]
            H1 = H1[0][None]
        else:
            M, K1, H1 = extractor.net(x)
            M = F.normalize(M, dim=1)
            M0 = M  # [B, C, h, w]

        K1h = extractor.get_kpts_heatmap(K1)
        xOut, mkpts = extractor.NMS(K1h, threshold=detection_threshold, kernel_size=5)

        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        invalid_mask = torch.all(mkpts == 0, dim=-1)
        scores = torch.where(invalid_mask, -1.0, scores)

        idxs = torch.topk(scores, top_k)[1]
        mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)
        mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)
        xOut = torch.gather(xOut, -1, idxs)
        mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)
        scores *= xOut > 0

        # Sample descriptors at keypoint locations.
        feats = extractor.interpolator(M0, mkpts, H=_H1, W=_W1)  # [B, N, C]

        if use_rotated_descriptors:
            mkpts1 = torch.stack([mkpts[..., 1], _W1 - 1 - mkpts[..., 0]], dim=-1)  # 90° CCW
            mkpts2 = torch.stack([_W1 - 1 - mkpts[..., 0], _H1 - 1 - mkpts[..., 1]], dim=-1)  # 180°
            mkpts3 = torch.stack([_H1 - 1 - mkpts[..., 1], mkpts[..., 0]], dim=-1)  # 270° CCW

            feats1 = extractor.interpolator(M1, mkpts1, H=_W1, W=_H1)
            feats2 = extractor.interpolator(M2, mkpts2, H=_H1, W=_W1)
            feats3 = extractor.interpolator(M3, mkpts3, H=_W1, W=_H1)

            # Stack orientations along axis 0: (4, B=1, N, C)
            feats = torch.stack([feats, feats1, feats2, feats3], dim=0)
        else:
            # Single orientation, keep shape uniform: (1, B, N, C)
            feats = feats[None]

        feats = F.normalize(feats, dim=-1)

        # Rescale keypoints back to the original input resolution.
        mkpts = mkpts.float() * torch.tensor([rw1, rh1], device=mkpts.device, dtype=torch.float).view(1, 1, -1)

        # Zero descriptors of invalid keypoints, broadcast (Norient, B, N, 1).
        feats = feats * (scores[None, ..., None] > 0)
        return mkpts, feats

    @torch.no_grad()
    def __init__(self, top_k, width, height, use_rotated_descriptors=True, batch=1,
                 min_pixels=1_000_000):
        if use_rotated_descriptors:
            assert batch == 1, "Rotated descriptors require batch == 1"
        self.batch = batch
        rot_suffix = "rot" if use_rotated_descriptors else "norot"

        # Process at >= min_pixels: small inputs are upscaled inside the trace so the
        # network never runs below ~1Mpx. Keypoints are rescaled back to the input
        # resolution before being returned, so downstream code is unaffected.
        fixed_H, fixed_W = (height // 32) * 32, (width // 32) * 32
        res_suffix = ""
        if fixed_H * fixed_W < min_pixels:
            scale = max(1.0, (min_pixels / (width * height)) ** 0.5)
            fixed_H = math.ceil(height * scale / 32) * 32
            fixed_W = math.ceil(width * scale / 32) * 32
            res_suffix = f"_p{fixed_W}x{fixed_H}"
        cache_path = f"models/cache/xfeat_{rot_suffix}_b{batch}{res_suffix}_{width}_{height}_{top_k}.pt"
        dummy_img = torch.randn(batch, 3, height, width).cuda().to(torch.half)
        if os.path.exists(cache_path):
            extractor = torch.jit.load(cache_path)
        else:
            print(f"Compiling feature extractor (batch={batch})")
            extractor = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=top_k)
            extractor = extractor.cuda().eval().to(torch.half)

            ## Overriding the functions to run at fixed size (fixed_H, fixed_W computed above)

            # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
            # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
            def preprocess_tensor(x):
                # Keep these values as Python constants in the traced graph.
                # Deriving them from x.shape serializes shape-to-int ops, which
                # are not CUDA-graph capturable in LibTorch.
                rh, rw = height / fixed_H, width / fixed_W

                x = F.interpolate(x, (fixed_H, fixed_W), mode='bilinear', align_corners=False)
                return x, rh, rw

            # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
            # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
            def NMS(x, threshold = 0.05, kernel_size = 5, nvalid = int(1.5 * top_k)):
                B, _, H, W = x.shape
                pad=kernel_size//2
                local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
                xTot = x * (x == local_max) * (x > threshold)
                xOut, pos1d = torch.topk(xTot.view(B, -1), nvalid)
                pos2d = torch.zeros((B, nvalid, 2), dtype=torch.long, device=x.device)
                pos2d[..., 0] = pos1d % W
                pos2d[..., 1] = pos1d // W
                return xOut, pos2d

            def detectAndCompute(x):
                return Detector._detect_and_compute(x, extractor, use_rotated_descriptors)

            def fixed_unfold2d(self, x, ws=2):
                # XFeat calls this only on the normalized grayscale input to build keypoint logits
                # Keep reshape sizes constant for CUDA graph
                B = 2 if use_rotated_descriptors else batch
                H_ws = fixed_H // ws
                W_ws = fixed_W // ws
                x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, 1, H_ws, W_ws, ws**2)
                return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H_ws, W_ws)

            extractor.net._unfold2d = types.MethodType(fixed_unfold2d, extractor.net)
            extractor.preprocess_tensor = preprocess_tensor
            extractor.NMS = NMS
            extractor.forward = detectAndCompute

            extractor = torch.jit.trace(extractor, [dummy_img])
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.jit.save(extractor, cache_path)
            extractor = torch.jit.load(cache_path)

        for p in extractor.parameters():
            p.requires_grad_(False)
        if not torch.version.hip:
            extractor = torch.cuda.make_graphed_callables(
                extractor, [dummy_img], allow_unused_input=True
            )
        self.extractor = extractor

    @torch.no_grad()
    def __call__(self, image):
        """Detect keypoints + descriptors for an image batch.

        Expects ``[B, 3, H, W]`` with ``B == self.batch``. Returns a
        ``DescribedKeypoints`` with ``kpts: [N, 2]`` and
        ``full_feats: [Norient, N, C]``.
        """
        assert image.dim() == 4, f"Detector expects [B,3,H,W], got {tuple(image.shape)}"
        assert image.shape[0] == self.batch, (
            f"Detector configured for batch={self.batch}, got input batch={image.shape[0]}"
        )
        mkpts, feats = self.extractor(image.half())  # [B,N,2], [Norient,B,N,C]
        return DescribedKeypoints(mkpts[0].clone(), feats[:, 0].clone())
