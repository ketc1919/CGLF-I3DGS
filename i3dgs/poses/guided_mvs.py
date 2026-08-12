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
from typing import List
import cupy
import torch
from scene.keyframe import Keyframe

class GuidedMVS():
    @torch.no_grad()
    def __init__(self, args, num_depth_candidates=16):
        self.n_cams = args.num_prev_keyframes_miniba_incr
        self.num_depth_candidates = num_depth_candidates
        self.idepth_range = 2e-1

        # Read the CUDA source code and set the include directory to poses/
        with open('poses/guided_mvs.cu', 'r') as f:
            cuda_source = f.read()
        cuda_source = cuda_source.replace("NUM_CAMS", str(self.n_cams))
        cuda_source = cuda_source.replace("NUM_DEPTH_CANDIDATES", str(num_depth_candidates))
        self.module = cupy.RawModule(
            code=cuda_source, 
            options=('--std=c++17', '-Iposes', "-I/opt/rocm/include"),
        )
        self.uvToDepth = self.module.get_function("uvToDepth")

    @torch.no_grad()
    def __call__(self,
                 uv: torch.Tensor,
                 refKeyframe: Keyframe,
                 keyframes: List[Keyframe],
                 mono_idepth: torch.Tensor,
    ):
        n_cams = len(keyframes)
        if n_cams > self.n_cams:
            keyframes = keyframes[:self.n_cams]
            n_cams = self.n_cams

        uv = uv.contiguous()
        ref_Rt = refKeyframe.get_Rt()
        other2ref = [kf.get_Rt() @ torch.linalg.inv(ref_Rt) for kf in keyframes]
        other2ref = torch.stack(other2ref, dim=0)[..., :3, :4].contiguous()
        refFeatMap = refKeyframe.feat_map.contiguous()
        featMaps = torch.stack(
            [kf.feat_map.cuda().contiguous() for kf in keyframes],
            dim=0,
        )
        intrinsics = torch.cat([refKeyframe.f, refKeyframe.centre], dim=0).contiguous()
        mono_idepth = mono_idepth.contiguous()

        depth = -torch.ones_like(uv[..., 0]).contiguous()
        idist = -torch.ones_like(uv[..., 0]).contiguous()

        block_size = self.num_depth_candidates
        grid_size = math.ceil(uv.shape[0])
        self.uvToDepth(
            block=(block_size,),
            grid=(grid_size,),
            args=(
                uv.data_ptr(),
                refFeatMap.data_ptr(),
                featMaps.data_ptr(),
                other2ref.data_ptr(), 
                intrinsics.data_ptr(),
                mono_idepth.data_ptr(),
                depth.data_ptr(),
                idist.data_ptr(),
                cupy.float32(self.idepth_range),
                uv.shape[0],
                n_cams,
                refFeatMap.shape[0],
                refFeatMap.shape[1],
                mono_idepth.shape[-2],
                mono_idepth.shape[-1],
                refKeyframe.height,
                refKeyframe.width,
            )
        )

        return depth, idist >= 0
