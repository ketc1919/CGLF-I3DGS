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
import cupy
import torch

from poses.feature_detector import DescribedKeypoints


class Triangulator():
    @torch.no_grad()
    def __init__(self, n_pts, n_cams, max_error, min_dis=None):
        self.n_cams   = n_cams
        self.max_error = max_error
        self.min_dis   = min_dis if min_dis is not None else max_error

        with open('poses/triangulate.cu', 'r') as f:
            cuda_source = f.read()
        self.module = cupy.RawModule(
            code=cuda_source,
            options=('--std=c++17', '-Iposes', '-I/opt/rocm/include'),
        )
        self.kernel = self.module.get_function('triangulatePoints')

    @torch.no_grad()
    def __call__(self, uv, uvs_others, Rt, Rts_others, f, centre, scale):
        """Triangulate keypoints.

        Returns ``kpts3d`` in world frame.
        """
        N = uv.shape[0]
        M = uvs_others.shape[0]

        Rts_others_inv = torch.linalg.inv(Rts_others)              # [M, 4, 4]
        rel_Rts = Rt.unsqueeze(0) @ Rts_others_inv                  # [M, 4, 4]
        rel_rts = rel_Rts[:, None, :3, :].expand(M, N, 3, 4).reshape(M * N, 12).contiguous()

        kpts3d_cam = torch.zeros(N, 3, device='cuda')
        best_dis   = torch.zeros(N, device='cuda')

        block_size = 256
        grid_size  = math.ceil(N / block_size)
        cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with cupy_stream:
            self.kernel(
                (grid_size,),
                (block_size,),
                (
                    uv.contiguous().data_ptr(),
                    uvs_others.contiguous().data_ptr(),
                    rel_rts.data_ptr(),
                    kpts3d_cam.data_ptr(),
                    best_dis.data_ptr(),
                    cupy.float32(float(f.item() if hasattr(f, 'item') else f)),
                    cupy.float32(float(centre[0].item())),
                    cupy.float32(float(centre[1].item())),
                    cupy.float32(float(self.max_error)),
                    N, M,
                )
            )

        depth  = kpts3d_cam[:, 2].clone()
        kpts3d = (kpts3d_cam - Rt[None, :3, 3]) @ Rt[:3, :3]
        return kpts3d, depth, best_dis, best_dis > (self.min_dis / scale)

    @torch.no_grad()
    def prepare_matches(self, desc_kpts: DescribedKeypoints):
        """Organise matches for triangulation.

        Selects up to ``n_cams`` neighbour keyframes by descending match count.

            uv:                [N, 2]
            uvs_others:        [M, N, 2]
            chosen_kfs_ids:    list[int]
        """
        uv = desc_kpts.kpts_flat()
        N  = uv.shape[0]

        n_matches  = torch.tensor([m.idx.shape[0] for m in desc_kpts.matches.values()])
        kf_indices = torch.tensor(list(desc_kpts.matches.keys()))
        M          = min(self.n_cams, n_matches.shape[0])
        chosen_ids = torch.topk(n_matches, M).indices
        chosen_kfs_ids = kf_indices[chosen_ids].tolist()

        uvs_others = -torch.ones(M, N, 2, device='cuda')
        for i, index in enumerate(chosen_kfs_ids):
            matches = desc_kpts.matches[index]
            uvs_others[i, matches.idx, :] = matches.kpts_other
        return uv, uvs_others, chosen_kfs_ids
