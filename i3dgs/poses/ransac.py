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
import cupy
import math
from enum import Enum
from utils import depth2points


class EstimatorType(Enum):
    FUNDAMENTAL_8PTS = 0
    P3P = 1


class RANSACEstimator:
    @torch.no_grad()
    def __init__(self, N: int, max_error: float, type: EstimatorType):
        """
        Initialize the RANSAC estimator.

        Args:
            N (int): Number of models to estimate.
            max_error (float): Maximum reprojection error for inliers.
            type (EstimatorType): Type of estimator to use.
        """
        self.N = N
        self.max_error = max_error
        self.type = type

        # Sampling threshold for inlier mask computation to reduce memory
        self.max_pts_inlier_mask = 2 * N  # Sample at most 2xN points

        # Set the functions and number of points required for each estimator
        if type == EstimatorType.FUNDAMENTAL_8PTS:
            # Read the CUDA source code and set the include directory to poses/
            with open("poses/ransac.cu", "r") as f:
                cuda_source = f.read()
            self.module = cupy.RawModule(
                code=cuda_source,
                options=("--std=c++17", "-Iposes", "-I/opt/rocm/include"),
            )
            self.model_estimator = self.module.get_function("batchFundMat8pts")
            self.fundmat_select_best_fn   = self.module.get_function("fundmat_select_best")
            self.fundmat_inliers_single_fn = self.module.get_function("fundmat_inliers_single")
            self.m = 8  # 8 pairs are required to estimate a fundamental matrix
            self.models = torch.zeros([N, 3, 3], device=torch.device("cuda"))
            self.valid_model_mask = torch.zeros(
                N, dtype=torch.bool, device=torch.device("cuda")
            )
            self.fundmat_counts_buf = torch.zeros(N, dtype=torch.int32, device="cuda")
            self.inliers_buffer = torch.zeros([N, self.max_pts_inlier_mask], dtype=torch.bool, device="cuda")
        elif type == EstimatorType.P3P:
            self.m = 3  # 3 3D-2D correspondences per minimal sample
            with open("poses/p3p.cu", "r") as f:
                cuda_source = f.read()
            self.module = cupy.RawModule(
                code=cuda_source,
                options=("--std=c++17", "-Iposes", "-I/opt/rocm/include"),
            )
            self.model_estimator = self.module.get_function("batchP3P")
            self.p3p_select_best_fn    = self.module.get_function("p3p_select_best")
            self.p3p_inliers_single_fn = self.module.get_function("p3p_inliers_single")

            # Pre-allocate buffers to avoid fragmentation (cached across calls).
            # batchP3P returns up to 4 (R, t) solutions per hypothesis.
            self.sol_counts = torch.zeros(N, dtype=torch.int32, device="cuda")
            self.valid_models = torch.zeros(N, dtype=torch.bool, device="cuda")
            self.Rs_buffer = torch.zeros([N * 4, 3, 3], device="cuda")
            self.ts_buffer = torch.zeros([N * 4, 3], device="cuda")

            # Per-hypothesis best (R,t,count), written by p3p_select_best
            self.best_Rs_buf     = torch.zeros([N, 3, 3], device="cuda")
            self.best_Ts_buf     = torch.zeros([N, 3],    device="cuda")
            self.best_counts_buf = torch.zeros(N, dtype=torch.int32, device="cuda")
            self.best_model_buf  = torch.zeros([3, 4],   device="cuda")

            # Final bool inlier mask for all points (resized on demand)
            self.inliers_final_buf = torch.zeros(self.max_pts_inlier_mask, dtype=torch.bool, device="cuda")
        else:
            raise ValueError(f"Unknown EstimatorType {type}")

        self.inliers_single_buffer = torch.zeros(self.max_pts_inlier_mask, dtype=torch.bool, device="cuda")

    def estimate(
        self,
        mkpts1: torch.Tensor,
        mkpts2: torch.Tensor,
        focal: torch.Tensor,
        centre: torch.Tensor,
    ):
        """
        Estimate N models from the given matches.
        """
        if self.type == EstimatorType.FUNDAMENTAL_8PTS:
            block_size = 64
            grid_size = math.ceil(self.N / block_size)
            cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
            with cupy_stream:
                self.model_estimator(
                    block=(block_size,),
                    grid=(grid_size,),
                    args=(
                        mkpts1.data_ptr(),
                        mkpts2.data_ptr(),
                        self.models.data_ptr(),
                        self.valid_model_mask.data_ptr(),
                        self.N,
                        mkpts1.shape[0],
                    ),
                )
        elif self.type == EstimatorType.P3P:
            pass

    def _estimate_p3p(self, mkpts1, mkpts2, focal, centre):
        n_pts    = mkpts1.shape[0]
        focal_v  = float(focal.squeeze()) if hasattr(focal, 'squeeze') else float(focal)
        cx_v     = float(centre[0])
        cy_v     = float(centre[1])

        # --- Step 1: generate up to 4 (R,t) per hypothesis ---
        block_size = 64
        grid_size  = math.ceil(self.N / block_size)
        self.Rs_buffer.zero_()
        self.ts_buffer.zero_()

        mkpts13D = depth2points(mkpts1, 1, focal, centre)
        mkpts13D = mkpts13D / torch.linalg.norm(mkpts13D, dim=-1, keepdim=True)
        cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with cupy_stream:
            self.model_estimator(
                (block_size,),
                (grid_size,),
                (
                    mkpts13D.contiguous().data_ptr(),
                    mkpts2.contiguous().data_ptr(),
                    self.Rs_buffer.data_ptr(),
                    self.ts_buffer.data_ptr(),
                    self.sol_counts.data_ptr(),
                    self.valid_models.data_ptr(),
                    self.N,
                    n_pts,
                ),
            )

        # --- Step 2: for each hypothesis pick the best of 4 solutions ---
        # One block per hypothesis, 128 threads (4 warps × 32).
        # Warp w counts inliers for solution (hyp*4+w) via stride loop +
        # __shfl_down_sync warp reduction; thread 0 writes winner.
        mkpts2_c = mkpts2.contiguous()
        mkpts1_c = mkpts1.contiguous()
        cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with cupy_stream:
            self.p3p_select_best_fn(
                (self.N,),          # grid: N blocks
                (128,),             # block: 4 warps
                (
                    n_pts,
                    mkpts2_c.data_ptr(),
                    mkpts1_c.data_ptr(),
                    self.Rs_buffer.data_ptr(),
                    self.ts_buffer.data_ptr(),
                    cupy.float32(focal_v),
                    cupy.float32(cx_v),
                    cupy.float32(cy_v),
                    cupy.float32(self.max_error ** 2),
                    self.best_Rs_buf.data_ptr(),
                    self.best_Ts_buf.data_ptr(),
                    self.best_counts_buf.data_ptr(),
                    self.N,
                ),
            )

        # One sync to read the winning hypothesis index
        best_id = int(self.best_counts_buf.argmax())

        # --- Step 3: compute final inlier mask for ALL points ---
        if n_pts > self.inliers_final_buf.shape[0]:
            self.inliers_final_buf = torch.zeros(n_pts, dtype=torch.bool, device="cuda")
        inliers_out = self.inliers_final_buf[:n_pts]

        cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with cupy_stream:
            self.p3p_inliers_single_fn(
                (math.ceil(n_pts / 256),),
                (256,),
                (
                    n_pts,
                    mkpts2_c.data_ptr(),
                    mkpts1_c.data_ptr(),
                    self.best_Rs_buf[best_id].data_ptr(),
                    self.best_Ts_buf[best_id].data_ptr(),
                    cupy.float32(focal_v),
                    cupy.float32(cx_v),
                    cupy.float32(cy_v),
                    cupy.float32(self.max_error ** 2),
                    inliers_out.data_ptr(),
                ),
            )

        self.best_model_buf[:, :3] = self.best_Rs_buf[best_id]
        self.best_model_buf[:, 3]  = self.best_Ts_buf[best_id]
        return self.best_model_buf, inliers_out

    def _estimate_fundamental(self, mkpts1, mkpts2, focal, centre):
        # Estimate N fundamental matrices
        self.estimate(mkpts1, mkpts2, focal, centre)

        n_pts = mkpts1.shape[0]
        thr   = cupy.float32(self.max_error ** 2)

        # Count inliers per hypothesis -- N blocks, 256 threads, no bool array
        self.fundmat_counts_buf.zero_()
        cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with cupy_stream:
            self.fundmat_select_best_fn(
                (self.N,),
                (256,),
                (
                    mkpts1.contiguous().data_ptr(),
                    mkpts2.contiguous().data_ptr(),
                    self.models.data_ptr(),
                    self.valid_model_mask.data_ptr(),
                    self.fundmat_counts_buf.data_ptr(),
                    thr,
                    self.N,
                    n_pts,
                ),
            )
        best_id = int(self.fundmat_counts_buf.argmax())

        # Final inlier mask for the best model
        if n_pts > self.inliers_single_buffer.shape[0]:
            self.inliers_single_buffer = torch.zeros(n_pts, dtype=torch.bool, device="cuda")
        inliers_out = self.inliers_single_buffer[:n_pts]
        cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with cupy_stream:
            self.fundmat_inliers_single_fn(
                (math.ceil(n_pts / 256),),
                (256,),
                (
                    mkpts1.contiguous().data_ptr(),
                    mkpts2.contiguous().data_ptr(),
                    self.models[best_id:best_id + 1].data_ptr(),
                    thr,
                    n_pts,
                    inliers_out.data_ptr(),
                ),
            )
        return self.models[best_id], inliers_out

    @torch.no_grad()
    def __call__(
        self,
        mkpts1,
        mkpts2,
        focal=None,
        centre=None,
        confs=None,
    ):
        """
        Run the RANSAC estimator to find the best model and inliers.
        args:
            mkpts1: ``[n, 2]`` pixel observations.
            mkpts2: ``[n, 2]`` for FUNDAMENTAL_8PTS; ``[n, 3]`` 3D world points for P3P.
            focal: ``[1]`` focal length (P3P only).
            centre: ``[2]`` principal point (P3P only).
            confs: per-correspondence confidence (currently unused at call time).

        returns:
            best_model: ``[3, 4]`` (P3P) or ``[3, 3]`` (FUNDAMENTAL).
            mask: ``[n]`` boolean inlier mask.
        """

        if self.type == EstimatorType.P3P:
            assert focal is not None
            assert centre is not None
            return self._estimate_p3p(mkpts1, mkpts2, focal, centre)
        else:
            return self._estimate_fundamental(mkpts1, mkpts2, focal, centre)
