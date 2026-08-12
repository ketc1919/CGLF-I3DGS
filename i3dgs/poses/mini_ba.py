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
import cupy

class MiniBAInternal(nn.Module):
    def __init__(
        self,
        n_cams,
        npts,
        n_cams_per_pt,
        optimize_focal,
        optimize_3Dpts,
        huber_delta,
        outlier_mad_scale,
        outlier_threshold,
        lm,
        ep,
        k,
        iters,
    ):
        super().__init__()
        self.n_cams = n_cams
        self.npts = npts
        self.n_cams_per_pt = n_cams_per_pt
        self.optimize_focal = optimize_focal
        self.optimize_3Dpts = optimize_3Dpts
        self.huber_delta = huber_delta
        self.outlier_mad_scale = outlier_mad_scale
        self.outlier_threshold = outlier_threshold
        self.lm = lm
        self.ep = ep
        self.k = k
        self.iters = iters

        self.n_intrinsics = optimize_focal
        self.n_cam_params = n_cams * 6 + self.n_intrinsics

        # Sparse if not many camera (dense assembly is not implemented when optimizing other than the pose)
        self.use_sparse = self.n_cams > 2 or self.optimize_3Dpts or self.optimize_focal

        # kthvalue indices (Python ints, 1-based) for MAD outlier rejection.
        # Computed here (before CUDA graph capture) assuming all observations valid,
        # which matches the all-positive dummy uv used during make_graphed_callables.
        n_obs = npts * n_cams_per_pt
        self.median_k = max(1, round(0.5 * (n_obs - 1)) + 1)
        self.q_k = max(1, round(0.85 * (n_obs - 1)) + 1)

        # Read the CUDA source code and set the include directory to poses/
        with open("poses/mini_ba_utils.cu", "r") as f:
            cuda_source = f.read()

        # With Jacobians
        utils_module = cupy.RawModule(
            code=cuda_source,
            options=("--std=c++11", "-Iposes",
                     "-DCOMPUTE_J_POSE=1",
                     f"-DCOMPUTE_J_F={int(self.optimize_focal)}",
                     f"-DCOMPUTE_J_XYZ={int(self.optimize_3Dpts)}",
                    ),
        )
        self.project_and_jacobian_kernel_cuda = utils_module.get_function("project_and_jacobian_kernel_cuda")

        # Kernel without Jacobian computation (for residuals)
        utils_module_nojac = cupy.RawModule(
            code=cuda_source,
            options=("--std=c++11", "-Iposes",
                     "-DCOMPUTE_J_POSE=0", "-DCOMPUTE_J_F=0", "-DCOMPUTE_J_XYZ=0"
                    ),
        )
        self.project_kernel_cuda = utils_module_nojac.get_function("project_and_jacobian_kernel_cuda")
        self.update_poses_kernel = utils_module.get_function("update_poses_kernel")
        self.inv_3x3_kernel_cuda = utils_module.get_function("inv_3x3_kernel")

    def project_and_jacobian_cuda(self, xyz, poses_matrix, f, centre, cam_indices, with_jacobian):
        output = torch.zeros((*xyz.shape[:-1], self.n_cams_per_pt, 2), device=xyz.device, dtype=xyz.dtype)
        J_se3  = torch.zeros((0), device=xyz.device, dtype=xyz.dtype)
        J_xyz = torch.zeros((0), device=xyz.device, dtype=xyz.dtype)
        J_intrinsics = torch.zeros((0), device=xyz.device, dtype=xyz.dtype)

        if with_jacobian:
            J_se3  = torch.zeros((*xyz.shape[:-1], self.n_cams_per_pt, 2, self.n_cams, 6), device=xyz.device, dtype=xyz.dtype)
            if self.optimize_3Dpts:
                J_xyz = torch.zeros((*xyz.shape[:-1], self.n_cams_per_pt, 2, 3), device=xyz.device, dtype=xyz.dtype)
            if self.n_intrinsics > 0:
                J_intrinsics = torch.zeros((*xyz.shape[:-1], self.n_cams_per_pt, 2, self.n_intrinsics), device=xyz.device, dtype=xyz.dtype)
            cuda_kernel = self.project_and_jacobian_kernel_cuda
        else:
            cuda_kernel = self.project_kernel_cuda

        poses_flat = poses_matrix.reshape(-1, 12)

        # Launch the kernel in the same stream as pytorch to ensure proper graph capture
        torch_stream = torch.cuda.current_stream().cuda_stream
        cupy_stream = cupy.cuda.ExternalStream(torch_stream)
        with cupy_stream:
            n_total = xyz.shape[0] * self.n_cams_per_pt
            grid = ((n_total + 255) // 256,)
            block = (256,)
            cuda_kernel(
                grid=grid,
                block=block,
                args=(
                    xyz.contiguous().data_ptr(),
                    poses_flat.contiguous().data_ptr(),
                    f.contiguous().data_ptr(),
                    centre.contiguous().data_ptr(),
                    cam_indices.contiguous().data_ptr(),
                    output.data_ptr(),
                    J_se3.data_ptr(),
                    J_xyz.data_ptr(),
                    J_intrinsics.data_ptr(),
                    xyz.shape[0],
                    self.n_cams,
                    self.n_cams_per_pt,
                ),
            )

        return output, J_se3, J_xyz, J_intrinsics

    def inv_3x3(self, A):
        """Batched 3x3 inverse via CUDA kernel (cofactor/determinant). CUDA-graph safe."""
        N = A.shape[0]
        A_inv = torch.empty_like(A)
        torch_stream = torch.cuda.current_stream().cuda_stream
        cupy_stream = cupy.cuda.ExternalStream(torch_stream)
        with cupy_stream:
            block = (256,)
            grid = ((N + 255) // 256,)
            self.inv_3x3_kernel_cuda(
                grid=grid, block=block,
                args=(
                    A.contiguous().data_ptr(),
                    A_inv.data_ptr(),
                    N,
                ),
            )
        return A_inv

    def _assemble_jtj(self, xyz, poses, f, centre, cam_indices, valid_per_obs, uv, fixed_mask):
        """Assemble the (Schur-structured) normal equations in PyTorch.

        Per-observation Jacobians are computed with project_and_jacobian_cuda, then
        weighted by w = valid * huber (applied to both J and r, so the contribution
        of each observation is w^2) and assembled into the camera/point blocks via
        dense/batched matmuls. The Jacobian of cameras flagged in fixed_mask is
        zeroed so they receive no update. Returns the same tuple layout the LM loop
        consumes; total_resid = Σ w * (|ru| + |rv|) matches the LM acceptance cost.

        valid_per_obs: float tensor (0/1), one entry per observation.
        fixed_mask: bool tensor [n_cams], True for cameras held fixed this iteration.
        """
        n_pts = xyz.shape[0]
        ncp_pt = self.n_cams_per_pt
        ncp = self.n_cam_params

        projected, duv_dcam, duv_dxyz, duv_df = self.project_and_jacobian_cuda(
            xyz, poses, f, centre, cam_indices, with_jacobian=True)
        r_in = projected - uv                                   # [n_pts, ncp_pt, 2]

        # Per-observation weight: validity mask times (optional) Huber down-weighting,
        # computed from the raw residual exactly as the removed kernel did.
        w = valid_per_obs.view(n_pts, ncp_pt)
        l1 = r_in.abs().sum(dim=-1)                             # |ru| + |rv|
        if self.huber_delta > 0:
            err = r_in.norm(dim=-1).clamp(min=1e-6)
            w = w * (self.huber_delta / err).clamp(max=1.0)
        total_resid = (w * l1).sum().view(1)

        # Fixed cameras get no pose update: zero their (already block-sparse) columns.
        free = (~fixed_mask).to(duv_dcam.dtype).view(1, 1, 1, self.n_cams, 1)
        duv_dcam = (duv_dcam * free).reshape(n_pts, ncp_pt, 2, -1)
        if self.optimize_focal:
            duv_dcam = torch.cat([duv_dcam, duv_df], dim=-1)

        # Weight the Jacobian rows and residuals (w on both -> w^2 in the products).
        w_row = w.view(n_pts, ncp_pt, 1, 1)
        duv_dcam = duv_dcam * w_row
        r = (r_in * w.view(n_pts, ncp_pt, 1)).reshape(-1)

        duv_dcam_flat = duv_dcam.reshape(-1, ncp)
        jtj_cam = duv_dcam_flat.T @ duv_dcam_flat
        jacxr_cam = duv_dcam_flat.T @ r

        if self.optimize_3Dpts:
            duv_dcam_reshaped = duv_dcam.reshape(-1, ncp_pt * 2, ncp)
            duv_dxyz = (duv_dxyz * w_row).reshape(-1, ncp_pt * 2, 3)
            jtj_xyz = torch.bmm(duv_dxyz.transpose(-1, -2).contiguous(), duv_dxyz)
            jtj_cam_xyz = (
                torch.bmm(duv_dcam_reshaped.transpose(1, 2), duv_dxyz)
                .transpose(1, 2)
                .reshape(-1, ncp)
            )
            jacxr_xyz = torch.bmm(
                duv_dxyz.transpose(1, 2),
                r.view(-1, ncp_pt * 2, 1),
            ).view(-1)
            return jtj_cam, jacxr_cam, jtj_xyz, jacxr_xyz, jtj_cam_xyz, total_resid
        return jtj_cam, jacxr_cam, total_resid

    def update_poses_fused(self, poses, dpose):
        """
        Fused kernel for: poses_new = exp(-dpose) * poses
        poses: [B, N_cams, 3, 4] or [N_cams, 3, 4]
        dpose: [B, N_cams, 6] or [N_cams, 6]
        """
        # Ensure flat views
        poses_flat = poses.view(-1, 12)
        dpose_flat = dpose.view(-1, 6)
        n_cams = poses_flat.shape[0]
        
        poses_out = torch.zeros_like(poses_flat)
        
        torch_stream = torch.cuda.current_stream().cuda_stream
        cupy_stream = cupy.cuda.ExternalStream(torch_stream)
        
        with cupy_stream:
            block = (256,)
            grid = ((n_cams + 255) // 256,)
            
            self.update_poses_kernel(
                grid=grid, block=block,
                args=(
                    poses_flat.contiguous().data_ptr(),
                    dpose_flat.contiguous().data_ptr(),
                    poses_out.data_ptr(),
                    n_cams
                )
            )
            
        return poses_out.view_as(poses)

    def get_mask(self, r_in, original_mask2):
        err = torch.linalg.vector_norm(
            r_in.view(-1, 2) * original_mask2.view(-1)[:, None],
            dim=-1,
        )
        mask = err < self.outlier_threshold
        mask = original_mask2 * mask.view(*original_mask2.shape)

        if self.outlier_mad_scale > 0:
            sorted_err, _ = torch.sort(err)
            med = sorted_err[self.median_index]
            ad = (err - med).abs()
            ad = torch.where(err != 0, ad, 0)
            sorted_ad, _ = torch.sort(ad)
            mad = sorted_ad[self.median_index]
            c = med + self.outlier_mad_scale * mad
            c = torch.max(c, sorted_err[self.q_index])
            # c = 5
            mask *= (err.view(*original_mask2.shape) < c)
            
        return mask[..., None].expand(-1, -1, 2).reshape(-1).float()
    
    def _build_observations(self, poses, f, xyz, centre, uv, cam_indices, original_mask2):
        lm = self.lm
        eff_iters = torch.zeros(1, device=f.device, dtype=torch.int)
        initial_r = torch.zeros_like(uv).view(-1)

        # Sparse path: MAD mask computed once pre-loop → valid_per_obs (binary, frozen).
        # When huber_delta > 0, the kernel also applies Huber weights on top, softening
        # the boundary for observations near the MAD threshold.
        valid_per_obs = None
        current_r = None
        if self.use_sparse:
            projected_init, _, _, _ = self.project_and_jacobian_cuda(
                xyz, poses, f, centre, cam_indices, with_jacobian=False)
            initial_r = (projected_init - uv).view(-1)
            weights = self.get_mask(initial_r, original_mask2)
            valid_per_obs = weights.view(-1, 2)[:, 0]
            current_r = initial_r  # tracks last accepted residuals for mask refresh

        return lm, eff_iters, initial_r, valid_per_obs, current_r

    def _run_lm_iterations(self, poses, f, xyz, centre, uv, cam_indices, fixed_masks, original_mask2, lm, eff_iters, initial_r, valid_per_obs, current_r):
        for iteration in range(self.iters):
            if self.use_sparse:
                # Normal equations assembled in PyTorch from the per-observation
                # Jacobians; Huber weights from each residual. total_resid =
                # Σ valid * w_huber * (|ru|+|rv|); used for the LM acceptance check.
                if self.optimize_3Dpts:
                    jtj_cam, jacxr_cam, jtj_xyz, jacxr_xyz, jtj_cam_xyz, total_resid = \
                        self._assemble_jtj(xyz, poses, f, centre,
                                           cam_indices, valid_per_obs, uv, fixed_masks[iteration])

                    # Damping
                    jtj_cam.diagonal().mul_(1 + lm)
                    jtj_cam.diagonal().clamp_min_(3e7 * self.ep)
                    jtj_xyz.diagonal(dim1=-2, dim2=-1).mul_(1 + lm)
                    jtj_xyz.diagonal(dim1=-2, dim2=-1).clamp_min_(10 * self.ep)

                    # Solve with Schur Complement
                    jtj_xyz_inv = self.inv_3x3(jtj_xyz)

                    BD = (
                        torch.bmm(jtj_xyz_inv, jtj_cam_xyz.view(-1, 3, self.n_cam_params))
                        .view(-1, self.n_cam_params)
                        .T
                    )
                    BmECm1Et = jtj_cam - BD @ jtj_cam_xyz

                    L_cam = torch.linalg.cholesky_ex(BmECm1Et)[0]
                    vmECm1w = jacxr_cam - BD @ jacxr_xyz
                    dcam = torch.linalg.solve_triangular(L_cam, vmECm1w.unsqueeze(-1), upper=False)
                    dcam = torch.linalg.solve_triangular(L_cam.mT, dcam, upper=True).squeeze(-1)
                    dcam.nan_to_num_()

                    # 3D Points Update
                    b = jacxr_xyz - jtj_cam_xyz @ dcam
                    dxyz = torch.bmm(b.view(-1, 1, 3), jtj_xyz_inv).view(xyz.shape)
                    dxyz.nan_to_num_()
                else:
                    jtj_cam, jacxr_cam, total_resid = self._assemble_jtj(
                        xyz, poses, f, centre,
                        cam_indices, valid_per_obs, uv, fixed_masks[iteration])
                    jtj_cam.diagonal().mul_(1 + lm)
                    jtj_cam.diagonal().clamp_min_(3e7 * self.ep)
                    dcam = torch.linalg.inv_ex(jtj_cam)[0] @ jacxr_cam
                    dcam.nan_to_num_()
                    dxyz = 0
            else:
                # Dense path: single projection with Jacobian.
                projected, duv_dcam, duv_dxyz, duv_df = self.project_and_jacobian_cuda(
                    xyz, poses, f, centre, cam_indices, with_jacobian=True)
                r_in = (projected - uv).view(-1)
                if iteration == 0:
                    initial_r = r_in.clone()
                weights = self.get_mask(r_in, original_mask2)
                r = r_in * weights.view(-1)
                duv_dcam = duv_dcam.view(*duv_dcam.shape[:-2], -1)
                duv_dcam_flat = duv_dcam.view(-1, self.n_cam_params)
                w_2d = weights.view(-1, 1)
                if self.huber_delta > 0:
                    r_norm = r.view(-1, 2).norm(dim=1).clamp(min=1e-6)
                    huber_w = (self.huber_delta / r_norm).clamp(max=1.0).repeat_interleave(2)
                    r = r * huber_w
                    w_2d = w_2d * huber_w.view(-1, 1)
                jacxr_cam = duv_dcam_flat.T @ r
                jtj_cam = duv_dcam_flat.T @ (duv_dcam_flat * w_2d)
                jtj_cam.diagonal().mul_(1 + lm)
                jtj_cam.diagonal().clamp_min_(3e7 * self.ep)
                dcam = torch.linalg.inv_ex(jtj_cam)[0] @ jacxr_cam
                dcam.nan_to_num_()
                dxyz = 0

            dpose = dcam[: self.n_cams * 6].view(self.n_cams, 6)
            dintrinsics = dcam[self.n_cams * 6:]
            df = dintrinsics[0] if self.optimize_focal else 0

            # Manifold update (Fused Kernel)
            poses_tmp = self.update_poses_fused(poses, dpose)

            f_tmp = f.clone() - df
            xyz_tmp = xyz - dxyz

            projected, _, _, _ = self.project_and_jacobian_cuda(
                xyz_tmp, poses_tmp, f_tmp, centre, cam_indices, with_jacobian=False)
            new_r = (projected - uv).view(-1)

            # Check if the update improves residuals
            if self.use_sparse:
                # Candidate Huber cost: Σ valid * w_huber(new_r) * (|ru|+|rv|)
                # Consistent with total_resid accumulated in the kernel.
                new_r_2d = new_r.view(-1, 2)
                new_l1 = new_r_2d.abs().sum(dim=1)
                if self.huber_delta > 0:
                    new_err = new_r_2d.norm(dim=1)
                    new_w = (self.huber_delta / new_err.clamp(min=1e-6)).clamp(max=1.0)
                    new_resid = (valid_per_obs * new_w * new_l1).sum()
                else:
                    new_resid = (valid_per_obs * new_l1).sum()
                success_mask = (new_resid < total_resid.squeeze()) * (f_tmp > 0)
            else:
                new_r = new_r * weights
                if self.huber_delta > 0:
                    new_r_norm = new_r.view(-1, 2).norm(dim=1).clamp(min=1e-6)
                    new_huber_w = (self.huber_delta / new_r_norm).clamp(max=1.0).repeat_interleave(2)
                    new_r = new_r * new_huber_w
                success_mask = ((new_r).abs().mean() < (r).abs().mean()) * (f_tmp > 0)

            # Graph Safe Updates (No 'if' on tensors)
            poses = torch.where(success_mask, poses_tmp, poses)
            f = torch.where(success_mask, f_tmp, f)
            xyz = torch.where(success_mask, xyz_tmp, xyz)

            # Update LM damping and stats
            lm = torch.where(success_mask, lm / self.k, lm * self.k)
            eff_iters = eff_iters + success_mask.int()

            # Sparse path: keep current_r pointing at the accepted residuals,
            # then periodically recompute the outlier mask from them.
            if self.use_sparse:
                current_r = torch.where(success_mask, new_r, current_r)
                if iteration % 2 == 0:
                    weights = self.get_mask(current_r, original_mask2)
                    valid_per_obs = weights.view(-1, 2)[:, 0]

        return poses, f, xyz, initial_r, eff_iters

    def optimize(self, Rts, f, xyz, centre, uv, cam_indices, fixed_masks):
        poses = Rts[..., :3, :4].contiguous()

        original_mask2 = (uv >= 0).all(dim=-1)
        q = 1 - 0.5 * original_mask2.float().mean()
        self.median_index = (q * (original_mask2.numel() - 1))[None].long().clamp(0, original_mask2.numel() - 1)
        q = 1 - 0.15 * original_mask2.float().mean()
        self.q_index = (q * (original_mask2.numel() - 1))[None].long().clamp(0, original_mask2.numel() - 1)

        lm, eff_iters, initial_r, valid_per_obs, current_r = self._build_observations(
            poses, f, xyz, centre, uv, cam_indices, original_mask2)

        poses, f, xyz, initial_r, eff_iters = self._run_lm_iterations(
            poses, f, xyz, centre, uv, cam_indices, fixed_masks,
            original_mask2, lm, eff_iters, initial_r, valid_per_obs, current_r)

        n = poses.shape[0]
        out_Rts = poses.new_zeros(n, 4, 4)
        out_Rts[:, :3, :] = poses
        out_Rts[:, 3, 3] = 1.0

        # Final Residual
        projected, _, _, _ = self.project_and_jacobian_cuda(
            xyz, poses, f, centre, cam_indices, with_jacobian=False)
        r = (projected - uv).view(-1)
        mask = self.get_mask(r, original_mask2)

        return out_Rts, f, xyz, r, initial_r, mask, eff_iters

    def forward(self, Rts, f, xyz, centre, uv, cam_indices, fixed_masks):
        return self.optimize(Rts, f, xyz, centre, uv, cam_indices, fixed_masks)

class MiniBA:
    @torch.no_grad()
    def __init__(
        self,
        n_cams,
        npts,
        n_cams_per_pt,
        optimize_focal,
        optimize_3Dpts,
        make_cuda_graph=True,
        huber_delta=0.0,
        outlier_mad_scale=-1.0,
        outlier_threshold=5.0,
        lm=1e-5,
        ep=1e-7,
        k=2.0,
        iters=200,
    ):
        """
        Initializes the MiniBA optimizer for camera poses, 3D points and focal length.

        Parameters:
        - n_cams: Total number of cameras.
        - npts: Number of 3D points.
        - n_cams_per_pt: Number of cameras supervising each 3D point.
        - optimize_focal: Optimize focal length if True.
        - optimize_3Dpts: Optimize 3D points if True.
        - make_cuda_graph: Whether to use CUDA graphs for optimization.
        - huber_delta:
        - outlier_mad_scale: If > 0, will not concider points with error > median(error) + outlier_mad_scale * MAD(error)
        - outlier_threshold: Hard threshold on reprojection error for initial outlier rejection.
        - lm: Initial Levenberg-Marquardt damping term.
        - ep: Regularization term to avoid singularities.
        - k: Damping adjustment factor.
        - iters: Number of optimization iterations.
        """

        self.optimizer = MiniBAInternal(
            n_cams,
            npts,
            n_cams_per_pt,
            optimize_focal,
            optimize_3Dpts,
            huber_delta,
            outlier_mad_scale,
            outlier_threshold,
            lm,
            ep,
            k,
            iters,
        )
        self.optimizer = self.optimizer.eval().cuda()
        self.n_cams = n_cams
        self.iters = iters

        Rts_init = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        f_init = torch.randn(1, device="cuda")
        xyz_init = torch.randn(npts, 3, device="cuda")
        centre = torch.randn(2, device="cuda")
        uv = torch.randn(npts, n_cams_per_pt, 2, device="cuda").abs()
        cam_indices = torch.randint(0, n_cams, (npts, n_cams_per_pt), device="cuda").int()
        fixed_masks_init = torch.zeros(iters, n_cams, dtype=torch.bool, device="cuda")

        # cholesky_ex does not support HIP stream capture, so CUDA graphs are unavailable on AMD GPUs
        if make_cuda_graph and not torch.version.hip:
            self.optimizer = torch.cuda.make_graphed_callables(
                self.optimizer, (Rts_init, f_init, xyz_init, centre, uv, cam_indices, fixed_masks_init), 2
            )

    @torch.no_grad()
    def __call__(self, Rts, f, xyz, centre, uv, cam_indices=None, fixed_masks=None):
        if cam_indices is None:
            assert self.optimizer.n_cams == 1
            cam_indices = torch.zeros(
                self.optimizer.npts, 1, dtype=torch.int, device=uv.device)
        else:
            cam_indices = cam_indices.to(device=uv.device, dtype=torch.int32)
        if fixed_masks is None:
            fixed_masks = torch.zeros(self.iters, self.n_cams, device=f.device, dtype=torch.bool)

        return self.optimizer(Rts, f, xyz, centre, uv, cam_indices, fixed_masks)
