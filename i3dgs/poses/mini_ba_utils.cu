/*
 * Copyright (C) 2026, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

 // Define macros here for my linter
#ifndef COMPUTE_J_POSE
    #define COMPUTE_J_POSE 1
#endif
#ifndef COMPUTE_J_XYZ
    #define COMPUTE_J_XYZ 1
#endif
#ifndef COMPUTE_J_F
    #define COMPUTE_J_F 1
#endif
#ifndef N_CAMS
    #define N_CAMS 1
#endif
#ifndef N_CAMS_PER_PT
    #define N_CAMS_PER_PT 1
#endif

extern "C" __global__ void project_and_jacobian_kernel_cuda(
    const float* xyz_ptr,
    const float* Rt_ptr,
    const float* f_ptr,
    const float* c_ptr,
    const int* cam_indices_ptr,
    float* output_ptr,
    float* jacobian_se3_ptr,
    float* jacobian_xyz_ptr,
    float* jacobian_intrinsics_ptr,
    int n_pts, 
    int n_cams,
    int n_cams_per_pt)
{
    int obs = blockIdx.x * blockDim.x + threadIdx.x;
    if (obs >= n_pts * n_cams_per_pt) return;
    int point_idx = obs / n_cams_per_pt;
    int cam_offset = obs % n_cams_per_pt;

    int camera_idx = cam_indices_ptr[obs];
    if (camera_idx < 0 || camera_idx >= n_cams)
        return;

    // ---- Load inputs ----
    // XYZ
    const float* base_xyz = xyz_ptr + point_idx * 3;
    float x = base_xyz[0];
    float y = base_xyz[1];
    float z = base_xyz[2];

    // Pose: We now read a 3x4 Matrix directly (Row Major)
    // Layout: [r00 r01 r02 tx], [r10 r11 r12 ty], [r20 r21 r22 tz]
    const float* base_Rt = Rt_ptr + camera_idx * 12;
    float r00 = base_Rt[0]; float r01 = base_Rt[1]; float r02 = base_Rt[2]; float tx = base_Rt[3];
    float r10 = base_Rt[4]; float r11 = base_Rt[5]; float r12 = base_Rt[6]; float ty = base_Rt[7];
    float r20 = base_Rt[8]; float r21 = base_Rt[9]; float r22 = base_Rt[10]; float tz = base_Rt[11];

    float f = f_ptr[0];
    float c0 = c_ptr[0];
    float c1 = c_ptr[1];

    // ---- Transform point ----
    float p_cam_x = r00*x + r01*y + r02*z + tx;
    float p_cam_y = r10*x + r11*y + r12*z + ty;
    float p_cam_z = r20*x + r21*y + r22*z + tz;

    // ---- Project to 2D (pinhole) ----
    float inv_z = 1.0f / p_cam_z;
    // Normalized image coordinates
    float xn = p_cam_x * inv_z;
    float yn = p_cam_y * inv_z;
    float out_x = f * xn + c0;
    float out_y = f * yn + c1;

    // Store output
    int residual_base = obs;
    int base_out = residual_base * 2;
    output_ptr[base_out + 0] = out_x;
    output_ptr[base_out + 1] = out_y;

    // ==== JACOBIAN COMPUTATION ====

    // 1. Projection Jacobian: d(uv) / d(p_cam) for a pinhole camera.
    // d(xn)/d(px) = 1/pz, d(xn)/d(py) = 0, d(xn)/d(pz) = -xn/pz
    // d(yn)/d(px) = 0,    d(yn)/d(py) = 1/pz, d(yn)/d(pz) = -yn/pz
    float du_dpx = f * inv_z;
    float du_dpy = 0.0f;
    float du_dpz = f * (-xn * inv_z);

    float dv_dpx = 0.0f;
    float dv_dpy = f * inv_z;
    float dv_dpz = f * (-yn * inv_z);

    float du_drx = du_dpx;
    float du_dry = du_dpy;
    float du_drz = du_dpz;

    float dv_drx = dv_dpx;
    float dv_dry = dv_dpy;
    float dv_drz = dv_dpz;

    #if COMPUTE_J_XYZ
        // Chain rule: d(uv)/d(X_world) = d(uv)/d(p_cam) * R_world2camera.
        int jac_xyz_idx = residual_base * 2 * 3;
        
        jacobian_xyz_ptr[jac_xyz_idx + 0] = du_drx*r00 + du_dry*r10 + du_drz*r20; // du/dx
        jacobian_xyz_ptr[jac_xyz_idx + 1] = du_drx*r01 + du_dry*r11 + du_drz*r21; // du/dy
        jacobian_xyz_ptr[jac_xyz_idx + 2] = du_drx*r02 + du_dry*r12 + du_drz*r22; // du/dz
        
        jacobian_xyz_ptr[jac_xyz_idx + 3] = dv_drx*r00 + dv_dry*r10 + dv_drz*r20; // dv/dx
        jacobian_xyz_ptr[jac_xyz_idx + 4] = dv_drx*r01 + dv_dry*r11 + dv_drz*r21; // dv/dy
        jacobian_xyz_ptr[jac_xyz_idx + 5] = dv_drx*r02 + dv_dry*r12 + dv_drz*r22; // dv/dz
    #endif

    // Intrinsics Jacobians
    // Layout: [df] (if COMPUTE_J_F)
    #define N_INTRINSICS (COMPUTE_J_F)
    #if COMPUTE_J_F
        int jac_intrinsics_idx = residual_base * 2 * N_INTRINSICS;
        // d(u)/df = xn,  d(v)/df = yn
        jacobian_intrinsics_ptr[jac_intrinsics_idx + 0] = xn;
        // Row v
        jacobian_intrinsics_ptr[jac_intrinsics_idx + N_INTRINSICS] = yn;
    #endif
    #undef N_INTRINSICS

    #if COMPUTE_J_POSE
        // 2. SE3 Jacobian (Left Perturbation convention).
        // d(p_cam) / d(xi) = [ I  |  -[p_cam]_cross ]
        
        // Translation part (d_pcam / d_t = I) -> just Projection Jacobian
        float du_dtx = du_drx;
        float du_dty = du_dry;
        float du_dtz = du_drz;

        float dv_dtx = dv_drx;
        float dv_dty = dv_dry;
        float dv_dtz = dv_drz;

        // Rotation part (d_pcam / d_omega = -[p_cam]x)
        // [ 0  z -y ]
        // [-z  0  x ]
        // [ y -x  0 ]
        float du_dwx = du_drx * 0.0f       + du_dry * (-p_cam_z)  + du_drz * (p_cam_y);
        float du_dwy = du_drx * (p_cam_z)  + du_dry * 0.0f        + du_drz * (-p_cam_x);
        float du_dwz = du_drx * (-p_cam_y) + du_dry * (p_cam_x)   + du_drz * 0.0f;

        float dv_dwx = dv_drx * 0.0f       + dv_dry * (-p_cam_z)  + dv_drz * (p_cam_y);
        float dv_dwy = dv_drx * (p_cam_z)  + dv_dry * 0.0f        + dv_drz * (-p_cam_x);
        float dv_dwz = dv_drx * (-p_cam_y) + dv_dry * (p_cam_x)   + dv_drz * 0.0f;

        // Store into the dense jacobian structure
        // Note: You use a dense jacobian [output_idx, n_cams, 6]
        int jac_base = residual_base * 2 * n_cams * 6; // Stride is now 6, not 9!
        int cam_param_offset = camera_idx * 6;         // Offset is 6

        // Row u
        jacobian_se3_ptr[jac_base + cam_param_offset + 0] = du_dtx;
        jacobian_se3_ptr[jac_base + cam_param_offset + 1] = du_dty;
        jacobian_se3_ptr[jac_base + cam_param_offset + 2] = du_dtz;
        jacobian_se3_ptr[jac_base + cam_param_offset + 3] = du_dwx;
        jacobian_se3_ptr[jac_base + cam_param_offset + 4] = du_dwy;
        jacobian_se3_ptr[jac_base + cam_param_offset + 5] = du_dwz;

        // Row v (next row in the dense block)
        jac_base += n_cams * 6;
        jacobian_se3_ptr[jac_base + cam_param_offset + 0] = dv_dtx;
        jacobian_se3_ptr[jac_base + cam_param_offset + 1] = dv_dty;
        jacobian_se3_ptr[jac_base + cam_param_offset + 2] = dv_dtz;
        jacobian_se3_ptr[jac_base + cam_param_offset + 3] = dv_dwx;
        jacobian_se3_ptr[jac_base + cam_param_offset + 4] = dv_dwy;
        jacobian_se3_ptr[jac_base + cam_param_offset + 5] = dv_dwz;
    #endif
}

extern "C" __global__ void update_poses_kernel(
    const float* __restrict__ poses_in,  // [N, 12] Row-major (3x4)
    const float* __restrict__ delta,     // [N, 6]  (tx, ty, tz, wx, wy, wz)
    float* __restrict__ poses_out,       // [N, 12] Row-major (3x4)
    int n_cams
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cams) return;

    // 1. Load Delta (and negate it immediately because Python does exp(-dpose))
    // Layout: tx, ty, tz, wx, wy, wz
    const float* d = delta + idx * 6;
    float u0 = -d[0]; 
    float u1 = -d[1]; 
    float u2 = -d[2];
    float w0 = -d[3]; 
    float w1 = -d[4]; 
    float w2 = -d[5];

    // 2. Compute Exponential Map (se3 -> SE3)
    // theta^2 = w . w
    float theta_sq = w0*w0 + w1*w1 + w2*w2;
    float theta = sqrtf(theta_sq);

    // Coefficients for Rodrigues formula
    // A = sin(th)/th, B = (1-cos(th))/th^2, C = (th-sin(th))/th^3
    float A, B, C;
    
    // Taylor expansion for small angles to avoid division by zero
    if (theta < 1e-5f) {
        A = 1.0f - theta_sq * (1.0f / 6.0f);
        B = 0.5f - theta_sq * (1.0f / 24.0f);
        C = (1.0f / 6.0f) - theta_sq * (1.0f / 120.0f);
    } else {
        float inv_theta = 1.0f / theta;
        float inv_theta_sq = inv_theta * inv_theta;
        A = sinf(theta) * inv_theta;
        B = (1.0f - cosf(theta)) * inv_theta_sq;
        C = (theta - sinf(theta)) * (inv_theta_sq * inv_theta);
    }

    // K = [w]x
    // K^2 = w * w^T - theta^2 * I
    
    // Construct R_upd = I + A*K + B*K^2
    // We compute elements directly to avoid building full temp matrices
    // K = [ 0  -w2  w1]
    //     [ w2  0  -w0]
    //     [-w1  w0  0 ]
    
    float wx2 = w0*w0, wy2 = w1*w1, wz2 = w2*w2;
    float wxy = w0*w1, wxz = w0*w2, wyz = w1*w2;

    float R_upd_00 = 1.0f - B*(wy2 + wz2);
    float R_upd_01 = -A*w2 + B*wxy;
    float R_upd_02 =  A*w1 + B*wxz;

    float R_upd_10 =  A*w2 + B*wxy;
    float R_upd_11 = 1.0f - B*(wx2 + wz2);
    float R_upd_12 = -A*w0 + B*wyz;

    float R_upd_20 = -A*w1 + B*wxz;
    float R_upd_21 =  A*w0 + B*wyz;
    float R_upd_22 = 1.0f - B*(wx2 + wy2);

    // V matrix for translation update: V = I + B*K + C*K^2
    // t_update = V * u
    // V = I + B*[w]x + C*(w*w^T - th^2*I)
    // It is effectively R(with A->B, B->C) applied to u? No, slightly different coeff.
    // Let's multiply directly: t_upd = u + B*(w x u) + C*(w x (w x u))
    
    // cross1 = w x u
    float c1_0 = w1*u2 - w2*u1;
    float c1_1 = w2*u0 - w0*u2;
    float c1_2 = w0*u1 - w1*u0;

    // cross2 = w x cross1
    float c2_0 = w1*c1_2 - w2*c1_1;
    float c2_1 = w2*c1_0 - w0*c1_2;
    float c2_2 = w0*c1_1 - w1*c1_0;

    float t_upd_0 = u0 + B*c1_0 + C*c2_0;
    float t_upd_1 = u1 + B*c1_1 + C*c2_1;
    float t_upd_2 = u2 + B*c1_2 + C*c2_2;

    // 3. Apply Update: Pose_new = T_upd * Pose_old
    // R_new = R_upd * R_old
    // t_new = R_upd * t_old + t_upd
    
    const float* p_in = poses_in + idx * 12;
    
    // Load Old R
    float r00 = p_in[0], r01 = p_in[1], r02 = p_in[2], tx = p_in[3];
    float r10 = p_in[4], r11 = p_in[5], r12 = p_in[6], ty = p_in[7];
    float r20 = p_in[8], r21 = p_in[9], r22 = p_in[10], tz = p_in[11];

    // Compute New R
    float n00 = R_upd_00*r00 + R_upd_01*r10 + R_upd_02*r20;
    float n01 = R_upd_00*r01 + R_upd_01*r11 + R_upd_02*r21;
    float n02 = R_upd_00*r02 + R_upd_01*r12 + R_upd_02*r22;

    float n10 = R_upd_10*r00 + R_upd_11*r10 + R_upd_12*r20;
    float n11 = R_upd_10*r01 + R_upd_11*r11 + R_upd_12*r21;
    float n12 = R_upd_10*r02 + R_upd_11*r12 + R_upd_12*r22;

    float n20 = R_upd_20*r00 + R_upd_21*r10 + R_upd_22*r20;
    float n21 = R_upd_20*r01 + R_upd_21*r11 + R_upd_22*r21;
    float n22 = R_upd_20*r02 + R_upd_21*r12 + R_upd_22*r22;

    // Compute New t
    float ntx = R_upd_00*tx + R_upd_01*ty + R_upd_02*tz + t_upd_0;
    float nty = R_upd_10*tx + R_upd_11*ty + R_upd_12*tz + t_upd_1;
    float ntz = R_upd_20*tx + R_upd_21*ty + R_upd_22*tz + t_upd_2;

    // 4. Store
    float* p_out = poses_out + idx * 12;
    p_out[0] = n00; p_out[1] = n01; p_out[2] = n02; p_out[3] = ntx;
    p_out[4] = n10; p_out[5] = n11; p_out[6] = n12; p_out[7] = nty;
    p_out[8] = n20; p_out[9] = n21; p_out[10] = n22; p_out[11] = ntz;
}

extern "C" __global__ void inv_3x3_kernel(
    const float* __restrict__ A,   // [N, 3, 3]
    float* __restrict__ A_inv,     // [N, 3, 3]
    int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* m = A + idx * 9;
    float a = m[0], b = m[1], c = m[2];
    float d = m[3], e = m[4], f = m[5];
    float g = m[6], h = m[7], i = m[8];

    float ei_fh = e * i - f * h;
    float fg_di = f * g - d * i;
    float dh_eg = d * h - e * g;

    float det = a * ei_fh + b * fg_di + c * dh_eg;
    float inv_det = (det != 0.0f) ? (1.0f / det) : 0.0f;

    float* o = A_inv + idx * 9;
    o[0] = ei_fh * inv_det;
    o[1] = (c * h - b * i) * inv_det;
    o[2] = (b * f - c * e) * inv_det;
    o[3] = fg_di * inv_det;
    o[4] = (a * i - c * g) * inv_det;
    o[5] = (c * d - a * f) * inv_det;
    o[6] = dh_eg * inv_det;
    o[7] = (b * g - a * h) * inv_det;
    o[8] = (a * e - b * d) * inv_det;
}
