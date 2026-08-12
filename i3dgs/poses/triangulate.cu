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

#include "cuda_utils.cuh"

/**
 * @brief Triangulate 3D points from a reference keyframe and N other keyframes.
 *
 * For each keypoint in the reference frame, iterates over all matching cameras,
 * runs midpoint triangulation, and keeps the result with the highest disambiguation
 * that also passes the reprojection error threshold.
 *
 * rel_rts is laid out per (camera, reference-kpt) pair (12 floats, row-major 3x4):
 *   [r00, r01, r02, tx,  r10, r11, r12, ty,  r20, r21, r22, tz]
 * R maps directions from cam j to cam 0. t is cam-j's origin in cam-0 space.
 *
 * @param uv          [N, 2]    keypoints in reference frame
 * @param uvs_others  [M, N, 2] matched keypoints in other frames (-1 if unmatched)
 * @param rel_rts     [M, N, 12]  per-(camera, ref-kpt) relative transform
 * @param kpts3d      [N, 3]    output: 3D points in cam 0 space
 * @param best_dis    [N]       output: best disambiguation score
 * @param f           focal length
 * @param cx, cy      principal point
 * @param max_error   reprojection error threshold
 * @param N           number of keypoints
 * @param M           number of cameras
 */
extern "C" __global__ void triangulatePoints(
    const float* uv,
    const float* uvs_others,
    const float* rel_rts,
    float*       kpts3d,
    float*       best_dis,
    float f, float cx, float cy,
    float max_error,
    int N,
    int M
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Normalised direction d1 in cam 0
    float d1x = (uv[idx * 2 + 0] - cx) / f;
    float d1y = (uv[idx * 2 + 1] - cy) / f;
    float inv_n1 = rsqrtf(d1x*d1x + d1y*d1y + 1.f);
    d1x *= inv_n1;
    float d1y_ = d1y * inv_n1;
    float d1z  = inv_n1;              // == 1 / norm(d1_raw)

    float out_x = 0.f, out_y = 0.f, out_z = 0.f, bdis = 0.f;

    for (int j = 0; j < M; j++) {
        float p2x = uvs_others[(j * N + idx) * 2 + 0];
        float p2y = uvs_others[(j * N + idx) * 2 + 1];
        if (p2x <= 0.f || p2y <= 0.f) continue;

        // rel_rts row-major [3,4] per (camera, ref-kpt):
        //   rt[0..2]=R row0, rt[3]=tx, rt[4..6]=R row1, rt[7]=ty, rt[8..10]=R row2, rt[11]=tz
        const float* rt = rel_rts + (j * N + idx) * 12;
        float tx = rt[3], ty_ = rt[7], tz = rt[11];

        // Normalised d2 in cam j, rotated to cam 0: d2r = R @ d2
        float d2x = (p2x - cx) / f;
        float d2y = (p2y - cy) / f;
        float inv_n2 = rsqrtf(d2x*d2x + d2y*d2y + 1.f);
        d2x *= inv_n2;
        d2y *= inv_n2;
        float d2z_ = inv_n2;

        float d2rx = rt[0]*d2x + rt[1]*d2y + rt[2]*d2z_;
        float d2ry = rt[4]*d2x + rt[5]*d2y + rt[6]*d2z_;
        float d2rz = rt[8]*d2x + rt[9]*d2y + rt[10]*d2z_;

        // Midpoint triangulation: n = cross(d1, d2r),  n2 = cross(d2r, n)
        float nx  = d1y_*d2rz - d1z*d2ry;
        float ny  = d1z*d2rx  - d1x*d2rz;
        float nz  = d1x*d2ry  - d1y_*d2rx;
        float n2x = d2ry*nz   - d2rz*ny;
        float n2y = d2rz*nx   - d2rx*nz;
        float n2z = d2rx*ny   - d2ry*nx;

        float denom = n2x*d1x + n2y*d1y_ + n2z*d1z;
        float dist  = (n2x*tx + n2y*ty_ + n2z*tz) / denom;

        float px = d1x*dist, py = d1y_*dist, pz = d1z*dist;

        // Disambiguation: project d1 (depth 1) and d1*10 into cam j  (R^T @ (v - t))
        float e1x = d1x - tx,       e1y = d1y_ - ty_,       e1z = d1z - tz;
        float c1x = rt[0]*e1x + rt[4]*e1y + rt[8]*e1z;
        float c1y = rt[1]*e1x + rt[5]*e1y + rt[9]*e1z;
        float c1z = rt[2]*e1x + rt[6]*e1y + rt[10]*e1z;
        float e2x = d1x*10.f - tx,  e2y = d1y_*10.f - ty_,  e2z = d1z*10.f - tz;
        float c2x = rt[0]*e2x + rt[4]*e2y + rt[8]*e2z;
        float c2y = rt[1]*e2x + rt[5]*e2y + rt[9]*e2z;
        float c2z = rt[2]*e2x + rt[6]*e2y + rt[10]*e2z;
        float du = f*(c1x/c1z - c2x/c2z);
        float dv = f*(c1y/c1z - c2y/c2z);
        float dis = sqrtf(du*du + dv*dv);

        // Reprojection error: project triangulated point into cam j  (R^T @ (xyz - t))
        float dx = px - tx, dy = py - ty_, dz = pz - tz;
        float mx = rt[0]*dx + rt[4]*dy + rt[8]*dz;
        float my = rt[1]*dx + rt[5]*dy + rt[9]*dz;
        float mz = rt[2]*dx + rt[6]*dy + rt[10]*dz;
        float eu = f*mx/mz + cx - p2x;
        float ev = f*my/mz + cy - p2y;
        float err = sqrtf(eu*eu + ev*ev);

        if (pz > 1e-6f && dis > bdis && err < max_error) {
            out_x = px; out_y = py; out_z = pz;
            bdis  = dis;
        }
    }

    kpts3d[idx * 3 + 0] = out_x;
    kpts3d[idx * 3 + 1] = out_y;
    kpts3d[idx * 3 + 2] = out_z;
    best_dis[idx]        = bdis;
}
