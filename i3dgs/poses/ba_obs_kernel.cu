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


// ---------------------------------------------------------------------------
// add_obs_to_existing_kernel
//
// One thread per match pair (match_idx[i], match_idx_other[i]).
// Adds an obs of the current KF to the matched landmark IFF projecting that
// landmark into the current KF gives a reprojection error below
// outlier_threshold. Otherwise the kpt is left unclaimed so another match (or
// the fresh-landmark path in append()) can use it.
//
// GPU-side atomic counter obs_size_ptr avoids any host sync for slot
// allocation.
// ---------------------------------------------------------------------------
extern "C" __global__ void add_obs_to_existing_kernel(
    int n_match,
    const int*          match_idx,          // [n_match]  current-KF kpt indices
    const int*          match_idx_other,    // [n_match]  other-KF  kpt indices
    const int*          id_in_ba_other,     // [n_kpts_other]
    bool*               existing_mask,      // [n_kpts]  read + write
    int                 keyframe_index,
    const float*        kpts_curr,          // [n_kpts * 2]  (u,v) interleaved
    // Reprojection-error gate: project landmarks[ba_id] with current KF's
    // (Rt, f, centre) and compare to kpts_curr[curr_idx].
    const float*        Rt_curr,            // [12]  row-major 3x4 (w2c)
    const float*        f_curr,             // [1]
    const float*        centre_curr,        // [2]
    float               outlier_threshold,  // pixels; obs rejected if error >=
    const float*        landmarks,          // [lm_cap * 3]  (read-only here)
    // COO obs arrays (written via atomic slot)
    int*                obs_lm_ids,
    int*                obs_kf_ids,
    int*                obs_pt2d_ids,
    float*              obs_uvs,            // [obs_cap * 2]
    int*                obs_size_ptr,       // GPU scalar, atomicAdd
    // Landmark arrays
    int*                n_obs_arr,          // [lm_cap]  atomicAdd
    int*                id_in_ba_curr       // [n_kpts]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_match) return;

    int curr_idx = match_idx[i];
    if (existing_mask[curr_idx]) return;

    int other_idx = match_idx_other[i];
    int ba_id     = id_in_ba_other[other_idx];
    if (ba_id < 0) return;

    // Reprojection-error gate
    float X = landmarks[3 * ba_id];
    float Y = landmarks[3 * ba_id + 1];
    float Z = landmarks[3 * ba_id + 2];

    float r00 = Rt_curr[0],  r01 = Rt_curr[1],  r02 = Rt_curr[2],  tx = Rt_curr[3];
    float r10 = Rt_curr[4],  r11 = Rt_curr[5],  r12 = Rt_curr[6],  ty = Rt_curr[7];
    float r20 = Rt_curr[8],  r21 = Rt_curr[9],  r22 = Rt_curr[10], tz = Rt_curr[11];

    float p_cam_x = r00 * X + r01 * Y + r02 * Z + tx;
    float p_cam_y = r10 * X + r11 * Y + r12 * Z + ty;
    float p_cam_z = r20 * X + r21 * Y + r22 * Z + tz;
    if (p_cam_z <= 1e-6f) return;  // behind camera

    float inv_z = 1.0f / p_cam_z;
    float xn = p_cam_x * inv_z;
    float yn = p_cam_y * inv_z;
    float f  = f_curr[0];
    float u_proj = f * xn + centre_curr[0];
    float v_proj = f * yn + centre_curr[1];

    float du = u_proj - kpts_curr[2 * curr_idx];
    float dv = v_proj - kpts_curr[2 * curr_idx + 1];
    float err = sqrtf(du * du + dv * dv);
    if (err >= outlier_threshold) return;

    int slot = atomicAdd(obs_size_ptr, 1);
    obs_lm_ids[slot]      = ba_id;
    obs_kf_ids[slot]      = keyframe_index;
    obs_pt2d_ids[slot]    = curr_idx;
    obs_uvs[2 * slot]     = kpts_curr[2 * curr_idx];
    obs_uvs[2 * slot + 1] = kpts_curr[2 * curr_idx + 1];
    atomicAdd(&n_obs_arr[ba_id], 1);
    id_in_ba_curr[curr_idx] = ba_id;
    // Claim this keypoint (no conflict within a kernel launch: curr_idx values are unique)
    existing_mask[curr_idx] = true;
}

// ---------------------------------------------------------------------------
// add_obs_from_other_kf_kernel
//
// One thread per match pair.  Adds an observation from the other keyframe to
// a landmark that was *just* created by the current keyframe -- but only if
// projecting that landmark into the other keyframe gives a reprojection
// error below outlier_threshold.  This filters out spurious matches that
// contributed to triangulation noise.
// ---------------------------------------------------------------------------
extern "C" __global__ void add_obs_from_other_kf_kernel(
    int n_match,
    const int*          match_idx,          // [n_match]  current-KF kpt indices
    const int*          match_idx_other,    // [n_match]  other-KF  kpt indices
    const bool*         new_lm_mask,        // [n_kpts]  True where new lm was created
    const int*          id_in_ba_curr,      // [n_kpts]
    int                 max_obs,
    const int*          n_obs_arr,          // [lm_cap]  read-only for room check
    const float*        kpts_other,         // [n_kpts_other * 2]
    int                 kf_id_other,
    // Reprojection-error gate: project landmarks[ba_id] with other KF's
    // (Rt, f, centre) and compare to kpts_other[other_idx].
    const float*        Rt_other,           // [12]  row-major 3x4 (w2c)
    const float*        f_other,            // [1]
    const float*        centre_other,       // [2]
    float               outlier_threshold,  // pixels; obs rejected if error >=
    const float*        landmarks,          // [lm_cap * 3]  (read-only here)
    // COO obs arrays
    int*                obs_lm_ids,
    int*                obs_kf_ids,
    int*                obs_pt2d_ids,
    float*              obs_uvs,
    int*                obs_size_ptr,       // GPU scalar, atomicAdd
    int*                n_obs_arr_write,    // [lm_cap]  atomicAdd
    int*                id_in_ba_other      // [n_kpts_other]  write
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_match) return;

    int curr_idx = match_idx[i];
    if (!new_lm_mask[curr_idx]) return;

    int ba_id = id_in_ba_curr[curr_idx];
    if (ba_id < 0 || n_obs_arr[ba_id] >= max_obs) return;

    int other_idx = match_idx_other[i];

    // Reprojection-error gate (same math as add_obs_to_existing_kernel)
    float X = landmarks[3 * ba_id];
    float Y = landmarks[3 * ba_id + 1];
    float Z = landmarks[3 * ba_id + 2];

    float r00 = Rt_other[0],  r01 = Rt_other[1],  r02 = Rt_other[2],  tx = Rt_other[3];
    float r10 = Rt_other[4],  r11 = Rt_other[5],  r12 = Rt_other[6],  ty = Rt_other[7];
    float r20 = Rt_other[8],  r21 = Rt_other[9],  r22 = Rt_other[10], tz = Rt_other[11];

    float p_cam_x = r00 * X + r01 * Y + r02 * Z + tx;
    float p_cam_y = r10 * X + r11 * Y + r12 * Z + ty;
    float p_cam_z = r20 * X + r21 * Y + r22 * Z + tz;
    if (p_cam_z <= 1e-6f) return;  // behind camera

    float inv_z = 1.0f / p_cam_z;
    float xn = p_cam_x * inv_z;
    float yn = p_cam_y * inv_z;
    float f  = f_other[0];
    float u_proj = f * xn + centre_other[0];
    float v_proj = f * yn + centre_other[1];

    float du = u_proj - kpts_other[2 * other_idx];
    float dv = v_proj - kpts_other[2 * other_idx + 1];
    float err = sqrtf(du * du + dv * dv);
    if (err >= outlier_threshold) return;

    int slot = atomicAdd(obs_size_ptr, 1);
    obs_lm_ids[slot]      = ba_id;
    obs_kf_ids[slot]      = kf_id_other;
    obs_pt2d_ids[slot]    = other_idx;
    obs_uvs[2 * slot]     = kpts_other[2 * other_idx];
    obs_uvs[2 * slot + 1] = kpts_other[2 * other_idx + 1];
    atomicAdd(&n_obs_arr_write[ba_id], 1);
    id_in_ba_other[other_idx] = ba_id;
}

// ---------------------------------------------------------------------------
// build_dense_obs_kernel
//
// One thread per COO observation entry (indices 0..obs_size-1).
// Converts the flat COO obs store to dense [n_sel, max_obs] tensors for BA.
//
// Replaces: isin, boolean-index (+ sync), argsort, scatter_add+cumsum for
// slot computation, torch.where/clamp, and two large 2-D scatter writes.
//
// Slot assignment uses atomicAdd on per-landmark counters -- no sort needed.
// Obs that overflow max_obs are silently dropped (same behaviour as before).
// ---------------------------------------------------------------------------
extern "C" __global__ void build_dense_obs_kernel(
    int                 obs_size,
    const int*          obs_lm,         // [obs_size]   landmark id
    const int*          obs_kf,         // [obs_size]   keyframe id
    const int*          obs_pt2d,       // [obs_size]   flat pool kpt id
    const float*        obs_uv,         // [obs_size*2] (u,v) interleaved
    const int*          lm_to_local,    // [lm_cap]     -1 if not selected
    const int*          kf_selected,    // [max_kf_id+1] 1 if selected kf, else 0
    const int*          kf_id_to_cam,   // [max_kf_id+1] cam index, -1 if not selected
    int                 max_obs,
    // per-landmark slot counter (initialised to 0 before launch)
    int*                slot_counters,  // [n_sel]
    // output dense tensors (initialised to -1/-1.0 before launch)
    float*              out_uvs,        // [n_sel * max_obs * 2]
    int*                out_kf_ids,     // [n_sel * max_obs]
    int*                out_cam_ids     // [n_sel * max_obs]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= obs_size) return;

    int kf = obs_kf[i];
    if (!kf_selected[kf]) return;

    int local_id = lm_to_local[obs_lm[i]];
    if (local_id < 0) return;

    int slot = atomicAdd(&slot_counters[local_id], 1);
    if (slot >= max_obs) return;

    int base = (local_id * max_obs + slot);
    out_uvs[base * 2]     = obs_uv[i * 2];
    out_uvs[base * 2 + 1] = obs_uv[i * 2 + 1];
    out_kf_ids [base]     = kf;
    out_cam_ids[base]     = kf_id_to_cam[kf];
}
