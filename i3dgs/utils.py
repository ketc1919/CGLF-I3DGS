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

import gc
import numpy as np
import torch
import math
import time
import cv2
import torch.nn.functional as F
import os
import cupy
import socket
from joblib import Parallel, delayed

def bench(func, *args, warmup=3, iters=10, desc=""):
    # warmup (ignore startup jit/allocs)
    for _ in range(warmup):
        func(*args)

    torch.cuda.synchronize()  # make sure GPU finished before timing
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()

    for _ in range(iters):
        out = func(*args)
    torch.cuda.synchronize()
    cupy.cuda.Device().synchronize()

    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / iters
    print(f"{desc:20s}: {avg_ms:.3f} ms")
    return out

@torch.no_grad()
def kmeans_pytorch(X, n_clusters, max_iters=100, tol=1e-4):
    """
    Fast K-means clustering in pure PyTorch (CUDA/ROCm compatible)

    Args:
        X: (N, D) tensor of points
        n_clusters: number of clusters
        max_iters: maximum iterations
        tol: convergence tolerance

    Returns:
        labels: (N,) cluster assignments
        centroids: (n_clusters, D) cluster centers
        counts: (n_clusters,) number of points per cluster
    """
    N, D = X.shape

    # K-means++ initialization
    indices = torch.randperm(N, device=X.device)[:n_clusters]
    centroids = X[indices].clone()

    for _ in range(max_iters):
        # Compute distances: (N, n_clusters)
        dists = torch.cdist(X, centroids)

        # Assign to nearest cluster
        labels = dists.argmin(dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centroids[k] = X[mask].mean(0)
            else:
                # Handle empty cluster: reinitialize
                new_centroids[k] = X[torch.randint(N, (1,), device=X.device)]

        # Check convergence
        if (centroids - new_centroids).norm() < tol:
            break
        centroids = new_centroids

    # Count points per cluster
    counts = torch.bincount(labels, minlength=n_clusters)

    return labels, centroids, counts

def parse_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def get_image_names(in_folder, image_extensions=[".jpg", ".png", ".jpeg", ".webp"]):
    return [
        f
        for f in os.listdir(in_folder)
        if os.path.splitext(f)[-1].lower() in image_extensions
    ]

def psnr(img1, img2):
    return 10 * torch.log10(1 / F.mse_loss(img1, img2)).item()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def write_tensor_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = torch.clamp(img, 0, 1) * 255
    image = to_numpy(image.permute(1, 2, 0)).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    is_jpeg = os.path.splitext(path)[-1].lower() in [
        ".jpg",
        ".jpeg",
    ]
    write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
    cv2.imwrite(
        os.path.join(path), image, write_flag
    )

def batch_write_imgs(imgs, paths):
    Parallel(n_jobs=-1, backend='threading')(
        delayed(write_tensor_img)(
            img, path
        ) for img, path in zip(imgs, paths)
    )

def get_lapla_norm(img, kernel):
    laplacian_kernel = (
        torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device="cuda", dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
    laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
    laplacian_norm = torch.linalg.vector_norm(laplacian, ord=1, dim=1, keepdim=True)
    laplacian_norm[..., :, 0] = 0
    laplacian_norm[..., :, -1] = 0
    laplacian_norm[..., 0, :] = 0
    laplacian_norm[..., -1, :] = 0
    return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)

def get_time():
    # torch.cuda.synchronize()
    return time.perf_counter()

def increment_runtime(runtime, start_time):
    # torch.cuda.synchronize()
    runtime[0] += time.perf_counter() - start_time
    runtime[1] += 1


def free_cuda_memory():
    """Run the garbage collector and empty the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


## Camera/triangulation/projection functions
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def depth2points(uv, depth, f, centre):
    xyz = torch.cat([(uv[..., :2] - centre) / f, torch.ones_like(uv[..., 0:1])], dim=-1)
    return depth * xyz


def reproject(uv, depth, f, centre, relR, relt):
    xyz = depth2points(uv, depth, f, centre)
    xyz = xyz @ relR.T + relt
    return pts2px(xyz, f, centre)


def make_torch_sampler(uv, width, height):
    """
    Converts OpenCV UV coordinates to a sampler for torch's grid_sample.
    To be used with align_corners=True
    """
    sampler = uv.clone()  # + 0.5
    sampler[..., 0] = sampler[..., 0] * (2.0 / (width - 1)) - 1.0
    sampler[..., 1] = sampler[..., 1] * (2.0 / (height - 1)) - 1.0
    return sampler


def sample(map, uv, width, height):
    sampler = make_torch_sampler(uv, width, height)
    return F.grid_sample(map, sampler, mode="bilinear", align_corners=True)


def pts2px(xyz, f, centre):
    return f * xyz[..., :2] / xyz[..., 2:3] + centre


def sixD2mtx(r):
    b1 = r[..., 0]
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    b2 = r[..., 1] - torch.sum(b1 * r[..., 1], dim=-1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def mtx2sixD(R):
    return R[..., :2].clone()


## Visualization functions
def display_matches(mkpts1, mkpts2, img1, img2, scale=1, match_step=1, indices=None):
    image1 = img1.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    image2 = img2.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    if indices is not None:
        mkpts1 = mkpts1[indices]
        mkpts2 = mkpts2[indices]
    matched_mkptsi_np = mkpts1[::match_step].cpu().float().numpy()
    matched_mkptsj_np = mkpts2[::match_step].cpu().float().numpy()
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in matched_mkptsi_np]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in matched_mkptsj_np]
    mask_np = (
        ((mkpts1 != -1).all(dim=-1) * (mkpts2 != -1).all(dim=-1))[::match_step]
        .cpu()
        .numpy()
    )
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask_np)) if mask_np[i]]
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    if scale != 1:
        img_matches = cv2.resize(img_matches, (0, 0), fx=scale, fy=scale)
    cv2.imshow("matches_img", img_matches[..., ::-1])
    cv2.waitKey()


@torch.no_grad()
def draw_poses(image, view_matrix, view_fovx, scale, cam_width, cam_height, Rts, cam_f, color):
    """
    Overlay the camera frustums on the np image

    Args:
       image (np.ndarray): The image to draw on
       view_matrix (torch.Tensor): The point of view to render from
       view_fov (float): The field of view to render with
       scale (float): The scale of the drawn poses
       cam_width (int): The width of the image to draw the frustums
       cam_height (int): The height of the image to draw the frustums
       Rts (torch.Tensor): The camera poses to draw (camera to world)
       cam_f (float): The focal length of the poses to draw
    Returns:
       image (np.ndarray): The image with the frustums drawn on
    """
    if len(Rts) > 0:
        # Rendering options
        width, height = image.shape[1], image.shape[0]
        f = fov2focal(view_fovx, width)
        centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')

        # Camera intrinsics to draw
        cam_centre = torch.tensor([(cam_width - 1) / 2, (cam_height - 1) / 2], device='cuda')
        # Make a 3D frustum using intrinsics
        origin = torch.tensor([0, 0, 0], device='cuda')
        corners2d = torch.tensor([[0, 0], [cam_width, 0], [cam_width, cam_height], [0, cam_height]], device='cuda')
        corners3d = depth2points(corners2d, scale, cam_f, cam_centre)
        # Duplicate and transform frustums for each pose
        cams_verts = torch.cat([origin.unsqueeze(0), corners3d], dim=0)
        n_cams = Rts.shape[0]
        cams_verts = torch.bmm((cams_verts - Rts[:n_cams, None, :3, 3]), Rts[:n_cams, :3, :3])
        cams_verts_view = (cams_verts @ view_matrix[:3, :3] + view_matrix[3:4, :3])
        cams_verts_2d = pts2px(cams_verts_view, f, centre).view(n_cams, -1, 2)
        # Out of view check
        valid_cams = (cams_verts_view[..., 2] > 0).all(dim=-1)
        cams_verts_2d = cams_verts_2d[valid_cams]

        # Draw frustums on the image
        draw_order = torch.tensor([1, 2, 0, 3, 4, 0, 1, 4, 3, 2], device="cuda")
        cams_verts_2d = cams_verts_2d[..., draw_order, :]
        image = cv2.polylines(
            image,
            cams_verts_2d.detach().cpu().numpy().astype(int),
            isClosed=False,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return image


def get_transform_mean_up_fwd(input, target, w_scale):
    """
    Get the transform that aligns input poses to target mean position, up and forward vectors.
    This appears more stable than Procrustes analysis.

    The input and target are both [N,4,4] transforms from world to camera.
    We want to:
      - match the mean position (camera center) of 'input' to that of 'target'
      - align the average "up" direction from 'input' to the average "up" direction of 'target'
      - align the average "forward" direction from 'input' to the average "forward" direction of 'target'

    """
    inv_input = torch.linalg.inv(input)
    inv_target = torch.linalg.inv(target)
    center_input = inv_input[:, :3, 3]
    center_target = inv_target[:, :3, 3]

    # Compute average up and forward vectors in world coords
    up_input_avg = inv_input[:, :3, 1].mean(dim=0)
    up_target_avg = inv_target[:, :3, 1].mean(dim=0)
    fwd_input_avg = inv_input[:, :3, 2].mean(dim=0)
    fwd_target_avg = inv_target[:, :3, 2].mean(dim=0)

    # Normalize these average directions to get unit vectors
    up_input_avg = up_input_avg / up_input_avg.norm()
    up_target_avg = up_target_avg / up_target_avg.norm()
    fwd_input_avg = fwd_input_avg / fwd_input_avg.norm()
    fwd_target_avg = fwd_target_avg / fwd_target_avg.norm()

    # Input basis
    right_input = torch.cross(up_input_avg, fwd_input_avg)
    right_input = right_input / right_input.norm()

    R_in = torch.stack([right_input, up_input_avg, fwd_input_avg], dim=1)

    # Target basis
    right_target = torch.cross(up_target_avg, fwd_target_avg)
    right_target = right_target / right_target.norm()

    R_tgt = torch.stack([right_target, up_target_avg, fwd_target_avg], dim=1)

    # This rotation aligns the input basis to target basis
    R = R_tgt @ R_in.transpose(0, 1)

    # This scale aligns the input center to target center
    center_input_mean = center_input.mean(dim=0)
    center_target_mean = center_target.mean(dim=0)
    if w_scale:
        s_input = ((center_input - center_input_mean)**2).sum(dim=-1).mean().sqrt()
        s_target = ((center_target - center_target_mean)**2).sum(dim=-1).mean().sqrt()
        s = s_target / s_input
    else:
        s = 1.0

    # This translation aligns the input center to target center
    t = center_target_mean - R @ center_input_mean * s

    return R, t, s


def align_mean_up_fwd(input, target, w_scale=False):
    """
    Align input poses to target mean position, up and forward vectors.

    Returns:
      A set of [N,4,4] transforms, which are the aligned poses of 'input'.
    """

    R, t, s = get_transform_mean_up_fwd(input, target, w_scale)
    inv_input = torch.linalg.inv(input)
    inv_input[:, :3, :3] = R @ inv_input[:, :3, :3]
    inv_input[:, :3, 3] = (R @ inv_input[:, :3, 3:4]).squeeze(-1) * s + t[None]

    return torch.linalg.inv(inv_input)

## Pose alignment and evaluation functions
def align_poses(input, target, w_scale=True):
    """Align input poses to target using Procrustes analysis on camera centers"""
    return align_poses_against_first_n(input, target, n=0, w_scale=w_scale)


## From https://github.com/chenhsuanlin/bundle-adjusting-NeRF
# BARF: Bundle-Adjusting Neural Radiance Fields
# Copyright (c) 2021 Chen-Hsuan Lin
# Under the MIT License.
# Modified to interface with our pose format 
def rotation_distance(R1, R2, eps=1e-9):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = (
        ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
    )  # numerical stability near -1/+1
    return angle


def procrustes_analysis(X0, X1, w_scale=True):  # [N,3]
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    if w_scale:
        s0 = (X0c**2).sum(dim=-1).mean().sqrt()
        s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    else:
        s0, s1 = 1, 1
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)
    U, S, V = (X0cs.t() @ X1cs).svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0:
        R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    return t0[0], t1[0], s0, s1, R

def split_into_clusters(dists):
    """
    Split nodes into 2 clusters based on distance matrix.
    Returns (cluster_a, cluster_b) as lists of indices.
    """
    dists = torch.tensor(dists) if not isinstance(dists, torch.Tensor) else dists
    n = dists.shape[0]
    
    # Find the two most distant nodes → they're the cluster seeds
    max_idx = dists.argmax()
    seed_a, seed_b = max_idx // n, max_idx % n
    
    # Assign each node to its nearest seed
    cluster_a = []
    cluster_b = []
    for i in range(n):
        if dists[i, seed_a] <= dists[i, seed_b]:
            cluster_a.append(i)
        else:
            cluster_b.append(i)
    
    return cluster_a, cluster_b

## Pose alignment and evaluation functions
def align_poses_against_first_n(input, target, n, w_scale=True):
    """Align input poses to target using Procrustes analysis on camera centers"""
    center_input = torch.linalg.inv(input)[:, :3, 3]
    center_target = torch.linalg.inv(target)[:, :3, 3]
    if n <=0:
        n = center_input.shape[0]
    t0, t1, s0, s1, R = procrustes_analysis(center_target[:n], center_input[:n], w_scale)
    center_aligned = (center_input - t1) / s1 @ R.t() * s0 + t0
    R_aligned = input[:, :3, :3] @ R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    aligned = torch.eye(4, device=input.device).repeat(input.shape[0], 1, 1)
    aligned[:, :3, :3] = R_aligned[:, :3, :3]
    aligned[:, :3, 3] = t_aligned
    return aligned

def quat_to_rotmat(q):
    # q: (w, x, y, z)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(dim=-1)

    R = torch.stack([
        1 - 2 * (y * y + z * z),  2 * (x * y - w * z),      2 * (x * z + w * y),
        2 * (x * y + w * z),      1 - 2 * (x * x + z * z),  2 * (y * z - w * x),
        2 * (x * z - w * y),      2 * (y * z + w * x),      1 - 2 * (x * x + y * y)
    ], dim=-1)

    return R.view(q.shape[:-1] + (3, 3))

def rotmat_to_quat(R):
    # R: (...,3,3)  -> returns (...,4) as (w,x,y,z)
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    trace = m00 + m11 + m22

    # Candidate squared pivots (4 * component^2); the largest is always >= 1.
    pivots = torch.stack([1.0 + trace,
                          1.0 + m00 - m11 - m22,
                          1.0 - m00 + m11 - m22,
                          1.0 - m00 - m11 + m22], dim=-1)
    case = pivots.argmax(dim=-1)

    # Per-case quaternion (w, x, y, z); denom = 2*sqrt(pivot) >= 2 in every branch.
    def branch(t, a, b, c, order):
        denom = 2.0 * torch.sqrt(t.clamp_min(1e-12))
        pivot = 0.25 * denom  # = 0.5*sqrt(t)
        vals = {order[0]: pivot, order[1]: a / denom, order[2]: b / denom, order[3]: c / denom}
        return torch.stack([vals['w'], vals['x'], vals['y'], vals['z']], dim=-1)

    q_w = branch(pivots[:, 0], m21 - m12, m02 - m20, m10 - m01, ['w', 'x', 'y', 'z'])
    q_x = branch(pivots[:, 1], m21 - m12, m01 + m10, m02 + m20, ['x', 'w', 'y', 'z'])
    q_y = branch(pivots[:, 2], m02 - m20, m01 + m10, m12 + m21, ['y', 'w', 'x', 'z'])
    q_z = branch(pivots[:, 3], m10 - m01, m02 + m20, m12 + m21, ['z', 'w', 'x', 'y'])

    q = torch.stack([q_w, q_x, q_y, q_z], dim=1)  # (N, 4, 4)
    q = q.gather(1, case.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)

    # Enforce positive w and normalize.
    q = torch.where(q[:, :1] < 0, -q, q)
    q = q / q.norm(dim=-1, keepdim=True)

    return q.view(*batch_shape, 4)

def frustum_overlap_mask_torch(depth_prev, K, T_prev, T_cur):
    """
    Returns a mask [H,W] of pixels in the previous view whose 3D points lie **inside**
    the camera frustum of the current view.
    
    Only checks:
        - valid depth
        - transforms in front of cur cam (z > 0)
        - projection lands inside the image bounds
    No depth comparison or occlusion reasoning.
    """

    device = depth_prev.device
    H, W = depth_prev.shape

    # 1) pixel grid
    ys = torch.arange(0, H, device=device).view(-1,1).expand(H,W).float()
    xs = torch.arange(0, W, device=device).view(1,-1).expand(H,W).float()
    ones = torch.ones_like(xs)
    pix_h = torch.stack([xs, ys, ones], dim=-1)  # H,W,3

    # 2) valid depth mask
    valid_prev = depth_prev > 0

    # 3) unproject previous depth to prev camera frame
    K_inv = torch.linalg.inv(K)
    pix_flat = pix_h.view(-1,3).T                 # 3, N
    rays = (K_inv @ pix_flat).T.view(H,W,3)        # H, W, 3
    X_prev_cam = rays * depth_prev.unsqueeze(-1)   # H, W, 3

    # turn into homogeneous
    X_prev_h = torch.cat([X_prev_cam, torch.ones(H,W,1, device=device)], dim=-1)  \
                    .view(-1,4).T  # 4, N

    # 4) Transform to current camera frame
    # X_cur = T_cur^{-1} * T_prev * X_prev
    T = torch.linalg.inv(T_cur) @ T_prev
    X_cur = (T @ X_prev_h)[:3].view(3, H, W)   # 3,H,W
    z_cur = X_cur[2]

    # 5) project to current image
    proj = (K @ X_cur.view(3,-1)).T   # N,3
    u_proj = (proj[:,0] / proj[:,2]).view(H,W)
    v_proj = (proj[:,1] / proj[:,2]).view(H,W)

    # 6) check frustum conditions
    in_front = z_cur > 0
    in_bounds = (u_proj >= 0) & (u_proj <= W-1) & \
                (v_proj >= 0) & (v_proj <= H-1)

    # Final mask
    mask = valid_prev & in_front & in_bounds
    return mask


def check_for_validity(rendered, dep, rendered_previous, dep_prev, K, Rt, Rt_prev, gt, gt_prev):
    vis_m = frustum_overlap_mask_torch(dep_prev, K, Rt_prev, Rt).unsqueeze(0).repeat(3,1,1)
    visual_diff = (rendered - gt).abs()
    visual_diff[~vis_m] = 0.0
    if vis_m.float().mean()<0.4:
        return False
    if visual_diff.mean()>0.8:
        return False
    
    return True

def knn_cuda(gauss_pos, ctrl_pos, kneighbors=4,batch_size=65536):
    """
    Fast CUDA k-NN search (exact) with torch.cdist + topk, in chunks.
    Returns:
    idx: [N,K] long tensor (CUDA)
    d2:  [N,K] float tensor (CUDA)  — squared distances
    """
    N = gauss_pos.shape[0]

    all_idx = []
    all_d2  = []

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        Xb = gauss_pos[start:end]         # (B,3)

        d2 = torch.cdist(Xb, ctrl_pos, p=2.0, compute_mode="use_mm_for_euclid_dist") ** 2

        # top-k smallest distances
        d2b, idxb = torch.topk(d2, k=kneighbors, largest=False, sorted=False)

        all_idx.append(idxb)
        all_d2.append(d2b)

    idx = torch.cat(all_idx, dim=0)  # (N,K)
    d2  = torch.cat(all_d2,  dim=0)  # (N,K)
    return idx, d2


def quat_mul(q1, q2):
    # q format: (x, y, z, w)
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack([x, y, z, w], dim=-1)


def apply_matrix_rotation_to_quaternions(q, R):
    """
    q: (N,4) or (4,)  quaternion(s) representing local->world (w,x,y,z)
    R: (3,3)  or (N,3,3) rotation matrix that maps vectors as v' = R @ v
       (e.g. world->cam)
    returns: q_after same shape as q, representing local->R-frame
    """
    # make batch dims
    q = q.reshape(-1,4)
    n = q.shape[0]

    # expand/reshape R to (n,3,3)
    if R.ndim == 2:
        Rb = R.unsqueeze(0).expand(n, -1, -1).contiguous()
    else:
        Rb = R.reshape(n,3,3)

    qR = rotmat_to_quat(Rb)        # (n,4)
    # If R maps world -> cam and q maps local -> world, local -> cam = R @ (local->world)
    # so quaternion composition: q_after = qR ⊗ q
    q_after = quat_mul(qR, q)
    # normalize
    q_after = q_after / (q_after.norm(dim=1, keepdim=True) + 1e-12)
    return q_after.view(*q.shape[:-1], 4)