import json
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import prefilter_voxel, render, render_with_dense_scaffold
from scene import GaussianModel, Scene
from scene.continuous_scaffold_field import ContinuousScaffoldField
from utils.graphics_utils import fov2focal, getWorld2View2
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

try:
    from lpipsPyTorch import LPIPS

    def build_lpips():
        return LPIPS(net_type="vgg").to("cuda")
except Exception:
    build_lpips = None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


class ResidualScaffoldField(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 64, k_neighbors: int = 8, temperature: float = 0.05, knn_chunk_size: int = 8192):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.knn_chunk_size = knn_chunk_size

        local_in_dim = feat_dim + 3 + 3 + 3 + 1
        self.local_mlp = nn.Sequential(
            nn.Linear(local_in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh(),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _knn(self, query_xyz: torch.Tensor, anchor_xyz: torch.Tensor):
        k = min(self.k_neighbors, anchor_xyz.shape[0])
        dist_chunks = []
        idx_chunks = []
        for start in range(0, query_xyz.shape[0], self.knn_chunk_size):
            end = min(start + self.knn_chunk_size, query_xyz.shape[0])
            dists = torch.cdist(query_xyz[start:end], anchor_xyz)
            knn_dist, knn_idx = torch.topk(dists, k=k, dim=1, largest=False, sorted=True)
            dist_chunks.append(knn_dist)
            idx_chunks.append(knn_idx)
        return torch.cat(dist_chunks, dim=0), torch.cat(idx_chunks, dim=0)

    def forward(self, query_xyz, query_cell, query_viewdir, anchor_xyz, anchor_feat):
        knn_dist, knn_idx = self._knn(query_xyz, anchor_xyz)
        neigh_xyz = anchor_xyz[knn_idx]
        neigh_feat = anchor_feat[knn_idx]
        rel_xyz = query_xyz[:, None, :] - neigh_xyz
        cell = query_cell[:, None, :].expand_as(rel_xyz)
        viewdir = query_viewdir[:, None, :].expand_as(rel_xyz)
        dist_feat = knn_dist.unsqueeze(-1)
        local_input = torch.cat([neigh_feat, rel_xyz, cell, viewdir, dist_feat], dim=-1)
        local_feat = self.local_mlp(local_input)
        weights = torch.softmax(-knn_dist / self.temperature, dim=1).unsqueeze(-1)
        fused = (weights * local_feat).sum(dim=1)
        fused = self.fuse_mlp(fused)
        residual_rgb = self.rgb_head(torch.cat([fused, query_viewdir, query_cell], dim=-1))
        confidence = self.conf_head(fused)
        return residual_rgb, confidence


def copy_shared_weights(residual_field: ResidualScaffoldField, dense_field: ContinuousScaffoldField):
    dense_field.local_mlp.load_state_dict(residual_field.local_mlp.state_dict())
    dense_field.fuse_mlp.load_state_dict(residual_field.fuse_mlp.state_dict())
    dense_field.conf_head.load_state_dict(residual_field.conf_head.state_dict())


def freeze_stage1(gaussians: GaussianModel):
    tensors = [gaussians._anchor, gaussians._offset, gaussians._anchor_feat, gaussians._scaling, gaussians._rotation, gaussians._opacity]
    for t in tensors:
        if isinstance(t, torch.Tensor):
            t.requires_grad_(False)
    modules = [gaussians.mlp_opacity, gaussians.mlp_cov, gaussians.mlp_color]
    if gaussians.use_feat_bank:
        modules.append(gaussians.mlp_feature_bank)
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(False)
    if gaussians.appearance_dim > 0 and gaussians.embedding_appearance is not None:
        for p in gaussians.embedding_appearance.parameters():
            p.requires_grad_(False)


def camera_c2w(camera):
    w2c = getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)
    return np.linalg.inv(w2c)


def pixels_to_world(camera, uv: torch.Tensor, depth: torch.Tensor):
    h = camera.image_height
    w = camera.image_width
    fx = fov2focal(camera.FoVx, w)
    fy = fov2focal(camera.FoVy, h)
    cx = w * 0.5
    cy = h * 0.5
    u = uv[:, 0].float() + 0.5
    v = uv[:, 1].float() + 0.5
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    xyz_cam = torch.stack([x, y, z], dim=-1)
    c2w = torch.from_numpy(camera_c2w(camera)).to(depth.device, dtype=depth.dtype)
    xyz_world = xyz_cam @ c2w[:3, :3].T + c2w[:3, 3]
    cell_x = depth / fx
    cell_y = depth / fy
    cell_z = 0.5 * (cell_x + cell_y)
    cell = torch.stack([cell_x, cell_y, cell_z], dim=-1)
    cam_center = camera.camera_center[None, :].to(depth.device, dtype=depth.dtype)
    viewdir = xyz_world - cam_center
    viewdir = viewdir / viewdir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return xyz_world, cell, viewdir


def sample_valid_pixels(depth: torch.Tensor, num_samples: int):
    valid = torch.isfinite(depth) & (depth > 1e-6)
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.shape[0] == 0:
        raise RuntimeError("No valid depth pixels found.")
    if coords.shape[0] <= num_samples:
        chosen = coords
    else:
        perm = torch.randperm(coords.shape[0], device=depth.device)[:num_samples]
        chosen = coords[perm]
    return torch.stack([chosen[:, 1], chosen[:, 0]], dim=-1)


def build_query_template(device, dtype):
    vals = [-0.25, 0.25]
    template = []
    for x in vals:
        for y in vals:
            for z in vals:
                template.append([x, y, z])
    return torch.tensor(template, device=device, dtype=dtype)


def build_dense_scaffold(viewpoint_cam, gaussians, field, visible_mask, max_parent_anchors, conf_threshold):
    base_anchor_all = gaussians.get_anchor.detach()
    base_feat_all = gaussians._anchor_feat.detach()
    parent_anchor = base_anchor_all[visible_mask]
    parent_scaling = gaussians.get_scaling.detach()[visible_mask]
    parent_offsets = gaussians._offset.detach()[visible_mask]

    if parent_anchor.shape[0] > max_parent_anchors:
        perm = torch.randperm(parent_anchor.shape[0], device=parent_anchor.device)[:max_parent_anchors]
        parent_anchor = parent_anchor[perm]
        parent_scaling = parent_scaling[perm]
        parent_offsets = parent_offsets[perm]

    template = build_query_template(parent_anchor.device, parent_anchor.dtype)
    q = template.shape[0]
    spatial_scale = parent_scaling[:, None, :3]
    query_xyz = parent_anchor[:, None, :] + template[None, :, :] * spatial_scale
    query_xyz = query_xyz.reshape(-1, 3)
    query_cell = spatial_scale.repeat(1, q, 1).reshape(-1, 3)
    cam_center = viewpoint_cam.camera_center[None, :].to(query_xyz.device, dtype=query_xyz.dtype)
    query_viewdir = query_xyz - cam_center
    query_viewdir = query_viewdir / query_viewdir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    dense_feat, conf, _, _ = field(query_xyz, query_cell, query_viewdir, base_anchor_all, base_feat_all)
    conf = conf.view(-1)
    keep = conf >= conf_threshold
    if keep.sum() == 0:
        topk = min(1024, conf.shape[0])
        keep_idx = torch.topk(conf, k=topk, largest=True).indices
        keep = torch.zeros_like(conf, dtype=torch.bool)
        keep[keep_idx] = True

    parent_scaling_rep = parent_scaling[:, None, :].repeat(1, q, 1).reshape(-1, parent_scaling.shape[-1])
    parent_offsets_rep = parent_offsets[:, None, :, :].repeat(1, q, 1, 1).reshape(-1, parent_offsets.shape[1], 3)
    return query_xyz[keep], dense_feat[keep], parent_scaling_rep[keep], parent_offsets_rep[keep], conf[keep]


def render_dense_image(viewpoint_cam, gaussians, field, pipe, background, max_parent_anchors, conf_threshold):
    visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
    dense_anchor_xyz, dense_anchor_feat, dense_scaling, dense_offsets, dense_conf = build_dense_scaffold(
        viewpoint_cam, gaussians, field, visible_mask, max_parent_anchors, conf_threshold
    )
    pkg = render_with_dense_scaffold(
        viewpoint_cam, gaussians, dense_anchor_xyz, dense_anchor_feat, dense_scaling, dense_offsets, pipe, background, retain_grad=True
    )
    pkg["dense_confidence"] = dense_conf
    pkg["dense_anchor_count"] = int(dense_anchor_xyz.shape[0])
    return pkg


def eval_residual(field, scene, gaussians, pipe, depth_dir, background, max_views=8):
    lpips_fn = build_lpips() if build_lpips is not None else None
    metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}
    anchor_xyz = gaussians.get_anchor.detach()
    anchor_feat = gaussians._anchor_feat.detach()
    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    for idx, cam in enumerate(cameras):
        if idx >= max_views:
            break
        depth = torch.from_numpy(np.load(depth_dir / f"{cam.image_name}.npy").astype(np.float32)).to("cuda")
        with torch.no_grad():
            voxel_visible_mask = prefilter_voxel(cam, gaussians, pipe, background)
            base = torch.clamp(render(cam, gaussians, pipe, background, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            valid = torch.isfinite(depth) & (depth > 1e-6)
            coords = torch.nonzero(valid, as_tuple=False)
            uv = torch.stack([coords[:, 1], coords[:, 0]], dim=-1)
            xyz, cell, viewdir = pixels_to_world(cam, uv, depth[coords[:, 0], coords[:, 1]])
            delta, conf = field(xyz, cell, viewdir, anchor_xyz, anchor_feat)
            pred = base.clone()
            base_rgb = base[:, coords[:, 0], coords[:, 1]].permute(1, 0)
            pred_rgb = torch.clamp(base_rgb + conf * delta, 0.0, 1.0)
            pred[:, coords[:, 0], coords[:, 1]] = pred_rgb.permute(1, 0)
            metrics["psnr"] += psnr(pred, gt).mean().item()
            metrics["ssim"] += ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            if lpips_fn is not None:
                metrics["lpips"] += lpips_fn(pred.unsqueeze(0), gt.unsqueeze(0)).mean().item()
            metrics["count"] += 1
    for k in ("psnr", "ssim", "lpips"):
        metrics[k] /= max(metrics["count"], 1)
    return metrics


def eval_dense(field, scene, gaussians, pipe, background, max_parent_anchors, conf_threshold, max_views=8):
    lpips_fn = build_lpips() if build_lpips is not None else None
    metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}
    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    for idx, cam in enumerate(cameras):
        if idx >= max_views:
            break
        with torch.no_grad():
            pkg = render_dense_image(cam, gaussians, field, pipe, background, max_parent_anchors, conf_threshold)
            pred = torch.clamp(pkg["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            metrics["psnr"] += psnr(pred, gt).mean().item()
            metrics["ssim"] += ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            if lpips_fn is not None:
                metrics["lpips"] += lpips_fn(pred.unsqueeze(0), gt.unsqueeze(0)).mean().item()
            metrics["count"] += 1
    for k in ("psnr", "ssim", "lpips"):
        metrics[k] /= max(metrics["count"], 1)
    return metrics


def main():
    parser = ArgumentParser(description="Stage2 transition: residual warmup -> dense scaffold takeover")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--stage2_output", type=str, required=True)
    parser.add_argument("--stage1_iteration", type=int, default=30000)
    parser.add_argument("--warmup_iterations", type=int, default=5000)
    parser.add_argument("--takeover_iterations", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr_residual", type=float, default=1e-3)
    parser.add_argument("--lr_dense", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max_parent_anchors", type=int, default=2000)
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--confidence_reg_weight", type=float, default=0.01)
    parser.add_argument("--depth_dir_name", type=str, default="depth")
    args = get_combined_args(parser)

    stage2_output = Path(args.stage2_output).resolve()
    ensure_dir(stage2_output)
    (stage2_output / "stage2_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    dataset = model.extract(args)
    dataset.eval = True
    dataset.resolution = 1
    pipe = pipeline.extract(args)
    depth_dir = Path(dataset.source_path) / args.depth_dir_name
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor,
        dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio,
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist
    )
    scene = Scene(dataset, gaussians, load_iteration=args.stage1_iteration, shuffle=False)
    gaussians.eval()
    freeze_stage1(gaussians)

    residual_field = ResidualScaffoldField(dataset.feat_dim, args.hidden_dim, args.k_neighbors, args.temperature).cuda()
    residual_optimizer = torch.optim.Adam(residual_field.parameters(), lr=args.lr_residual)

    train_cameras = scene.getTrainCameras()
    anchor_xyz = gaussians.get_anchor.detach()
    anchor_feat = gaussians._anchor_feat.detach()

    progress = tqdm(range(1, args.warmup_iterations + args.takeover_iterations + 1), desc="Stage2 transition")
    for iteration in progress:
        cam = train_cameras[randint(0, len(train_cameras) - 1)]

        if iteration <= args.warmup_iterations:
            depth = torch.from_numpy(np.load(depth_dir / f"{cam.image_name}.npy").astype(np.float32)).to("cuda")
            uv = sample_valid_pixels(depth, args.batch_size)
            with torch.no_grad():
                voxel_visible_mask = prefilter_voxel(cam, gaussians, pipe, background)
                base = torch.clamp(render(cam, gaussians, pipe, background, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            sampled_depth = depth[uv[:, 1], uv[:, 0]]
            xyz, cell, viewdir = pixels_to_world(cam, uv, sampled_depth)
            delta, conf = residual_field(xyz, cell, viewdir, anchor_xyz, anchor_feat)
            base_rgb = base[:, uv[:, 1], uv[:, 0]].permute(1, 0)
            gt_rgb = gt[:, uv[:, 1], uv[:, 0]].permute(1, 0)
            pred_rgb = torch.clamp(base_rgb + conf * delta, 0.0, 1.0)
            loss_rgb = l1_loss(pred_rgb, gt_rgb)
            loss_res = delta.abs().mean()
            loss = loss_rgb + 0.1 * loss_res
            residual_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            residual_optimizer.step()
            progress.set_postfix({"stage": "warmup", "loss": f"{loss.item():.5f}", "rgb": f"{loss_rgb.item():.5f}"})

            if iteration % args.eval_interval == 0 or iteration == args.warmup_iterations:
                residual_field.eval()
                metrics = eval_residual(residual_field, scene, gaussians, pipe, depth_dir, background)
                with open(stage2_output / "metrics_history.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iteration": iteration, "mode": "warmup_residual", **metrics}) + "\n")
                residual_field.train()

            if iteration % args.save_interval == 0 or iteration == args.warmup_iterations:
                torch.save({"iteration": iteration, "field": residual_field.state_dict(), "mode": "warmup_residual"},
                           stage2_output / f"warmup_{iteration}.pth")

            if iteration == args.warmup_iterations:
                dense_field = ContinuousScaffoldField(dataset.feat_dim, args.hidden_dim, args.k_neighbors, args.temperature).cuda()
                copy_shared_weights(residual_field, dense_field)
                dense_optimizer = torch.optim.Adam(dense_field.parameters(), lr=args.lr_dense)

        else:
            pkg = render_dense_image(cam, gaussians, dense_field, pipe, background, args.max_parent_anchors, args.conf_threshold)
            pred = torch.clamp(pkg["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            loss_rgb = l1_loss(pred, gt)
            loss_ssim = 1.0 - ssim(pred.unsqueeze(0), gt.unsqueeze(0))
            loss_conf = 1.0 - pkg["dense_confidence"].mean()
            loss = 0.8 * loss_rgb + 0.2 * loss_ssim + args.confidence_reg_weight * loss_conf
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            dense_optimizer.step()
            progress.set_postfix({
                "stage": "dense",
                "loss": f"{loss.item():.5f}",
                "l1": f"{loss_rgb.item():.5f}",
                "anchors": pkg["dense_anchor_count"],
            })

            takeover_iter = iteration - args.warmup_iterations
            if takeover_iter % args.eval_interval == 0 or takeover_iter == args.takeover_iterations:
                dense_field.eval()
                metrics = eval_dense(dense_field, scene, gaussians, pipe, background, args.max_parent_anchors, args.conf_threshold)
                with open(stage2_output / "metrics_history.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iteration": iteration, "mode": "dense_takeover", **metrics}) + "\n")
                dense_field.train()

            if takeover_iter % args.save_interval == 0 or takeover_iter == args.takeover_iterations:
                torch.save({"iteration": iteration, "field": dense_field.state_dict(), "mode": "dense_takeover"},
                           stage2_output / f"dense_{takeover_iter}.pth")

    (stage2_output / "summary.json").write_text(
        json.dumps(
            {
                "stage1_model_path": dataset.model_path,
                "stage1_iteration": args.stage1_iteration,
                "stage2_output": str(stage2_output),
                "schedule": {"warmup_iterations": args.warmup_iterations, "takeover_iterations": args.takeover_iterations},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
