import json
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import prefilter_voxel, render, render_with_appended_dense_scaffold
from scene import GaussianModel, Scene
from scene.continuous_scaffold_field import ContinuousScaffoldField
from utils.graphics_utils import fov2focal, getWorld2View2
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from lpipsPyTorch import LPIPS

    def build_lpips():
        return LPIPS(net_type="vgg").to("cuda")
except Exception:
    build_lpips = None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def freeze_stage1(gaussians: GaussianModel, finetune_decoder: bool = True):
    tensors = [
        gaussians._anchor,
        gaussians._offset,
        gaussians._anchor_feat,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
    ]
    for t in tensors:
        if isinstance(t, torch.Tensor):
            t.requires_grad_(False)

    modules = [gaussians.mlp_opacity, gaussians.mlp_cov, gaussians.mlp_color]
    if gaussians.use_feat_bank:
        modules.append(gaussians.mlp_feature_bank)
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(finetune_decoder)

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


def load_depth(depth_dir: Path, image_name: str) -> torch.Tensor:
    arr = np.load(depth_dir / f"{image_name}.npy").astype(np.float32)
    return torch.from_numpy(arr).to("cuda")


def choose_error_pixels(error_map: torch.Tensor, depth: torch.Tensor, max_candidates: int, error_threshold: float):
    valid = torch.isfinite(depth) & (depth > 1e-6)
    score = error_map.clone()
    score[~valid] = -1.0
    flat = score.view(-1)
    valid_flat = flat > error_threshold
    if valid_flat.sum() == 0:
        topk = min(max_candidates, flat.numel())
        idx = torch.topk(flat, k=topk, largest=True).indices
    else:
        idx = torch.nonzero(valid_flat, as_tuple=False).view(-1)
        if idx.numel() > max_candidates:
            vals = flat[idx]
            keep = torch.topk(vals, k=max_candidates, largest=True).indices
            idx = idx[keep]
    y = idx // error_map.shape[1]
    x = idx % error_map.shape[1]
    return torch.stack([x, y], dim=-1)


def voxelize_points(points_xyz: torch.Tensor, voxel_size: float):
    return torch.round(points_xyz / voxel_size).to(torch.int64)


def build_voted_dense_scaffold(
    scene,
    gaussians: GaussianModel,
    pipe,
    background,
    depth_dir: Path,
    voxel_size: float,
    max_candidates_per_view: int,
    error_threshold: float,
    vote_threshold: int,
    max_voted_anchors: int,
):
    cameras = scene.getTrainCameras()
    vote_dict = {}

    for cam in tqdm(cameras, desc="Building voted dense scaffold"):
        with torch.no_grad():
            visible = prefilter_voxel(cam, gaussians, pipe, background)
            pred = torch.clamp(render(cam, gaussians, pipe, background, visible_mask=visible)["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            error_map = (pred - gt).abs().mean(dim=0)
            depth = load_depth(depth_dir, cam.image_name)
            uv = choose_error_pixels(error_map, depth, max_candidates_per_view, error_threshold)
            sampled_depth = depth[uv[:, 1], uv[:, 0]]
            xyz_world, _, _ = pixels_to_world(cam, uv, sampled_depth)
            vox = voxelize_points(xyz_world, voxel_size)
            unique_vox = torch.unique(vox, dim=0)
            for row in unique_vox.cpu().tolist():
                key = tuple(int(v) for v in row)
                vote_dict[key] = vote_dict.get(key, 0) + 1

    voted_keys = [k for k, v in vote_dict.items() if v >= vote_threshold]
    voted_scores = [vote_dict[k] for k in voted_keys]
    if len(voted_keys) == 0:
        raise RuntimeError("No voted dense scaffold positions survived thresholding.")

    if len(voted_keys) > max_voted_anchors:
        order = np.argsort(np.array(voted_scores))[::-1][:max_voted_anchors]
        voted_keys = [voted_keys[i] for i in order]

    dense_xyz = torch.tensor(voted_keys, dtype=torch.float32, device="cuda") * voxel_size
    return dense_xyz


def assign_parent_attributes(dense_xyz: torch.Tensor, gaussians: GaussianModel, knn_chunk_size: int = 8192):
    anchor_xyz = gaussians.get_anchor.detach()
    anchor_feat = gaussians._anchor_feat.detach()
    parent_scaling = gaussians.get_scaling.detach()
    parent_offsets = gaussians._offset.detach()

    idx_chunks = []
    for start in range(0, dense_xyz.shape[0], knn_chunk_size):
        end = min(start + knn_chunk_size, dense_xyz.shape[0])
        dists = torch.cdist(dense_xyz[start:end], anchor_xyz)
        idx = torch.argmin(dists, dim=1)
        idx_chunks.append(idx)
    nn_idx = torch.cat(idx_chunks, dim=0)
    return anchor_xyz, anchor_feat, parent_scaling[nn_idx], parent_offsets[nn_idx]


def render_stage2_image(
    viewpoint_cam,
    gaussians,
    field,
    dense_xyz,
    parent_scaling,
    parent_offsets,
    anchor_xyz_all,
    anchor_feat_all,
    pipe,
    background,
):
    visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
    cam_center = viewpoint_cam.camera_center[None, :].to(dense_xyz.device, dtype=dense_xyz.dtype)
    query_viewdir = dense_xyz - cam_center
    query_viewdir = query_viewdir / query_viewdir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    query_cell = parent_scaling[:, :3].detach()
    dense_feat, conf, _, _ = field(dense_xyz, query_cell, query_viewdir, anchor_xyz_all, anchor_feat_all)

    pkg = render_with_appended_dense_scaffold(
        viewpoint_cam,
        gaussians,
        visible_mask,
        dense_xyz,
        dense_feat,
        parent_scaling,
        parent_offsets,
        pipe,
        background,
        retain_grad=True,
    )
    pkg["dense_confidence"] = conf.view(-1)
    pkg["dense_anchor_count"] = int(dense_xyz.shape[0])
    return pkg


def evaluate(
    field,
    scene,
    gaussians,
    dense_xyz,
    parent_scaling,
    parent_offsets,
    anchor_xyz_all,
    anchor_feat_all,
    pipe,
    background,
    out_dir: Path,
    max_views=8,
):
    ensure_dir(out_dir)
    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    lpips_fn = build_lpips() if build_lpips is not None else None
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}

    for idx, cam in enumerate(cameras):
        if idx >= max_views:
            break
        with torch.no_grad():
            pkg = render_stage2_image(
                cam,
                gaussians,
                field,
                dense_xyz,
                parent_scaling,
                parent_offsets,
                anchor_xyz_all,
                anchor_feat_all,
                pipe,
                background,
            )
            pred = torch.clamp(pkg["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            totals["psnr"] += psnr(pred, gt).mean().item()
            totals["ssim"] += ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            if lpips_fn is not None:
                totals["lpips"] += lpips_fn(pred.unsqueeze(0), gt.unsqueeze(0)).mean().item()
            totals["count"] += 1

    if totals["count"] == 0:
        return None
    for k in ("psnr", "ssim", "lpips"):
        totals[k] /= totals["count"]
    return totals


def main():
    parser = ArgumentParser(description="Stage-2 x2 training with voting-based dense scaffold rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--stage2_output", type=str, required=True)
    parser.add_argument("--stage1_iteration", type=int, default=30000)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--confidence_reg_weight", type=float, default=0.01)
    parser.add_argument("--depth_dir_name", type=str, default="depth")
    parser.add_argument("--vote_voxel_size", type=float, default=0.03)
    parser.add_argument("--max_candidates_per_view", type=int, default=6000)
    parser.add_argument("--error_threshold", type=float, default=0.08)
    parser.add_argument("--vote_threshold", type=int, default=2)
    parser.add_argument("--max_voted_anchors", type=int, default=12000)
    parser.add_argument("--decoder_lr_scale", type=float, default=0.1)
    args = get_combined_args(parser)

    stage2_output = Path(args.stage2_output).resolve()
    ensure_dir(stage2_output)
    with open(stage2_output / "stage2_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset = model.extract(args)
    dataset.eval = True
    dataset.resolution = 1
    pipe = pipeline.extract(args)
    depth_dir = Path(dataset.source_path) / args.depth_dir_name

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
    )
    scene = Scene(dataset, gaussians, load_iteration=args.stage1_iteration, shuffle=False)
    gaussians.train()
    freeze_stage1(gaussians, finetune_decoder=True)

    dense_xyz = build_voted_dense_scaffold(
        scene,
        gaussians,
        pipe,
        background,
        depth_dir=depth_dir,
        voxel_size=args.vote_voxel_size,
        max_candidates_per_view=args.max_candidates_per_view,
        error_threshold=args.error_threshold,
        vote_threshold=args.vote_threshold,
        max_voted_anchors=args.max_voted_anchors,
    )
    anchor_xyz_all, anchor_feat_all, parent_scaling, parent_offsets = assign_parent_attributes(dense_xyz, gaussians)

    torch.save(
        {
            "dense_xyz": dense_xyz.detach().cpu(),
            "parent_scaling": parent_scaling.detach().cpu(),
            "parent_offsets": parent_offsets.detach().cpu(),
        },
        stage2_output / "voted_dense_scaffold.pth",
    )

    field = ContinuousScaffoldField(
        feat_dim=dataset.feat_dim,
        hidden_dim=args.hidden_dim,
        k_neighbors=args.k_neighbors,
        temperature=args.temperature,
    ).cuda()
    optim_params = [{"params": field.parameters(), "lr": args.lr}]
    decoder_modules = [gaussians.mlp_opacity, gaussians.mlp_cov, gaussians.mlp_color]
    if gaussians.use_feat_bank:
        decoder_modules.append(gaussians.mlp_feature_bank)
    for module in decoder_modules:
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            optim_params.append({"params": params, "lr": args.lr * args.decoder_lr_scale})
    optimizer = torch.optim.Adam(optim_params)

    train_cameras = scene.getTrainCameras()
    progress = tqdm(range(1, args.iterations + 1), desc="Stage2 voting x2")
    for iteration in progress:
        cam = train_cameras[randint(0, len(train_cameras) - 1)]
        pkg = render_stage2_image(
            cam,
            gaussians,
            field,
            dense_xyz,
            parent_scaling,
            parent_offsets,
            anchor_xyz_all,
            anchor_feat_all,
            pipe,
            background,
        )
        pred = torch.clamp(pkg["render"], 0.0, 1.0)
        gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)

        loss_rgb = l1_loss(pred, gt)
        loss_ssim = 1.0 - ssim(pred.unsqueeze(0), gt.unsqueeze(0))
        loss_conf = 1.0 - pkg["dense_confidence"].mean()
        loss = 0.8 * loss_rgb + 0.2 * loss_ssim + args.confidence_reg_weight * loss_conf

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix(
            {
                "loss": f"{loss.item():.5f}",
                "l1": f"{loss_rgb.item():.5f}",
                "ssim": f"{loss_ssim.item():.5f}",
                "anchors": pkg["dense_anchor_count"],
            }
        )

        if iteration % args.save_interval == 0 or iteration == args.iterations:
            torch.save(
                {
                    "iteration": iteration,
                    "field": field.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "stage1_model_path": dataset.model_path,
                    "stage1_iteration": args.stage1_iteration,
                },
                stage2_output / f"stage2_{iteration}.pth",
            )

        if iteration % args.eval_interval == 0 or iteration == args.iterations:
            field.eval()
            metrics = evaluate(
                field,
                scene,
                gaussians,
                dense_xyz,
                parent_scaling,
                parent_offsets,
                anchor_xyz_all,
                anchor_feat_all,
                pipe,
                background,
                out_dir=stage2_output / f"eval_{iteration}",
                max_views=8,
            )
            if metrics is not None:
                with open(stage2_output / "metrics_history.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iteration": iteration, **metrics}) + "\n")
            field.train()

    summary = {
        "stage1_model_path": dataset.model_path,
        "stage1_iteration": args.stage1_iteration,
        "stage2_output": str(stage2_output),
        "mode": "voting_dense_scaffold_rendering",
        "vote_voxel_size": args.vote_voxel_size,
        "vote_threshold": args.vote_threshold,
        "max_voted_anchors": args.max_voted_anchors,
    }
    (stage2_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
