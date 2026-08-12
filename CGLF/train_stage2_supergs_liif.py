import json
import re
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import prefilter_voxel, render_with_feat_override
from scene import GaussianModel, Scene
from scene.liif_anchor_field import LIIFAnchorField
from utils.image_utils import psnr
from utils.loss_utils import ssim

try:
    from lpipsPyTorch import LPIPS

    def build_lpips():
        return LPIPS(net_type="vgg").to("cuda")
except Exception:
    build_lpips = None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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
    y = torch.div(idx, error_map.shape[1], rounding_mode="floor")
    x = idx % error_map.shape[1]
    return torch.stack([x, y], dim=-1)


def pixels_to_world(camera, uv: torch.Tensor, depth: torch.Tensor):
    from utils.graphics_utils import fov2focal, getWorld2View2

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
    w2c = getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)
    c2w = torch.from_numpy(np.linalg.inv(w2c)).to(depth.device, dtype=depth.dtype)
    xyz_world = xyz_cam @ c2w[:3, :3].T + c2w[:3, 3]
    return xyz_world


def voxelize_points(points_xyz: torch.Tensor, voxel_size: float):
    return torch.round(points_xyz / voxel_size).to(torch.int64)


def build_voted_anchor_positions(scene, gaussians, liif_field, pipe, background, depth_dir: Path, voxel_size: float,
                                 max_candidates_per_view: int, error_threshold: float, vote_threshold: int,
                                 max_new_anchors: int, liif_k_growth: int = None, base_anchor_count: int = None):
    vote_dict = {}
    cameras = scene.getTrainCameras()
    for cam in cameras:
        with torch.no_grad():
            visible = prefilter_voxel(cam, gaussians, pipe, background)
            feat_override, unc = build_stage2_feat_for_visible(gaussians, liif_field, visible, k_override=liif_k_growth, base_anchor_count=base_anchor_count, deterministic=True)
            pred = torch.clamp(
                render_with_feat_override(
                    cam, gaussians, feat_override, unc, pipe, background, visible_mask=visible
                )["render"],
                0.0,
                1.0,
            )
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            error_map = (pred - gt).abs().mean(dim=0)
            depth = load_depth(depth_dir, cam.image_name)
            if depth.shape != error_map.shape:
                depth = F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=error_map.shape,
                    mode="nearest",
                ).squeeze(0).squeeze(0)
            uv = choose_error_pixels(error_map, depth, max_candidates_per_view, error_threshold)
            sampled_depth = depth[uv[:, 1], uv[:, 0]]
            xyz_world = pixels_to_world(cam, uv, sampled_depth)
            vox = voxelize_points(xyz_world, voxel_size)
            unique_vox = torch.unique(vox, dim=0)
            for row in unique_vox.cpu().tolist():
                key = tuple(int(v) for v in row)
                vote_dict[key] = vote_dict.get(key, 0) + 1
    voted_keys = [k for k, v in vote_dict.items() if v >= vote_threshold]
    voted_scores = [vote_dict[k] for k in voted_keys]
    if len(voted_keys) == 0:
        return torch.empty((0, 3), device="cuda")
    if len(voted_keys) > max_new_anchors:
        order = np.argsort(np.array(voted_scores))[::-1][:max_new_anchors]
        voted_keys = [voted_keys[i] for i in order]
    return torch.tensor(voted_keys, dtype=torch.float32, device="cuda") * voxel_size


def freeze_for_stage2(gaussians: GaussianModel, freeze_color_head: bool = False):
    for t in [gaussians._anchor, gaussians._offset, gaussians._anchor_feat, gaussians._scaling, gaussians._rotation, gaussians._opacity]:
        if isinstance(t, torch.Tensor):
            t.requires_grad_(False)
    for p in gaussians.mlp_opacity.parameters():
        p.requires_grad_(False)
    for p in gaussians.mlp_cov.parameters():
        p.requires_grad_(False)
    for p in gaussians.mlp_color.parameters():
        p.requires_grad_(not freeze_color_head)
    if gaussians.use_feat_bank:
        for p in gaussians.mlp_feature_bank.parameters():
            p.requires_grad_(not freeze_color_head)
    if gaussians.embedding_appearance is not None:
        for p in gaussians.embedding_appearance.parameters():
            p.requires_grad_(False)


def build_stage2_optimizer(gaussians: GaussianModel, liif_field: LIIFAnchorField, lr_sr: float, lr_decoder: float, lr_liif: float):
    params = [{"params": liif_field.parameters(), "lr": lr_liif}]
    if isinstance(getattr(gaussians, "_anchor_feat_sr_mu", None), torch.nn.Parameter) and gaussians._anchor_feat_sr_mu.numel() > 0:
        params.append({"params": [gaussians._anchor_feat_sr_mu], "lr": lr_sr})
    if isinstance(getattr(gaussians, "_anchor_feat_sr_logvar", None), torch.nn.Parameter) and gaussians._anchor_feat_sr_logvar.numel() > 0:
        params.append({"params": [gaussians._anchor_feat_sr_logvar], "lr": lr_sr})
    color_params = [p for p in gaussians.mlp_color.parameters() if p.requires_grad]
    if color_params:
        params.append({"params": color_params, "lr": lr_decoder})
    if gaussians.use_feat_bank:
        bank_params = [p for p in gaussians.mlp_feature_bank.parameters() if p.requires_grad]
        if bank_params:
            params.append({"params": bank_params, "lr": lr_decoder})
    return torch.optim.Adam(params)


def uncertainty_guided_loss(pred, gt, uncertainty_map):
    weight = 1.0 - torch.sigmoid(uncertainty_map)
    return (weight * (pred - gt).abs()).mean()


def build_query_cell(anchor_xyz: torch.Tensor, voxel_size: float):
    return torch.full_like(anchor_xyz, float(voxel_size))


def build_stage2_feat_for_visible(
    gaussians: GaussianModel,
    liif_field: LIIFAnchorField,
    visible_mask: torch.Tensor,
    k_override: int = None,
    base_anchor_count: int = None,
    deterministic: bool = False,
):
    anchor = gaussians.get_anchor[visible_mask]
    context_xyz = gaussians.get_anchor.detach()
    context_feat = gaussians._anchor_feat.detach()
    query_cell = build_query_cell(anchor, gaussians.voxel_size)
    field_feat = liif_field(anchor, query_cell, context_xyz, context_feat, k_override=k_override)

    if gaussians._anchor_feat_sr_mu.numel() > 0:
        mu = gaussians._anchor_feat_sr_mu[visible_mask]
        logvar = gaussians._anchor_feat_sr_logvar[visible_mask]
        sr_supp = mu if deterministic else mu + torch.randn_like(mu) * torch.exp(logvar)
        uncertainty = torch.norm(torch.exp(logvar), dim=1, keepdim=True)
    else:
        sr_supp = gaussians._anchor_feat[visible_mask]
        uncertainty = torch.zeros((anchor.shape[0], 1), device=anchor.device, dtype=anchor.dtype)

    feat = field_feat + sr_supp
    return feat, uncertainty

def evaluate(scene, gaussians, liif_field, pipe, background, out_dir: Path, max_views=8, liif_k_render: int = 1, base_anchor_count: int = None):
    ensure_dir(out_dir)
    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    lpips_fn = build_lpips() if build_lpips is not None else None
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}
    for idx, cam in enumerate(cameras):
        if idx >= max_views:
            break
        with torch.no_grad():
            visible = prefilter_voxel(cam, gaussians, pipe, background)
            feat_override, unc = build_stage2_feat_for_visible(gaussians, liif_field, visible, k_override=liif_k_render, base_anchor_count=base_anchor_count, deterministic=True)
            pred = torch.clamp(
                render_with_feat_override(cam, gaussians, feat_override, unc, pipe, background, visible_mask=visible)["render"],
                0.0,
                1.0,
            )
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            totals["psnr"] += psnr(pred, gt).mean().item()
            totals["ssim"] += ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            if lpips_fn is not None:
                totals["lpips"] += lpips_fn(pred.unsqueeze(0), gt.unsqueeze(0)).mean().item()
            totals["count"] += 1
    for k in ("psnr", "ssim", "lpips"):
        totals[k] /= max(totals["count"], 1)
    return totals


def build_renderable_stage2_state(gaussians: GaussianModel, liif_field: LIIFAnchorField):
    state = {
        "anchor": gaussians._anchor.detach().cpu(),
        "offset": gaussians._offset.detach().cpu(),
        "anchor_feat": gaussians._anchor_feat.detach().cpu(),
        "scaling": gaussians._scaling.detach().cpu(),
        "rotation": gaussians._rotation.detach().cpu(),
        "opacity": gaussians._opacity.detach().cpu(),
        "sr_mu": gaussians._anchor_feat_sr_mu.detach().cpu(),
        "sr_logvar": gaussians._anchor_feat_sr_logvar.detach().cpu(),
        "liif_field": liif_field.state_dict(),
        "mlp_color": gaussians.mlp_color.state_dict(),
    }
    if gaussians.use_feat_bank:
        state["mlp_feature_bank"] = gaussians.mlp_feature_bank.state_dict()
    return state


def _replace_param(module, name, tensor):
    setattr(module, name, torch.nn.Parameter(tensor.cuda().contiguous().requires_grad_(True)))


def restore_renderable_state(gaussians: GaussianModel, liif_field: LIIFAnchorField, state: dict):
    _replace_param(gaussians, "_anchor", state["anchor"].float())
    _replace_param(gaussians, "_offset", state["offset"].float())
    _replace_param(gaussians, "_anchor_feat", state["anchor_feat"].float())
    _replace_param(gaussians, "_scaling", state["scaling"].float())
    _replace_param(gaussians, "_rotation", state["rotation"].float())
    _replace_param(gaussians, "_opacity", state["opacity"].float())
    _replace_param(gaussians, "_anchor_feat_sr_mu", state["sr_mu"].float())
    _replace_param(gaussians, "_anchor_feat_sr_logvar", state["sr_logvar"].float())
    liif_field.load_state_dict(state["liif_field"])
    gaussians.mlp_color.load_state_dict(state["mlp_color"])
    if gaussians.use_feat_bank and "mlp_feature_bank" in state:
        gaussians.mlp_feature_bank.load_state_dict(state["mlp_feature_bank"])


def find_latest_stage2_checkpoint(stage2_output: Path):
    ckpts = []
    for p in stage2_output.glob("stage2_*.pth"):
        m = re.fullmatch(r"stage2_(\d+)\.pth", p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def main():
    parser = ArgumentParser(description="Stage-2 SuperGS with LIIF-style continuous field")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--stage2_output", type=str, required=True)
    parser.add_argument("--stage1_iteration", type=int, default=30000)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--stage2_feature_lr", type=float, default=1e-3)
    parser.add_argument("--decoder_lr", type=float, default=3e-5)
    parser.add_argument("--liif_lr", type=float, default=1e-3)
    parser.add_argument("--liif_hidden_dim", type=int, default=64)
    parser.add_argument("--liif_k", type=int, default=8)
    parser.add_argument("--liif_k_render", type=int, default=1)
    parser.add_argument("--liif_k_growth", type=int, default=0)
    parser.add_argument("--liif_temperature", type=float, default=0.05)
    parser.add_argument("--liif_knn_chunk_size", type=int, default=1024)
    parser.add_argument("--liif_init_checkpoint", type=str, default="")
    parser.add_argument("--depth_dir_name", type=str, default="depth")
    parser.add_argument("--vote_voxel_size", type=float, default=0.03)
    parser.add_argument("--max_candidates_per_view", type=int, default=3000)
    parser.add_argument("--error_threshold", type=float, default=0.08)
    parser.add_argument("--vote_threshold", type=int, default=2)
    parser.add_argument("--max_new_anchors", type=int, default=3000)
    parser.add_argument("--growth_interval", type=int, default=100)
    parser.add_argument("--densify_until", type=int, default=3000)
    parser.add_argument("--refine_iters", type=int, default=1000)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.02)
    parser.add_argument("--refine_max_new", type=int, default=1024)
    parser.add_argument("--lambda_ssim", type=float, default=0.2)
    parser.add_argument("--lambda_vol", type=float, default=0.01)
    parser.add_argument("--lambda_lpips", type=float, default=0.05)
    parser.add_argument("--eval_source", type=str, default="")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--freeze_color_head", action="store_true")
    args = get_combined_args(parser)

    stage2_output = Path(args.stage2_output).resolve()
    ensure_dir(stage2_output)
    (stage2_output / "stage2_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    dataset = model.extract(args)
    dataset.eval = True
    dataset.resolution = 1
    pipe = pipeline.extract(args)
    depth_dir = Path(dataset.source_path) / args.depth_dir_name

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
        dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
        dataset.add_cov_dist, dataset.add_color_dist,
    )
    scene = Scene(dataset, gaussians, load_iteration=args.stage1_iteration, shuffle=False)
    base_anchor_count = int(gaussians.get_anchor.shape[0])
    eval_scene = scene
    if args.eval_source:
        eval_dataset = model.extract(args)
        eval_dataset.eval = True
        eval_dataset.resolution = 1
        eval_dataset.source_path = args.eval_source
        eval_gaussians = GaussianModel(
            eval_dataset.feat_dim, eval_dataset.n_offsets, eval_dataset.voxel_size, eval_dataset.update_depth,
            eval_dataset.update_init_factor, eval_dataset.update_hierachy_factor, eval_dataset.use_feat_bank,
            eval_dataset.appearance_dim, eval_dataset.ratio, eval_dataset.add_opacity_dist,
            eval_dataset.add_cov_dist, eval_dataset.add_color_dist,
        )
        eval_scene = Scene(eval_dataset, eval_gaussians, load_iteration=args.stage1_iteration, shuffle=False)
    gaussians.train()
    gaussians.init_stage2_features()
    gaussians.activate_stage2_mode(True)
    freeze_for_stage2(gaussians, args.freeze_color_head)

    liif_field = LIIFAnchorField(
        feat_dim=gaussians.feat_dim,
        hidden_dim=args.liif_hidden_dim,
        k_neighbors=args.liif_k,
        temperature=args.liif_temperature,
        knn_chunk_size=args.liif_knn_chunk_size,
    ).cuda()
    if args.liif_init_checkpoint:
        init_ckpt = torch.load(args.liif_init_checkpoint, map_location="cpu")
        init_state = init_ckpt.get("state_dict", init_ckpt)
        missing, unexpected = liif_field.load_state_dict(init_state, strict=False)
        print(f"Loaded LIIF init from {args.liif_init_checkpoint}")
        print(f"LIIF init missing={missing}")
        print(f"LIIF init unexpected={unexpected}")

    optimizer = build_stage2_optimizer(gaussians, liif_field, args.stage2_feature_lr, args.decoder_lr, args.liif_lr)
    start_iteration = 1
    resume_path = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
    elif args.resume_latest:
        resume_path = find_latest_stage2_checkpoint(stage2_output)
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        restore_renderable_state(gaussians, liif_field, ckpt["renderable_state"])
        freeze_for_stage2(gaussians, args.freeze_color_head)
        optimizer = build_stage2_optimizer(gaussians, liif_field, args.stage2_feature_lr, args.decoder_lr, args.liif_lr)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = int(ckpt.get("iteration", 0)) + 1
    lpips_fn = build_lpips() if build_lpips is not None else None
    train_cameras = scene.getTrainCameras()

    progress = tqdm(range(start_iteration, args.iterations + 1), desc="SuperGS LIIF stage2")
    for iteration in progress:
        if iteration <= args.densify_until and iteration % args.growth_interval == 1:
            growth_k = args.liif_k if args.liif_k_growth <= 0 else args.liif_k_growth
            new_xyz = build_voted_anchor_positions(
                scene, gaussians, liif_field, pipe, background, depth_dir, args.vote_voxel_size,
                args.max_candidates_per_view, args.error_threshold, args.vote_threshold, args.max_new_anchors,
                liif_k_growth=growth_k,
                base_anchor_count=base_anchor_count
            )
            if new_xyz.shape[0] > 0:
                gaussians.append_stage2_anchors(new_xyz)
                freeze_for_stage2(gaussians, args.freeze_color_head)
                optimizer = build_stage2_optimizer(gaussians, liif_field, args.stage2_feature_lr, args.decoder_lr, args.liif_lr)

        if iteration > args.densify_until and iteration <= args.densify_until + args.refine_iters and iteration % args.growth_interval == 0:
            added = gaussians.refine_stage2_uncertain_anchors(args.uncertainty_threshold, args.refine_max_new)
            if added > 0:
                freeze_for_stage2(gaussians, args.freeze_color_head)
                optimizer = build_stage2_optimizer(gaussians, liif_field, args.stage2_feature_lr, args.decoder_lr, args.liif_lr)

        cam = train_cameras[randint(0, len(train_cameras) - 1)]
        visible = prefilter_voxel(cam, gaussians, pipe, background)
        feat_override, uncertainty = build_stage2_feat_for_visible(gaussians, liif_field, visible, k_override=args.liif_k_render, base_anchor_count=base_anchor_count, deterministic=False)
        pkg = render_with_feat_override(cam, gaussians, feat_override, uncertainty, pipe, background, visible_mask=visible)
        pred = torch.clamp(pkg["render"], 0.0, 1.0)
        gt_hr = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
        gt_lr = F.interpolate(gt_hr.unsqueeze(0), scale_factor=0.5, mode="bilinear", align_corners=False).squeeze(0)
        uncertainty_map = pkg.get("uncertainty", torch.zeros_like(pred[:1]))

        l_rec = uncertainty_guided_loss(pred, gt_hr, uncertainty_map)
        l_ssim = 1.0 - ssim(pred.unsqueeze(0), gt_hr.unsqueeze(0))
        l_vol = pkg["scaling"].prod(dim=1).mean() if "scaling" in pkg else torch.tensor(0.0, device=pred.device)
        pred_lr = F.interpolate(pred.unsqueeze(0), scale_factor=0.5, mode="bilinear", align_corners=False)
        if lpips_fn is not None:
            l_lpips = lpips_fn(pred_lr, gt_lr.unsqueeze(0)).mean()
        else:
            l_lpips = torch.tensor(0.0, device=pred.device)
        loss = l_rec + args.lambda_ssim * l_ssim + args.lambda_vol * l_vol + args.lambda_lpips * l_lpips

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix({
            "loss": f"{loss.item():.5f}",
            "rec": f"{l_rec.item():.5f}",
            "ssim": f"{l_ssim.item():.5f}",
            "anchors": int(gaussians.get_anchor.shape[0]),
        })

        if iteration % args.save_interval == 0 or iteration == args.iterations:
            torch.save(
                {
                    "iteration": iteration,
                    "anchor_count": int(gaussians.get_anchor.shape[0]),
                    "renderable_state": build_renderable_stage2_state(gaussians, liif_field),
                    "optimizer": optimizer.state_dict(),
                },
                stage2_output / f"stage2_{iteration}.pth",
            )

        if iteration % args.eval_interval == 0 or iteration == args.iterations:
            metrics = evaluate(eval_scene, gaussians, liif_field, pipe, background, stage2_output / f"eval_{iteration}", max_views=8, liif_k_render=args.liif_k_render, base_anchor_count=base_anchor_count)
            with open(stage2_output / "metrics_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"iteration": iteration, **metrics, "anchor_count": int(gaussians.get_anchor.shape[0])}) + "\n")

    (stage2_output / "summary.json").write_text(json.dumps({
        "mode": "supergs_liif_stage2",
        "stage1_model_path": dataset.model_path,
        "stage1_iteration": args.stage1_iteration,
        "final_anchor_count": int(gaussians.get_anchor.shape[0]),
        "base_anchor_count": base_anchor_count,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()









