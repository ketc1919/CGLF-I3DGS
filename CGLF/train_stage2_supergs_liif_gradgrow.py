import json
import math
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


def evaluate(scene, gaussians, liif_field, pipe, background, out_dir: Path, max_views=8, liif_k_render: int = 1):
    ensure_dir(out_dir)
    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    lpips_fn = build_lpips() if build_lpips is not None else None
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}
    for idx, cam in enumerate(cameras):
        if idx >= max_views:
            break
        with torch.no_grad():
            visible = prefilter_voxel(cam, gaussians, pipe, background)
            feat_override, unc = build_stage2_feat_for_visible(gaussians, liif_field, visible, k_override=liif_k_render, deterministic=True)
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
        stem = p.stem.split("_")
        if len(stem) == 2 and stem[1].isdigit():
            ckpts.append((int(stem[1]), p))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def reset_growth_stats(gaussians: GaussianModel):
    gaussians.opacity_accum = torch.zeros((gaussians.get_anchor.shape[0], 1), device="cuda")
    gaussians.offset_gradient_accum = torch.zeros((gaussians.get_anchor.shape[0] * gaussians.n_offsets, 1), device="cuda")
    gaussians.offset_denom = torch.zeros((gaussians.get_anchor.shape[0] * gaussians.n_offsets, 1), device="cuda")
    gaussians.anchor_demon = torch.zeros((gaussians.get_anchor.shape[0], 1), device="cuda")


def build_gradient_anchor_positions(
    gaussians: GaussianModel,
    check_interval: int,
    success_threshold: float,
    grad_threshold: float,
    max_new_anchors: int,
):
    grads = gaussians.offset_gradient_accum / gaussians.offset_denom.clamp_min(1.0)
    grads[grads.isnan()] = 0.0
    grads_norm = torch.norm(grads, dim=-1)
    offset_mask = (gaussians.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

    init_length = gaussians.get_anchor.shape[0] * gaussians.n_offsets
    candidates = []
    for level in range(gaussians.update_depth):
        cur_threshold = grad_threshold * ((gaussians.update_hierachy_factor // 2) ** level)
        candidate_mask = torch.logical_and(grads_norm >= cur_threshold, offset_mask)
        if not candidate_mask.any():
            continue

        rand_mask = (torch.rand_like(candidate_mask.float()) > (0.5 ** (level + 1))).cuda()
        candidate_mask = torch.logical_and(candidate_mask, rand_mask)
        if not candidate_mask.any():
            continue

        length_inc = gaussians.get_anchor.shape[0] * gaussians.n_offsets - init_length
        if length_inc > 0:
            candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device="cuda")], dim=0)

        all_xyz = gaussians.get_anchor.unsqueeze(dim=1) + gaussians._offset * gaussians.get_scaling[:, :3].unsqueeze(dim=1)
        size_factor = gaussians.update_init_factor // (gaussians.update_hierachy_factor ** level)
        cur_size = gaussians.voxel_size * size_factor
        grid_coords = torch.round(gaussians.get_anchor / cur_size).int()
        selected_xyz = all_xyz.view(-1, 3)[candidate_mask]
        if selected_xyz.shape[0] == 0:
            continue
        selected_grid_coords = torch.round(selected_xyz / cur_size).int()
        selected_grid_coords_unique = torch.unique(selected_grid_coords, dim=0)

        if selected_grid_coords_unique.shape[0] == 0:
            continue

        chunk_size = 4096
        remove_duplicates_list = []
        max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
        for idx in range(max_iters):
            begin = idx * chunk_size
            end = (idx + 1) * chunk_size
            cur_remove = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[begin:end, :]).all(-1).any(-1).view(-1)
            remove_duplicates_list.append(cur_remove)
        remove_duplicates = torch.zeros((selected_grid_coords_unique.shape[0],), dtype=torch.bool, device="cuda")
        for item in remove_duplicates_list:
            remove_duplicates = torch.logical_or(remove_duplicates, item)
        candidate_anchor = selected_grid_coords_unique[~remove_duplicates] * cur_size
        if candidate_anchor.shape[0] > 0:
            candidates.append(candidate_anchor)

    if not candidates:
        return torch.empty((0, 3), device="cuda")
    new_xyz = torch.unique(torch.cat(candidates, dim=0), dim=0)
    if new_xyz.shape[0] > max_new_anchors:
        perm = torch.randperm(new_xyz.shape[0], device=new_xyz.device)[:max_new_anchors]
        new_xyz = new_xyz[perm]
    return new_xyz


def main():
    parser = ArgumentParser(description="Stage-2 CGLF with LIIF and ScaffoldGS-style gradient growth")
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
    parser.add_argument("--liif_k", type=int, default=4)
    parser.add_argument("--liif_k_render", type=int, default=1)
    parser.add_argument("--liif_temperature", type=float, default=0.05)
    parser.add_argument("--liif_knn_chunk_size", type=int, default=8192)
    parser.add_argument("--liif_init_checkpoint", type=str, default="")
    parser.add_argument("--growth_interval", type=int, default=100)
    parser.add_argument("--densify_until", type=int, default=2000)
    parser.add_argument("--refine_iters", type=int, default=1000)
    parser.add_argument("--success_threshold", type=float, default=0.8)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--max_new_anchors", type=int, default=1000)
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
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
        dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
        dataset.add_cov_dist, dataset.add_color_dist,
    )
    scene = Scene(dataset, gaussians, load_iteration=args.stage1_iteration, shuffle=False)
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
    reset_growth_stats(gaussians)

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
        liif_field.load_state_dict(init_state, strict=False)

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
        reset_growth_stats(gaussians)
        optimizer = build_stage2_optimizer(gaussians, liif_field, args.stage2_feature_lr, args.decoder_lr, args.liif_lr)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = int(ckpt.get("iteration", 0)) + 1

    lpips_fn = build_lpips() if build_lpips is not None else None
    train_cameras = scene.getTrainCameras()
    progress = tqdm(range(start_iteration, args.iterations + 1), desc="CGLF LIIF stage2 grad-grow")
    for iteration in progress:
        cam = train_cameras[randint(0, len(train_cameras) - 1)]
        visible = prefilter_voxel(cam, gaussians, pipe, background)
        feat_override, uncertainty = build_stage2_feat_for_visible(gaussians, liif_field, visible, k_override=args.liif_k_render, deterministic=False)
        pkg = render_with_feat_override(cam, gaussians, feat_override, uncertainty, pipe, background, visible_mask=visible, retain_grad=True)
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

        if iteration <= args.densify_until:
            gaussians.training_statis(
                pkg["viewspace_points"],
                pkg["neural_opacity"],
                pkg["visibility_filter"],
                pkg["selection_mask"],
                visible,
            )

        optimizer.step()

        if iteration <= args.densify_until and iteration % args.growth_interval == 0:
            new_xyz = build_gradient_anchor_positions(
                gaussians,
                check_interval=args.growth_interval,
                success_threshold=args.success_threshold,
                grad_threshold=args.densify_grad_threshold,
                max_new_anchors=args.max_new_anchors,
            )
            reset_growth_stats(gaussians)
            if new_xyz.shape[0] > 0:
                gaussians.append_stage2_anchors(new_xyz)
                freeze_for_stage2(gaussians, args.freeze_color_head)
                reset_growth_stats(gaussians)
                optimizer = build_stage2_optimizer(gaussians, liif_field, args.stage2_feature_lr, args.decoder_lr, args.liif_lr)

        if iteration > args.densify_until and iteration <= args.densify_until + args.refine_iters and iteration % args.growth_interval == 0:
            added = gaussians.refine_stage2_uncertain_anchors(args.uncertainty_threshold, args.refine_max_new)
            if added > 0:
                freeze_for_stage2(gaussians, args.freeze_color_head)
                reset_growth_stats(gaussians)
                optimizer = build_stage2_optimizer(gaussians, liif_field, args.stage2_feature_lr, args.decoder_lr, args.liif_lr)

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
            metrics = evaluate(eval_scene, gaussians, liif_field, pipe, background, stage2_output / f"eval_{iteration}", max_views=8, liif_k_render=args.liif_k_render)
            with open(stage2_output / "metrics_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"iteration": iteration, **metrics, "anchor_count": int(gaussians.get_anchor.shape[0])}) + "\n")

    (stage2_output / "summary.json").write_text(json.dumps({
        "mode": "cglf_liif_stage2_gradgrow",
        "stage1_model_path": dataset.model_path,
        "stage1_iteration": args.stage1_iteration,
        "final_anchor_count": int(gaussians.get_anchor.shape[0]),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
