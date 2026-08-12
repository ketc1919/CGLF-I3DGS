import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import prefilter_voxel, render, render_with_feat_override
from scene import GaussianModel, Scene
from scene.liif_anchor_field import LIIFAnchorField
from utils.image_utils import psnr
from utils.loss_utils import ssim

try:
    import lpips as _lpips_pkg

    def build_lpips():
        return _lpips_pkg.LPIPS(net='vgg').to("cuda")
except ImportError:
    from lpipsPyTorch import LPIPS

    def build_lpips():
        return LPIPS(net_type='vgg').to("cuda")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_stage2_args(stage2_dir: Path):
    args_path = stage2_dir / "stage2_args.json"
    if not args_path.exists():
        return {}
    return json.loads(args_path.read_text(encoding="utf-8"))


def assign_tensor(module, name: str, value: torch.Tensor):
    setattr(module, name, nn.Parameter(value.cuda().requires_grad_(False)))


def restore_renderable_state(gaussians: GaussianModel, renderable_state: dict):
    assign_tensor(gaussians, "_anchor", renderable_state["anchor"].float())
    assign_tensor(gaussians, "_offset", renderable_state["offset"].float())
    assign_tensor(gaussians, "_anchor_feat", renderable_state["anchor_feat"].float())
    assign_tensor(gaussians, "_scaling", renderable_state["scaling"].float())
    assign_tensor(gaussians, "_rotation", renderable_state["rotation"].float())
    assign_tensor(gaussians, "_opacity", renderable_state["opacity"].float())
    gaussians.init_stage2_features()
    assign_tensor(gaussians, "_anchor_feat_sr_mu", renderable_state["sr_mu"].float())
    assign_tensor(gaussians, "_anchor_feat_sr_logvar", renderable_state["sr_logvar"].float())
    gaussians.activate_stage2_mode(True)
    gaussians.max_radii2D = torch.zeros((gaussians.get_anchor.shape[0]), device="cuda")
    if "mlp_color" in renderable_state:
        gaussians.mlp_color.load_state_dict(renderable_state["mlp_color"])
    if gaussians.use_feat_bank and "mlp_feature_bank" in renderable_state:
        gaussians.mlp_feature_bank.load_state_dict(renderable_state["mlp_feature_bank"])


def build_liif_field(gaussians: GaussianModel, stage2_args: dict, renderable_state: dict):
    liif_field = LIIFAnchorField(
        feat_dim=gaussians.feat_dim,
        hidden_dim=stage2_args.get("liif_hidden_dim", 64),
        k_neighbors=stage2_args.get("liif_k", 8),
        temperature=stage2_args.get("liif_temperature", 0.05),
    ).cuda()
    liif_field.load_state_dict(renderable_state["liif_field"])
    liif_field.eval()
    return liif_field


def build_stage2_feat_for_visible_liif(gaussians: GaussianModel, liif_field: LIIFAnchorField, visible_mask: torch.Tensor, k_override: int = None):
    anchor = gaussians.get_anchor[visible_mask]
    context_xyz = gaussians.get_anchor.detach()
    context_feat = gaussians._anchor_feat.detach()
    query_cell = torch.full_like(anchor, float(gaussians.voxel_size))
    field_feat = liif_field(anchor, query_cell, context_xyz, context_feat, k_override=k_override)
    mu = gaussians._anchor_feat_sr_mu[visible_mask]
    logvar = gaussians._anchor_feat_sr_logvar[visible_mask]
    feat = field_feat + mu
    uncertainty = torch.norm(torch.exp(logvar), dim=1, keepdim=True)
    return feat, uncertainty


def compute_metrics(pred, gt, lpips_fn):
    return {
        "psnr": float(psnr(pred, gt).mean().item()),
        "ssim": float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()),
        "lpips": float(lpips_fn(pred.unsqueeze(0), gt.unsqueeze(0)).mean().item()),
    }


def main():
    parser = ArgumentParser(description="Render and recompute metrics for stage2 checkpoints")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_mode", type=str, choices=["strict", "liif"], required=True)
    parser.add_argument("--stage1_iteration", type=int, default=30000)
    parser.add_argument("--out_method_name", type=str, required=True)
    parser.add_argument("--max_views", type=int, default=8)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    dataset.eval = True
    dataset.resolution = 1
    pipe = pipeline.extract(args)

    stage2_ckpt_path = Path(args.stage2_checkpoint).resolve()
    stage2_dir = stage2_ckpt_path.parent
    stage2_args = load_stage2_args(stage2_dir)

    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
        dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
        dataset.add_cov_dist, dataset.add_color_dist
    )
    scene = Scene(dataset, gaussians, load_iteration=args.stage1_iteration, shuffle=False)
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu")
    renderable_state = ckpt["renderable_state"]
    restore_renderable_state(gaussians, renderable_state)
    gaussians.eval()

    liif_field = None
    if args.stage2_mode == "liif":
        liif_field = build_liif_field(gaussians, stage2_args, renderable_state)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    lpips_fn = build_lpips()

    method_dir = Path(dataset.model_path) / "test" / args.out_method_name
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    ensure_dir(render_dir)
    ensure_dir(gt_dir)

    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0, "per_view": {}}
    for idx, cam in enumerate(tqdm(cameras[: args.max_views], desc=f"Render {args.stage2_mode}")):
        with torch.no_grad():
            visible = prefilter_voxel(cam, gaussians, pipe, background)
            if args.stage2_mode == "strict":
                pred = torch.clamp(render(cam, gaussians, pipe, background, visible_mask=visible)["render"], 0.0, 1.0)
            else:
                feat_override, uncertainty = build_stage2_feat_for_visible_liif(gaussians, liif_field, visible, k_override=stage2_args.get("liif_k_render", 1))
                pred = torch.clamp(
                    render_with_feat_override(cam, gaussians, feat_override, uncertainty, pipe, background, visible_mask=visible)["render"],
                    0.0,
                    1.0,
                )
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)

        name = f"{idx:05d}.png"
        torchvision.utils.save_image(pred, render_dir / name)
        torchvision.utils.save_image(gt, gt_dir / name)

        m = compute_metrics(pred, gt, lpips_fn)
        metrics["psnr"] += m["psnr"]
        metrics["ssim"] += m["ssim"]
        metrics["lpips"] += m["lpips"]
        metrics["count"] += 1
        metrics["per_view"][name] = m

    for k in ("psnr", "ssim", "lpips"):
        metrics[k] /= max(metrics["count"], 1)

    (method_dir / "recomputed_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("psnr", "ssim", "lpips", "count")}, indent=2))


if __name__ == "__main__":
    main()
