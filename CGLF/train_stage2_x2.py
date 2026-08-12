import json
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import prefilter_voxel, render_with_dense_scaffold
from scene import GaussianModel, Scene
from scene.continuous_scaffold_field import ContinuousScaffoldField
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


def freeze_stage1(gaussians: GaussianModel):
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
            p.requires_grad_(False)
    if gaussians.appearance_dim > 0 and gaussians.embedding_appearance is not None:
        for p in gaussians.embedding_appearance.parameters():
            p.requires_grad_(False)


def build_query_template(device, dtype):
    # 8 octant samples for x2-style densification
    vals = [-0.25, 0.25]
    template = []
    for x in vals:
        for y in vals:
            for z in vals:
                template.append([x, y, z])
    return torch.tensor(template, device=device, dtype=dtype)


def build_dense_scaffold(
    viewpoint_cam,
    gaussians: GaussianModel,
    field: ContinuousScaffoldField,
    visible_mask: torch.Tensor,
    max_parent_anchors: int = 12000,
    conf_threshold: float = 0.25,
):
    base_anchor_all = gaussians.get_anchor.detach()
    base_feat_all = gaussians._anchor_feat.detach()
    parent_anchor = base_anchor_all[visible_mask]
    parent_scaling = gaussians.get_scaling.detach()[visible_mask]
    parent_offsets = gaussians._offset.detach()[visible_mask]

    if parent_anchor.shape[0] == 0:
        raise RuntimeError("No visible anchors for dense scaffold generation.")

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

    dense_feat, conf, _, _ = field(
        query_xyz,
        query_cell,
        query_viewdir,
        base_anchor_all,
        base_feat_all,
    )

    conf = conf.view(-1)
    keep = conf >= conf_threshold
    if keep.sum() == 0:
        topk = min(1024, conf.shape[0])
        keep_idx = torch.topk(conf, k=topk, largest=True).indices
        keep = torch.zeros_like(conf, dtype=torch.bool)
        keep[keep_idx] = True

    parent_scaling_rep = parent_scaling[:, None, :].repeat(1, q, 1).reshape(-1, parent_scaling.shape[-1])
    parent_offsets_rep = parent_offsets[:, None, :, :].repeat(1, q, 1, 1).reshape(-1, parent_offsets.shape[1], 3)

    return (
        query_xyz[keep],
        dense_feat[keep],
        parent_scaling_rep[keep],
        parent_offsets_rep[keep],
        conf[keep],
    )


def render_stage2_image(viewpoint_cam, gaussians, field, pipe, background, max_parent_anchors, conf_threshold):
    visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
    dense_anchor_xyz, dense_anchor_feat, dense_scaling, dense_offsets, dense_conf = build_dense_scaffold(
        viewpoint_cam,
        gaussians,
        field,
        visible_mask=visible_mask,
        max_parent_anchors=max_parent_anchors,
        conf_threshold=conf_threshold,
    )
    render_pkg = render_with_dense_scaffold(
        viewpoint_cam,
        gaussians,
        dense_anchor_xyz,
        dense_anchor_feat,
        dense_scaling,
        dense_offsets,
        pipe,
        background,
        retain_grad=True,
    )
    render_pkg["dense_confidence"] = dense_conf
    render_pkg["dense_anchor_count"] = int(dense_anchor_xyz.shape[0])
    return render_pkg


def evaluate(field, scene, gaussians, pipe, background, out_dir: Path, max_views=8, max_parent_anchors=12000, conf_threshold=0.25):
    ensure_dir(out_dir)
    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    lpips_fn = build_lpips() if build_lpips is not None else None
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}

    for idx, cam in enumerate(cameras):
        if idx >= max_views:
            break
        with torch.no_grad():
            pkg = render_stage2_image(cam, gaussians, field, pipe, background, max_parent_anchors, conf_threshold)
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
    parser = ArgumentParser(description="Stage-2 x2 training with dense scaffold rendering")
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
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--max_parent_anchors", type=int, default=12000)
    parser.add_argument("--confidence_reg_weight", type=float, default=0.01)
    args = get_combined_args(parser)

    stage2_output = Path(args.stage2_output).resolve()
    ensure_dir(stage2_output)
    with open(stage2_output / "stage2_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset = model.extract(args)
    dataset.eval = True
    dataset.resolution = 1  # high-resolution supervision

    pipe = pipeline.extract(args)

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
    gaussians.eval()
    freeze_stage1(gaussians)

    field = ContinuousScaffoldField(
        feat_dim=dataset.feat_dim,
        hidden_dim=args.hidden_dim,
        k_neighbors=args.k_neighbors,
        temperature=args.temperature,
    ).cuda()
    optimizer = torch.optim.Adam(field.parameters(), lr=args.lr)

    train_cameras = scene.getTrainCameras()
    progress = tqdm(range(1, args.iterations + 1), desc="Stage2 dense x2")
    for iteration in progress:
        cam = train_cameras[randint(0, len(train_cameras) - 1)]
        pkg = render_stage2_image(
            cam,
            gaussians,
            field,
            pipe,
            background,
            max_parent_anchors=args.max_parent_anchors,
            conf_threshold=args.conf_threshold,
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
                pipe,
                background,
                out_dir=stage2_output / f"eval_{iteration}",
                max_views=8,
                max_parent_anchors=args.max_parent_anchors,
                conf_threshold=args.conf_threshold,
            )
            if metrics is not None:
                with open(stage2_output / "metrics_history.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iteration": iteration, **metrics}) + "\n")
            field.train()

    summary = {
        "stage1_model_path": dataset.model_path,
        "stage1_iteration": args.stage1_iteration,
        "stage2_output": str(stage2_output),
        "mode": "dense_scaffold_rendering",
    }
    (stage2_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
