import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import prefilter_voxel, render
from scene import GaussianModel, Scene


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def assign_tensor(module, name: str, value: torch.Tensor):
    setattr(module, name, torch.nn.Parameter(value.cuda().requires_grad_(False)))


def restore_renderable_state_scaffold(gaussians: GaussianModel, renderable_state: dict):
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


def main():
    parser = ArgumentParser(description="Render LIIF stage2 checkpoint with pure ScaffoldGS base render path")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--stage1_iteration", type=int, default=30000)
    parser.add_argument("--out_method_name", type=str, required=True)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--warmup_views", type=int, default=5)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    dataset.eval = True
    dataset.resolution = 1
    pipe = pipeline.extract(args)

    stage2_ckpt_path = Path(args.stage2_checkpoint).resolve()
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu")
    renderable_state = ckpt["renderable_state"]

    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
        dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
        dataset.add_cov_dist, dataset.add_color_dist
    )
    scene = Scene(dataset, gaussians, load_iteration=args.stage1_iteration, shuffle=False)
    restore_renderable_state_scaffold(gaussians, renderable_state)
    gaussians.eval()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    method_dir = Path(dataset.model_path) / "test" / args.out_method_name
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    ensure_dir(render_dir)
    ensure_dir(gt_dir)

    cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()
    if args.max_views and args.max_views > 0:
        cameras = cameras[: args.max_views]

    times = []
    warmup = min(args.warmup_views, len(cameras))

    for idx, cam in enumerate(tqdm(cameras, desc="Render scaffold-base")):
        with torch.no_grad():
            visible = prefilter_voxel(cam, gaussians, pipe, background)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = torch.clamp(render(cam, gaussians, pipe, background, visible_mask=visible)["render"], 0.0, 1.0)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)

        if idx >= warmup:
            times.append(dt)

        name = f"{idx:05d}.png"
        torchvision.utils.save_image(pred, render_dir / name)
        torchvision.utils.save_image(gt, gt_dir / name)

    fps = 0.0 if not times else (len(times) / sum(times))
    stats = {
        "count_total": len(cameras),
        "warmup_views": warmup,
        "count_measured": len(times),
        "avg_sec_per_view": (sum(times) / len(times)) if times else None,
        "fps": fps,
    }
    (method_dir / "scaffold_base_fps.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
