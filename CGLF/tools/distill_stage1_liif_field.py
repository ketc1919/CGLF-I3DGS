import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.liif_anchor_field import LIIFAnchorField


def build_query_cell(anchor_xyz: torch.Tensor, voxel_size: float):
    return torch.full_like(anchor_xyz, float(voxel_size))


def main():
    parser = ArgumentParser(description="Distill a stage-1 LIIF continuous field from frozen anchor features")
    model = ModelParams(parser)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stage1_iteration", type=int, default=30000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--liif_hidden_dim", type=int, default=64)
    parser.add_argument("--liif_k", type=int, default=8)
    parser.add_argument("--liif_temperature", type=float, default=0.05)
    parser.add_argument("--liif_knn_chunk_size", type=int, default=2048)
    parser.add_argument("--save_every", type=int, default=500)
    args = get_combined_args(parser)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "distill_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    dataset = model.extract(args)
    dataset.eval = True
    dataset.resolution = 1

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
    _scene = Scene(dataset, gaussians, load_iteration=args.stage1_iteration, shuffle=False)
    gaussians.eval()

    anchor_xyz = gaussians.get_anchor.detach()
    anchor_feat = gaussians._anchor_feat.detach()
    query_cell_full = build_query_cell(anchor_xyz, gaussians.voxel_size)
    num_anchors = anchor_xyz.shape[0]

    liif_field = LIIFAnchorField(
        feat_dim=gaussians.feat_dim,
        hidden_dim=args.liif_hidden_dim,
        k_neighbors=args.liif_k,
        temperature=args.liif_temperature,
        knn_chunk_size=args.liif_knn_chunk_size,
    ).cuda()
    optimizer = torch.optim.Adam(liif_field.parameters(), lr=args.lr)

    best_loss = float("inf")
    progress = tqdm(range(1, args.steps + 1), desc="Distill stage1 LIIF field")
    for step in progress:
        if args.batch_size >= num_anchors:
            batch_idx = torch.arange(num_anchors, device=anchor_xyz.device)
        else:
            batch_idx = torch.randint(0, num_anchors, (args.batch_size,), device=anchor_xyz.device)

        q_xyz = anchor_xyz[batch_idx]
        q_cell = query_cell_full[batch_idx]
        target = anchor_feat[batch_idx]

        pred = liif_field(q_xyz, q_cell, anchor_xyz, anchor_feat)
        loss_l1 = F.l1_loss(pred, target)
        loss_mse = F.mse_loss(pred, target)
        loss = loss_l1 + 0.1 * loss_mse

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=float(loss.item()), l1=float(loss_l1.item()), mse=float(loss_mse.item()))

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            torch.save(
                {
                    "state_dict": liif_field.state_dict(),
                    "best_loss": best_loss,
                    "step": step,
                    "meta": {
                        "feat_dim": gaussians.feat_dim,
                        "hidden_dim": args.liif_hidden_dim,
                        "k_neighbors": args.liif_k,
                        "temperature": args.liif_temperature,
                        "knn_chunk_size": args.liif_knn_chunk_size,
                        "voxel_size": float(gaussians.voxel_size),
                        "num_anchors": int(num_anchors),
                        "stage1_model_path": dataset.model_path,
                        "stage1_iteration": int(args.stage1_iteration),
                    },
                },
                output / "liif_field_init_best.pth",
            )

        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "state_dict": liif_field.state_dict(),
                    "best_loss": best_loss,
                    "step": step,
                    "meta": {
                        "feat_dim": gaussians.feat_dim,
                        "hidden_dim": args.liif_hidden_dim,
                        "k_neighbors": args.liif_k,
                        "temperature": args.liif_temperature,
                        "knn_chunk_size": args.liif_knn_chunk_size,
                        "voxel_size": float(gaussians.voxel_size),
                        "num_anchors": int(num_anchors),
                        "stage1_model_path": dataset.model_path,
                        "stage1_iteration": int(args.stage1_iteration),
                    },
                },
                output / f"liif_field_init_step{step}.pth",
            )

    (output / "summary.json").write_text(
        json.dumps(
            {
                "best_loss": best_loss,
                "num_anchors": int(num_anchors),
                "feat_dim": gaussians.feat_dim,
                "voxel_size": float(gaussians.voxel_size),
                "steps": int(args.steps),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
