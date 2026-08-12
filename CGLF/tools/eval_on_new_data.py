#!/usr/bin/env python
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, OptimizationParams
from train import evaluate, get_logger, render_sets


def main():
    parser = ArgumentParser(description="Render and evaluate a trained Stage-1 model on a new dataset root.")
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--copy_train_to_test_if_missing", action="store_true")
    args = parser.parse_args()

    parser2 = ArgumentParser()
    lp = ModelParams(parser2)
    _op = OptimizationParams(parser2)
    pp = PipelineParams(parser2)
    defaults = parser2.parse_args([])
    defaults.source_path = args.source_path
    defaults.model_path = args.model_path

    os.makedirs(args.model_path, exist_ok=True)
    logger = get_logger(args.model_path)

    dataset = lp.extract(defaults)
    pipeline = pp.extract(defaults)
    visible_count = render_sets(
        dataset,
        -1,
        pipeline,
        skip_train=False,
        skip_test=False,
        wandb=None,
        tb_writer=None,
        dataset_name=None,
        logger=logger,
    )

    if args.copy_train_to_test_if_missing:
        import shutil

        model_dir = Path(args.model_path)
        test_dir = model_dir / "test"
        train_dir = model_dir / "train"
        if not test_dir.exists() or not any(test_dir.iterdir()):
            if train_dir.exists():
                if test_dir.exists():
                    shutil.rmtree(test_dir)
                shutil.copytree(train_dir, test_dir)
        elif train_dir.exists():
            for method in train_dir.iterdir():
                src = train_dir / method.name
                dst = test_dir / method.name
                src_renders = src / "renders"
                dst_renders = dst / "renders"
                src_has = src_renders.exists() and any(src_renders.iterdir())
                dst_has = dst_renders.exists() and any(dst_renders.iterdir()) if dst_renders.exists() else False
                if src_has and not dst_has:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)

    evaluate(args.model_path, visible_count=visible_count, wandb=None, tb_writer=None, dataset_name=None, logger=logger)
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
