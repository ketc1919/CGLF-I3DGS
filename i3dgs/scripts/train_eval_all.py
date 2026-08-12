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

import argparse
import subprocess
import os
import sys
import lpips
import numpy as np
import torch
import cv2
from tqdm import tqdm
import json
from fused_ssim import fused_ssim

sys.path.append(".")
from utils import get_image_names, parse_time, psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run reconstruction and compute metrics for all scenes"
    )
    parser.add_argument("--base_dir", default="data")
    parser.add_argument("--base_out_dir", default="results")
    parser.add_argument('-i', '--images_dir', type=str, default="images")
    parser.add_argument("--downsampling", default=1.0, type=float,
        help="Downsampling for reconstruction. Use the original size when recomputing metrics",
    )
    parser.add_argument("--skip_reconstruction", action="store_true")
    parser.add_argument("--skip_gs", action="store_true",
        help="Pass --skip_gs to training and report pose metrics (R°, t) instead of quality metrics")
    parser.add_argument("--skip_rerun_metrics", action="store_true")
    parser.add_argument("--save_at_finetune_epoch", type=str, nargs="+", default=[])
    parser.add_argument("--extra_args", type=str, default="")
    parser.add_argument("--scenes", type=str, default="",
        help="Comma-separated list of scenes to run. If empty, runs all scenes.")
    args = parser.parse_args()

    # Set the test parameters for each scene
    all_test_params_full = {
        "MipNeRF360/garden": {"test_hold": 8},
        "MipNeRF360/bonsai": {"test_hold": 8},
        "MipNeRF360/counter": {"test_hold": 8},
        "MipNeRF360U/bicycle": {"test_hold": 8},
        "MipNeRF360U/kitchen": {"test_hold": 8},
        "MipNeRF360U/room": {"test_hold": 8},
        "MipNeRF360U/stump": {"test_hold": 8},
        "tandt/train": {"test_hold": 8},
        "tandt/truck": {"test_hold": 8},
        "db/drjohnson": {"test_hold": 8},
        "db/playroom": {"test_hold": 8},
        "TUM/rgbd_dataset_freiburg1_desk": {"test_hold": 30},
        "TUM/rgbd_dataset_freiburg2_xyz": {"test_hold": 30},
        "TUM/rgbd_dataset_freiburg3_long_office_household": {"test_hold": 30},
        "StaticHikes/forest1": {"test_hold": 10},
        "StaticHikes/forest2": {"test_hold": 10},
        "StaticHikes/university2": {"test_hold": 10},
    }

    # Filter scenes if specified
    if args.scenes:
        selected_scenes = [s.strip() for s in args.scenes.split(",")]
        all_test_params = {k: v for k, v in all_test_params_full.items() if k in selected_scenes}
    else:
        all_test_params = all_test_params_full

    common_extra_args = [
        "--downsampling",
        str(args.downsampling),
        "--images_dir",
        args.images_dir,
    ]
    if args.skip_gs:
        common_extra_args += ["--skip_gs"]
    if args.extra_args != "":
        common_extra_args += args.extra_args.split(" ")
    if len(args.save_at_finetune_epoch) > 0:
        common_extra_args += ["--save_at_finetune_epoch"] + args.save_at_finetune_epoch

    if not args.skip_reconstruction:
        print("Reconstructing all scenes")
        for index, (scene, test_params) in enumerate(all_test_params.items()):
            print(f"Optimizing scene {index + 1}/{len(all_test_params)}: {scene}")
            subprocess.run(
                args=[
                    "python",
                    "train.py",
                    "-s",
                    os.path.join(args.base_dir, scene),
                    "--test_hold",
                    str(test_params["test_hold"]),
                    "--model_path",
                    os.path.join(args.base_out_dir, scene),
                ]
                + common_extra_args
            )

    if args.skip_gs:
        print("Reading pose metrics from metadata")
        R_sum, t_sum, time_sum, count = 0, 0, 0, 0
        for scene in all_test_params:
            model_path = os.path.join(args.base_out_dir, scene)
            with open(f"{model_path}/metadata.json", "r") as f:
                json_dict = json.load(f)
            if "R\u00b0" in json_dict and "t" in json_dict:
                R_sum += json_dict["R\u00b0"]
                t = json_dict["t"] / (10 if "TUM" in scene else 1)
                t_sum += t
                time_sum += json_dict["time"]
                count += 1
                print(f"  {scene}: R°={json_dict['R\u00b0']:.4f}, t={t:.4f}, time={parse_time(json_dict['time'])}")
            else:
                print(f"  {scene}: pose metrics not found in metadata")
        if count > 0:
            print(f"Average over {count} scenes: R°={R_sum/count:.4f}, t={t_sum/count:.4f}, time={parse_time(time_sum/count)}")
        with open(f"{args.base_out_dir}/metrics.csv", "w") as f:
            f.write(f"R°,t,time\n{R_sum/count:.4f},{t_sum/count:.4f},{parse_time(time_sum/count)}\n")
    else:
        print("Computing metrics")
        lpips = lpips.LPIPS(net="vgg").cuda()
        metrics_list = []
        finetuning_epochs = [""] + args.save_at_finetune_epoch
        for epoch_id, finetuning_epoch in enumerate(finetuning_epochs):
            print(
                f"Computing metrics for finetuning epoch {epoch_id + 1}/{len(finetuning_epochs)}: {finetuning_epoch}"
            )
            for index, (scene, test_params) in enumerate(all_test_params.items()):
                print(
                    f"Computing metrics for scene {index + 1}/{len(all_test_params)}: {scene}"
                )
                model_path = os.path.join(args.base_out_dir, scene, finetuning_epoch)
                with open(f"{model_path}/metadata.json", "r") as f:
                    json_dict = json.load(f)
                recons_time = json_dict["time"]

                if args.skip_rerun_metrics:
                    PSNR = json_dict["PSNR"]
                    SSIM = json_dict["SSIM"]
                    LPIPS = json_dict["LPIPS"]
                else:
                    gt_dir = os.path.join(args.base_dir, scene, args.images_dir)
                    render_dir = os.path.join(model_path, "test_images")

                    image_names = get_image_names(gt_dir)
                    image_names.sort()
                    image_names = image_names[:: test_params["test_hold"]]

                    PSNR, SSIM, LPIPS = 0, 0, 0
                    missing_renders = []

                    for image_name in tqdm(image_names):
                        gt_image = cv2.cvtColor(
                            cv2.imread(f"{gt_dir}/{image_name}"), cv2.COLOR_BGR2RGB
                        )
                        raw_image = cv2.imread(f"{render_dir}/{image_name}")
                        if raw_image is None:
                            # Image failed to register: count it as a black image
                            missing_renders.append(image_name)
                            image = np.zeros_like(gt_image)
                        else:
                            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                        if image.shape != gt_image.shape:
                            image = cv2.resize(
                                image,
                                (gt_image.shape[1], gt_image.shape[0]),
                                interpolation=cv2.INTER_AREA,
                            )
                        image = (
                            torch.from_numpy(image).permute(2, 0, 1).cuda().float() / 255
                        )
                        gt_image = (
                            torch.from_numpy(gt_image).permute(2, 0, 1).cuda().float() / 255
                        )

                        mask = (
                            gt_image.sum(0) > 0
                            if "TUM" in scene
                            else torch.ones_like(gt_image[0].bool())
                        )
                        mask = mask.expand_as(image)
                        image *= mask

                        PSNR += psnr(image[mask], gt_image[mask])
                        SSIM += fused_ssim(image[None], gt_image[None], train=False).item()
                        LPIPS += lpips(image, gt_image).item()

                    if missing_renders:
                        print(
                            f"  Warning: {len(missing_renders)}/{len(image_names)} test renders "
                            f"missing in {render_dir} (scored as black):"
                        )
                        print(f"  {missing_renders}")

                    PSNR, SSIM, LPIPS = (
                        PSNR / len(image_names),
                        SSIM / len(image_names),
                        LPIPS / len(image_names),
                    )

                metrics_list += [
                    f"{PSNR:.4f},{SSIM:.4f},{LPIPS:.4f},{parse_time(recons_time)}",
                    ", ",
                ]
            metrics_list[-1] = "\n"  # Swap last comma for newline
        metrics_list = metrics_list[:-1]  # remove last comma
        print("Displaying PSNR, SSIM, LPIPS, time for each scene")
        print("Each line corresponds to a number of epochs (starting from 0)")
        print("".join(metrics_list))

        with open(f"{args.base_out_dir}/metrics.csv", "w") as f:
            f.write("".join(metrics_list))
