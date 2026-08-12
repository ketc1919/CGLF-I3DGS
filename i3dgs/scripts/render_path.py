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
import sys
import os

sys.path.append(".")
from dataloaders.read_write_model import read_model, qvec2rotmat
from scene.scene_model import SceneModel
from utils import focal2fov, get_transform_mean_up_fwd
import torch
from tqdm import tqdm
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--render_path", required=True,
        help="Directory to the COLMAP model of the render path")
    parser.add_argument("--out_dir",required=True,
        help="Directory to save the rendered images and video")
    parser.add_argument("--alignment_path", default="",
                        help="COLMAP model of the training camera poses"
                             "used for aligning the render path to the scene.")
    parser.add_argument("--framerate", type=int, default=30)
    args = parser.parse_args()

    print("Loading scene")
    scene_model = SceneModel.from_scene(args.model_path, args)

    # Read and parse the render path
    render_cameras, render_images, _ = read_model(args.render_path)

    # Read the path used to align the render path to the scene
    if args.alignment_path != "":
        scene_images_names = [
            keyframe.info["name"] for keyframe in scene_model.keyframes
        ]
        scene_cam_centers = torch.stack(
            [keyframe.get_centre() for keyframe in scene_model.keyframes]
        ).to("cuda")
        scene_Rts = torch.stack(
            [keyframe.get_Rt() for keyframe in scene_model.keyframes]
        ).to("cuda")
        _, alignment_images, _ = read_model(args.alignment_path)

        # only keep input poses that are in the scene keyframes
        alignment_extrinsics_dict = {
            os.path.basename(extr.name): extr for extr in alignment_images.values()
        }
        alignment_extrinsics = [
            alignment_extrinsics_dict[name]
            for name in scene_images_names
            if name in alignment_extrinsics_dict
        ]

        alignment_Rts = torch.eye(4).to("cuda").repeat(len(alignment_extrinsics), 1, 1)
        alignment_cam_centers = torch.zeros((len(alignment_extrinsics), 3)).to("cuda")
        for idx, cam_extrinsics in enumerate(alignment_extrinsics):
            alignment_Rts[idx][:3, :3] = torch.Tensor(qvec2rotmat(cam_extrinsics.qvec))
            alignment_Rts[idx][:3, 3] = torch.Tensor(cam_extrinsics.tvec)
            alignment_cam_centers[idx] = -alignment_Rts[idx][:3, :3].T @ alignment_Rts[idx][:3, 3]


    os.makedirs(args.out_dir, exist_ok=True)
    render_camera = render_cameras[list(render_cameras.keys())[0]]
    # Open the video writer
    out = cv2.VideoWriter(
        os.path.join(args.out_dir, "rendered_path.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.framerate,
        (render_camera.width, render_camera.height),
        True,
    )

    print("Rendering frames")
    for render_image in tqdm(render_images.values()):
        # Get the rendering parameters
        render_camera = render_cameras[render_image.camera_id]
        fov_x = focal2fov(render_camera.params[0], render_camera.width)
        fov_y = focal2fov(render_camera.params[0], render_camera.height)

        # Get the camera pose
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = torch.Tensor(qvec2rotmat(render_image.qvec)).to("cuda")
        Rt[:3, 3] = torch.Tensor(render_image.tvec).to("cuda")

        # Align the camera pose to the scene
        if args.alignment_path != "":
            # Get the training cameras closest to the camera to render
            n_closest = 10
            Rt_inv = torch.linalg.inv(Rt)
            path_cam_center = Rt_inv[:3, 3]
            distances = torch.linalg.norm(
                path_cam_center[None] - alignment_cam_centers, dim=1
            )
            closest_indices = distances.topk(n_closest, largest=False).indices

            # Get the transform that aligns the alignment cameras to the scene
            R, t, s = get_transform_mean_up_fwd(
                alignment_Rts[closest_indices], scene_Rts[closest_indices], True
            )

            # Apply the transform to the camera to render
            Rt_inv[:3, :3] = R @ Rt_inv[:3, :3]
            Rt_inv[:3, 3] = R @ Rt_inv[:3, 3] * s + t
            Rt = torch.linalg.inv(Rt_inv)

        # Render
        render = scene_model.render(
            width=render_camera.width,
            height=render_camera.height,
            fov_x=fov_x,
            fov_y=fov_y,
            view_matrix=(Rt.transpose(0, 1)).to("cuda"),
            scaling_modifier=1.0,
        )["render"].clamp(0, 1.0)

        # Save the frame
        frame = render.mul(255).permute(1, 2, 0).byte().cpu().numpy()[:, :, ::-1]
        # cv2.imwrite(os.path.join(args.out_dir, render_image.name), frame)
        out.write(frame)

    # Release the VideoWriter
    out.release()