#!/usr/bin/env python3
"""
Convert TUM RGB-D groundtruth.txt to COLMAP sparse format.

TUM format: timestamp tx ty tz qx qy qz qw
COLMAP sparse format:
    - sparse/0/cameras.txt / cameras.bin
    - sparse/0/images.txt / images.bin
    - sparse/0/points3D.txt / points3D.bin (empty)
"""

import os
import sys
import argparse
import numpy as np
import trimesh
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataloaders.read_write_model import (
    write_images_binary, write_cameras_binary, Image, Camera
)
from dataloaders.read_write_model import qvec2rotmat


def parse_list(filepath, skiprows=0):
    """Parse space-delimited text file."""
    data = np.loadtxt(filepath, delimiter=" ", dtype=np.str_, skiprows=skiprows)
    return data


def associate_frames(tstamp_image, tstamp_pose, max_dt=0.08):
    """Associate image timestamps with pose timestamps."""
    associations = []
    for i, t in enumerate(tstamp_image):
        k = np.argmin(np.abs(tstamp_pose.copy() - t))
        if (np.abs(tstamp_pose[k].copy() - t) < max_dt):
            associations.append((i, k))
    return associations


def load_tum_poses(datapath):
    """Load poses from TUM groundtruth.txt and rgb.txt."""
    if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
        pose_list = os.path.join(datapath, "groundtruth.txt")
    elif os.path.isfile(os.path.join(datapath, "pose.txt")):
        pose_list = os.path.join(datapath, "pose.txt")
    else:
        raise FileNotFoundError(f"No groundtruth.txt or pose.txt in {datapath}")

    image_list = os.path.join(datapath, "rgb.txt")
    if not os.path.isfile(image_list):
        image_list = os.path.join(datapath, "rgb.txt")
    if not os.path.isfile(image_list):
        raise FileNotFoundError(f"No rgb.txt in {datapath}")

    image_data = parse_list(image_list)
    pose_data = parse_list(pose_list, skiprows=1)
    pose_vecs = pose_data[:, 0:].astype(np.float64)

    tstamp_image = image_data[:, 0].astype(np.float64)
    pose_tt = np.array([tt + "00" for tt in pose_data[:, 0]], dtype='<U25')
    tstamp_pose = pose_tt.astype(np.float64)
    associations = associate_frames(tstamp_image, tstamp_pose)

    frames = []
    for (i, k) in associations:
        quat = pose_vecs[k][4:]  # qx, qy, qz, qw
        trans = pose_vecs[k][1:4]  # tx, ty, tz
        
        # Convert quaternion (qx, qy, qz, qw) to rotation matrix
        # TUM uses (qx, qy, qz, qw) format
        T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))  # roll to (qw, qx, qy, qz)
        T[:3, 3] = trans

        frame = {
            "img_name": os.path.basename(image_data[i, 1]),
            "timestamp": tstamp_image[i],
            "transform_matrix": T
        }
        frames.append(frame)
    
    return frames


def create_colmap_sparse(frames, output_dir, camera_params=None):
    """
    Create COLMAP sparse format from TUM frames.
    
    Args:
        frames: List of frames with img_name, timestamp, transform_matrix
        output_dir: Output directory for sparse model
        camera_params: Optional camera parameters (model, width, height, fx, fy, cx, cy)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "0"), exist_ok=True)

    # Default camera parameters if not provided
    # Using typical RGB-D camera parameters (e.g., Kinect)
    if camera_params is None:
        camera_params = {
            "model": "PINHOLE",
            "width": 640,
            "height": 480,
            "fx": 525.0,
            "fy": 525.0,
            "cx": 319.5,
            "cy": 239.5
        }

    # Create cameras
    cameras = {}
    camera_id = 1
    cameras[camera_id] = Camera(
        id=camera_id,
        model=camera_params["model"],
        width=camera_params["width"],
        height=camera_params["height"],
        params=np.array([
            camera_params["fx"],
            camera_params["fy"],
            camera_params["cx"],
            camera_params["cy"]
        ])
    )

    # Create images
    # TUM poses are camera-to-world, COLMAP needs world-to-camera (W2C)
    # The transform_matrix from TUM is C2W, we need W2C for COLMAP
    images = {}
    for idx, frame in enumerate(frames):
        img_id = idx + 1
        
        # Get the C2W transform
        T_c2w = frame["transform_matrix"]
        
        # Convert to W2C (COLMAP format)
        T_w2c = np.linalg.inv(T_c2w)
        
        # Extract rotation and translation
        R_w2c = T_w2c[:3, :3]
        t_w2c = T_w2c[:3, 3]
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        qvec = rotmat2qvec(R_w2c)
        tvec = t_w2c
        
        images[img_id] = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=1,
            name=frame["img_name"],
            xys=np.array([]),
            point3D_ids=np.array([], dtype=np.int64)
        )

    # Write binary format
    cameras_path = os.path.join(output_dir, "0", "cameras.bin")
    images_path = os.path.join(output_dir, "0", "images.bin")
    points3d_path = os.path.join(output_dir, "0", "points3D.bin")
    
    write_cameras_binary(cameras, cameras_path)
    write_images_binary(images, images_path)
    
    # Write empty points3D with proper binary format (num_points = 0)
    with open(points3d_path, "wb") as fid:
        import struct
        fid.write(struct.pack("<Q", 0))  # Write 0 as uint64
    
    # Also write text versions for debugging
    write_cameras_text(cameras, os.path.join(output_dir, "0", "cameras.txt"))
    write_images_text(images, os.path.join(output_dir, "0", "images.txt"))
    write_points3D_text({}, os.path.join(output_dir, "0", "points3D.txt"))
    
    print(f"Written {len(frames)} images to {output_dir}")
    return cameras, images


def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def write_cameras_text(cameras, path):
    """Write cameras in text format."""
    with open(path, "w") as fid:
        for _, cam in cameras.items():
            fid.write(f"{cam.id} {cam.model} {cam.width} {cam.height} ")
            fid.write(" ".join(map(str, cam.params)) + "\n")


def write_images_text(images, path):
    """Write images in text format."""
    with open(path, "w") as fid:
        for _, img in images.items():
            fid.write(f"# Image ID {img.id}\n")
            fid.write(f"{img.id} {img.qvec[0]} {img.qvec[1]} {img.qvec[2]} {img.qvec[3]} ")
            fid.write(f"{img.tvec[0]} {img.tvec[1]} {img.tvec[2]} ")
            fid.write(f"{img.camera_id} {img.name}\n")
            fid.write("\n")


def write_points3D_text(points3D, path):
    """Write points3D in text format."""
    with open(path, "w") as fid:
        fid.write("# 3D point list is empty\n")
        fid.write("# ID X Y Z R G B ERROR TRACK[]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TUM groundtruth to COLMAP sparse format")
    parser.add_argument("--input", type=str, required=True, help="Path to TUM dataset (containing groundtruth.txt and rgb.txt)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for COLMAP sparse model")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--fx", type=float, default=525.0, help="Focal length x")
    parser.add_argument("--fy", type=float, default=525.0, help="Focal length y")
    parser.add_argument("--cx", type=float, default=319.5, help="Principal point x")
    parser.add_argument("--cy", type=float, default=239.5, help="Principal point y")
    args = parser.parse_args()

    print(f"Loading TUM dataset from {args.input}")
    frames = load_tum_poses(args.input)
    print(f"Loaded {len(frames)} poses")

    camera_params = {
        "model": "PINHOLE",
        "width": args.width,
        "height": args.height,
        "fx": args.fx,
        "fy": args.fy,
        "cx": args.cx,
        "cy": args.cy
    }

    create_colmap_sparse(frames, args.output, camera_params)
    print(f"Done! COLMAP sparse model written to {args.output}")
    print(f"Run: colmap gui --database_path <db_path> --image_path <image_path> --import_path {args.output}")