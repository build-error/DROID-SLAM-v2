import sys
sys.path.append("droid_slam")

import os
import torch
import numpy as np
import cv2
import argparse

import droid_backends
from lietorch import SE3
from cuda_timer import CudaTimer

def rotmat_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)."""
    m = R
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    return qw, qx, qy, qz

def save_reconstruction_colmap(filename: str, filter_thresh=0.005, filter_count=2, output_dir="colmap_output"):
    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # --- Load reconstruction ---
    reconstruction_blob = torch.load(filename)
    
    images = reconstruction_blob["images"].cuda()
    images_downsampled = images[..., ::2, ::2]
    disps = reconstruction_blob["disps"].cuda()[..., ::2, ::2]
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()
    num_images = poses.shape[0]

    # --- Save camera intrinsics ---
    fx, fy, cx, cy = [float(x) for x in intrinsics[0]]
    with open(os.path.join(output_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {int(images.shape[3])} {int(images.shape[2])} {fx} {fy} {cx} {cy}\n")

    # --- Save images and camera poses ---
    with open(os.path.join(output_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(num_images):
            pose_mat = SE3(poses[i:i+1]).inv().matrix().cpu().numpy()[0]
            R = pose_mat[:3, :3]
            t = pose_mat[:3, 3]
            qw, qx, qy, qz = rotmat_to_quaternion(R)
            image_name = f"{i:05d}.png"
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {image_name}\n")
            f.write("\n")  # No 2D points

            # Save image
            img = images[i].permute(1,2,0).cpu().numpy()
            img = np.clip(img, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, "images", image_name), img_bgr)

    # --- Backproject points ---
    disps = disps.contiguous()
    index = torch.arange(len(images_downsampled), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])

    colors = images_downsampled[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0

    # --- Filter points ---
    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = np.clip(colors[mask].cpu().numpy(), 0, 1)

    points3D_path = os.path.join(output_dir, "points3D.txt")
    with open(points3D_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (p, c) in enumerate(zip(points_np, colors_np)):
            X, Y, Z = p
            Rcol, Gcol, Bcol = (c * 255).astype(int).tolist()
            f.write(f"{i+1} {X:.6f} {Y:.6f} {Z:.6f} {Rcol} {Gcol} {Bcol} 0\n")

    print(f"💾 Saved {len(points_np)} colored 3D points to {points3D_path}")
    print(f"📂 COLMAP files saved to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="path to reconstruction file (e.g., bench.pth)")
    parser.add_argument("--output", type=str, default="colmap_output", help="output directory for COLMAP files")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    args = parser.parse_args()

    save_reconstruction_colmap(args.filename, args.filter_threshold, args.filter_count, output_dir=args.output)