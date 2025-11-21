import sys
sys.path.append("droid_slam")

import torch
import argparse
import numpy as np
import pyvista as pv

import droid_backends
from lietorch import SE3
from cuda_timer import CudaTimer

def create_camera_frustum(K_vec, pose, scale=0.2):
    """
    K_vec: 1D array [fx, fy, cx, cy]
    pose: 4x4 camera-to-world matrix
    """
    fx, fy, cx, cy = K_vec[:4]

    # image width/height (approx)
    w, h = 2*cx, 2*cy

    # Define corners in camera space
    z = scale
    corners = np.array([
        [ z*(0 - cx)/fx,  z*(0 - cy)/fy, z],
        [ z*(w - cx)/fx,  z*(0 - cy)/fy, z],
        [ z*(w - cx)/fx,  z*(h - cy)/fy, z],
        [ z*(0 - cx)/fx,  z*(h - cy)/fy, z],
        [0,0,0]  # camera center
    ])

    # Transform to world coordinates
    corners_h = np.hstack([corners, np.ones((5,1))])
    world_corners = (pose @ corners_h.T).T[:, :3]

    # Lines connecting camera center to corners
    lines = [
        [0,4], [1,4], [2,4], [3,4],
        [0,1], [1,2], [2,3], [3,0]
    ]

    # Create PyVista mesh
    pts = world_corners
    mesh = pv.PolyData()
    mesh.points = pts
    mesh.lines = np.hstack([[2, a, b] for a,b in lines])
    return mesh

def view_reconstruction(filename: str, filter_thresh=0.005, filter_count=2, off_screen=True):
    # --- Load reconstruction ---
    reconstruction_blob = torch.load(filename)
    images = reconstruction_blob["images"].cuda()[..., ::2, ::2]
    disps = reconstruction_blob["disps"].cuda()[..., ::2, ::2]
    poses = reconstruction_blob["poses"].cuda()
    print(poses.shape)
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()

    disps = disps.contiguous()
    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    # --- Back-project points ---
    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0

    # --- Filter points ---
    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    # --- Setup PyVista plotter ---
    pv.set_jupyter_backend(None)
    plotter = pv.Plotter(off_screen=off_screen, window_size=(960, 960))
    plotter.set_background("white")

    # --- Add point cloud ---
    cloud = pv.PolyData(points_np)
    cloud["colors"] = colors_np
    plotter.add_points(cloud, scalars="colors", rgb=True, point_size=2, render_points_as_spheres=False)

    # --- Add camera frustums ---
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()
    K = intrinsics[0].cpu().numpy()
    for i in range(0, len(pose_mats), max(1, len(pose_mats)//10)):  # sample some cameras
        frustum = create_camera_frustum(K, pose_mats[i], scale=0.2)
        plotter.add_mesh(frustum, color="red", line_width=1)

    # --- Display or save ---
    if off_screen:
        plotter.show(screenshot="reconstruction.png")
        print("✅ Saved visualization to reconstruction.png (off-screen mode)")
    else:
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="path to reconstruction file (e.g., bench.pth)")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    parser.add_argument("--on_screen", action="store_true", help="use GUI window if supported")
    args = parser.parse_args()

    view_reconstruction(
        args.filename,
        args.filter_threshold,
        args.filter_count,
        off_screen=not args.on_screen
    )