import sys
sys.path.append("droid_slam")

import os
import torch
import numpy as np
import cv2
import argparse
import struct
from pathlib import Path

import droid_backends
from lietorch import SE3
from cuda_timer import CudaTimer

def write_cameras_text(cameras, path_to_model_file):
    """Write COLMAP cameras.txt file"""
    with open(path_to_model_file, "w") as fid:
        fid.write("# Camera list with one line of data per camera:\n")
        fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fid.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_id, cam in sorted(cameras.items()):
            params_str = " ".join([f"{p:.10f}" for p in cam["params"]])
            fid.write(f"{camera_id} PINHOLE {cam['width']} {cam['height']} {params_str}\n")

def write_images_text(images, path_to_model_file):
    """Write COLMAP images.txt file"""
    with open(path_to_model_file, "w") as fid:
        fid.write("# Image list with two lines of data per image:\n")
        fid.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fid.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fid.write(f"# Number of images: {len(images)}\n")
        for image_id, img in sorted(images.items()):
            qvec = img["qvec"]
            tvec = img["tvec"]
            qvec_str = " ".join([f"{q:.10f}" for q in qvec])
            tvec_str = " ".join([f"{t:.10f}" for t in tvec])
            fid.write(f"{image_id} {qvec_str} {tvec_str} {img['camera_id']} {img['name']}\n")
            
            # Write 2D points (empty line if no points)
            if len(img["xys"]) == 0:
                fid.write("\n")
            else:
                points_str = " ".join([f"{x:.10f} {y:.10f} {pid}" 
                                      for (x, y), pid in zip(img["xys"], img["point3D_ids"])])
                fid.write(f"{points_str}\n")

def write_points3D_text(points3D, path_to_model_file):
    """Write COLMAP points3D.txt file"""
    with open(path_to_model_file, "w") as fid:
        fid.write("# 3D point list with one line of data per point:\n")
        fid.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fid.write(f"# Number of points: {len(points3D)}\n")
        for point3D_id, pt in sorted(points3D.items()):
            xyz = pt["xyz"]
            rgb = pt["rgb"]
            xyz_str = " ".join([f"{coord:.10f}" for coord in xyz])
            rgb_str = " ".join([str(int(c)) for c in rgb])
            
            # Write track (empty if no track)
            track_str = ""
            if len(pt["track"]) > 0:
                track_str = " " + " ".join([f"{img_id} {p2d_idx}" 
                                           for img_id, p2d_idx in pt["track"]])
            
            fid.write(f"{point3D_id} {xyz_str} {rgb_str} {pt['error']:.10f}{track_str}\n")

def project_points_to_images(points_np, colors_np, poses_list, intrinsics, width, height, max_points_per_image=1000):
    """
    Project 3D points into images to create tracks for COLMAP visualization
    
    Args:
        points_np: [N, 3] array of 3D points
        colors_np: [N, 3] array of colors
        poses_list: List of (R, t) tuples for each image
        intrinsics: Camera intrinsics [fx, fy, cx, cy]
        width: Image width
        height: Image height
        max_points_per_image: Maximum points to associate per image
    
    Returns:
        points3D_dict: Dictionary with point tracks
        images_dict_update: Dictionary with 2D observations for each image
    """
    fx, fy, cx, cy = intrinsics
    num_points = len(points_np)
    num_images = len(poses_list)
    
    print(f"🔄 Creating point tracks by projecting {num_points} points into {num_images} images...")
    print(f"   This may take a few minutes for large point clouds...")
    
    points3D_dict = {}
    images_observations = {i + 1: {"xys": [], "point3D_ids": []} for i in range(num_images)}
    
    # For each point, find which images can see it
    points_with_tracks = 0
    for pt_idx, (point, color) in enumerate(zip(points_np, colors_np)):
        if (pt_idx + 1) % 100000 == 0:
            print(f"   Processing point {pt_idx + 1}/{num_points}...")
        
        track = []
        
        for img_idx, (R, t) in enumerate(poses_list):
            # Transform point to camera coordinates (world to camera)
            # Camera pose is camera-to-world, so we need world-to-camera
            R_inv = R.T
            t_inv = -R_inv @ t
            point_cam = R_inv @ point + t_inv
            
            # Check if point is in front of camera
            if point_cam[2] <= 0.01:
                continue
            
            # Project to image
            x = fx * point_cam[0] / point_cam[2] + cx
            y = fy * point_cam[1] / point_cam[2] + cy
            
            # Check if projection is within image bounds (with small margin)
            margin = 5
            if -margin <= x < width + margin and -margin <= y < height + margin:
                point2D_idx = len(images_observations[img_idx + 1]["xys"])
                track.append((img_idx + 1, point2D_idx))
                images_observations[img_idx + 1]["xys"].append([x, y])
                images_observations[img_idx + 1]["point3D_ids"].append(pt_idx + 1)
        
        # Only add point if it's visible in at least 2 images
        if len(track) >= 2:
            points3D_dict[pt_idx + 1] = {
                "xyz": point,
                "rgb": (color * 255).astype(np.uint8),
                "error": 1.0,
                "track": track
            }
            points_with_tracks += 1
    
    # Convert lists to numpy arrays
    for img_idx in images_observations:
        xys = images_observations[img_idx]["xys"]
        point3D_ids = images_observations[img_idx]["point3D_ids"]
        
        if len(xys) > 0:
            # Optionally limit number of points per image for performance
            if max_points_per_image is not None and len(xys) > max_points_per_image:
                print(f"   ⚠ Image {img_idx} has {len(xys)} observations, sampling {max_points_per_image} for performance...")
                indices = np.random.choice(len(xys), max_points_per_image, replace=False)
                indices = sorted(indices)  # Keep in order
                xys = [xys[i] for i in indices]
                point3D_ids = [point3D_ids[i] for i in indices]
            
            images_observations[img_idx]["xys"] = np.array(xys)
            images_observations[img_idx]["point3D_ids"] = np.array(point3D_ids, dtype=np.int64)
        else:
            images_observations[img_idx]["xys"] = np.zeros((0, 2))
            images_observations[img_idx]["point3D_ids"] = np.zeros(0, dtype=np.int64)
    
    print(f"✓ Created tracks for {points_with_tracks}/{num_points} points (visible in 2+ images)")
    
    return points3D_dict, images_observations

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
    return np.array([qw, qx, qy, qz])
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
    return np.array([qw, qx, qy, qz])

def save_image_with_downsamples(img, base_path, image_name, downsample_factors=[2, 4, 8]):
    """
    Save image at full resolution and downsampled versions
    
    Args:
        img: Image array (H, W, 3)
        base_path: Base output directory
        image_name: Image filename
        downsample_factors: List of downsampling factors
    """
    # Save full resolution image
    images_dir = os.path.join(base_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(images_dir, image_name), img_bgr)
    
    # Save downsampled versions
    for factor in downsample_factors:
        downsample_dir = os.path.join(base_path, f"images_{factor}")
        os.makedirs(downsample_dir, exist_ok=True)
        
        h, w = img.shape[:2]
        new_h, new_w = h // factor, w // factor
        img_downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_downsampled_bgr = cv2.cvtColor(img_downsampled, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(downsample_dir, image_name), img_downsampled_bgr)

def create_poses_bounds_npy(poses_list, intrinsics, height, width, near=0.1, far=100.0):
    """
    Create poses_bounds.npy file in LLFF format
    
    Format: [N, 17] array where each row is:
    [R (3x3 flattened, row-major), t (3x1), h, w, focal]
    followed by near and far bounds
    
    Args:
        poses_list: List of (R, t) tuples for each image
        intrinsics: Camera intrinsics [fx, fy, cx, cy]
        height: Image height
        width: Image width
        near: Near plane
        far: Far plane
    """
    N = len(poses_list)
    poses_bounds = np.zeros((N, 17))
    
    fx, fy, cx, cy = intrinsics
    
    for i, (R, t) in enumerate(poses_list):
        # LLFF format: first 12 values are 3x4 pose matrix (R|t) flattened row-major
        # But we store in camera-to-world format (inverse of COLMAP's world-to-camera)
        pose_mat = np.concatenate([R, t.reshape(3, 1)], axis=1)  # 3x4
        poses_bounds[i, :12] = pose_mat.flatten()
        
        # Store height, width, focal
        poses_bounds[i, 12] = height
        poses_bounds[i, 13] = width
        poses_bounds[i, 14] = fx  # focal length
        
        # Store near and far bounds
        poses_bounds[i, 15] = near
        poses_bounds[i, 16] = far
    
    return poses_bounds

def save_reconstruction_360v2(filename: str, filter_thresh=0.005, filter_count=2, 
                              output_dir="colmap_output", max_points=None,
                              downsample_factors=[2, 4, 8], near=0.1, far=100.0,
                              output_format="txt", create_tracks=True):
    """
    Convert DROID-SLAM reconstruction to complete Mip-NeRF 360_v2 format
    
    Directory structure (matching 360_v2 datasets):
    output_dir/
        ├── images/              # Full resolution images
        ├── images_2/            # Images downsampled by 2x
        ├── images_4/            # Images downsampled by 4x
        ├── images_8/            # Images downsampled by 8x
        ├── poses_bounds.npy     # LLFF format poses
        └── sparse/
            └── 0/
                ├── cameras.txt (or .bin)
                ├── images.txt (or .bin)
                └── points3D.txt (or .bin)
    
    Args:
        filename: Path to DROID-SLAM .pth file
        filter_thresh: Threshold for depth filtering
        filter_count: Minimum view count for 3D points
        output_dir: Output directory
        max_points: Maximum number of 3D points to save
        downsample_factors: List of downsampling factors for images
        near: Near plane for poses_bounds.npy
        far: Far plane for poses_bounds.npy
        output_format: Output format for COLMAP files ('txt' or 'bin')
        create_tracks: Whether to create point tracks for COLMAP visualization
    """
    
    # Create directory structure
    sparse_dir = os.path.join(output_dir, "sparse", "0_txt")
    os.makedirs(sparse_dir, exist_ok=True)

    print(f"📂 Creating Mip-NeRF 360_v2 format structure at {output_dir}")
    print(f"   ├── images/")
    for factor in downsample_factors:
        print(f"   ├── images_{factor}/")
    print(f"   ├── poses_bounds.npy")
    print(f"   └── sparse/0_txt/")

    # Load reconstruction
    print(f"\n🔄 Loading DROID-SLAM reconstruction from {filename}...")
    reconstruction_blob = torch.load(filename)
    
    images = reconstruction_blob["images"].cuda()
    images_downsampled = images[..., ::2, ::2]
    disps = reconstruction_blob["disps"].cuda()[..., ::2, ::2]
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()
    num_images = poses.shape[0]

    print(f"✓ Found {num_images} camera poses")
    print(f"✓ Image size: {images.shape[3]}x{images.shape[2]}")

    # Extract camera intrinsics
    fx, fy, cx, cy = [float(x) for x in intrinsics[0]]
    width = int(images.shape[3])
    height = int(images.shape[2])
    
    print(f"✓ Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Create COLMAP cameras dictionary
    cameras = {
        1: {
            "model_id": 1,  # PINHOLE model
            "width": width,
            "height": height,
            "params": [fx, fy, cx, cy]
        }
    }

    # Save images and create poses list
    print(f"\n💾 Saving {num_images} images at multiple resolutions...")
    images_dict = {}
    poses_list = []
    
    for i in range(num_images):
        # Get pose matrix (camera to world)
        pose_mat = SE3(poses[i:i+1]).inv().matrix().cpu().numpy()[0]
        R = pose_mat[:3, :3]
        t = pose_mat[:3, 3]

        # F = np.diag([1, 1, -1])   # flip Z axis
        # R = F @ R
        # t = F @ t
        
        # Store for poses_bounds.npy
        poses_list.append((R, t))
        
        # Convert rotation to quaternion for COLMAP
        qvec = rotmat_to_quaternion(R)
        tvec = t
        
        # Image name
        image_name = f"{i:05d}.png"
        
        images_dict[i + 1] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": 1,
            "name": image_name,
            "xys": np.zeros((0, 2)),
            "point3D_ids": np.zeros(0, dtype=np.int64)
        }

        # Save image at all resolutions
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        save_image_with_downsamples(img, output_dir, image_name, downsample_factors)
        
        if (i + 1) % 20 == 0:
            print(f"  Saved {i+1}/{num_images} images...")

    print(f"✓ Saved images at resolutions: 1x, {', '.join([f'{f}x' for f in downsample_factors])}")

    # Create poses_bounds.npy
    print(f"\n📊 Creating poses_bounds.npy...")
    poses_bounds = create_poses_bounds_npy(poses_list, [fx, fy, cx, cy], height, width, near, far)
    np.save(os.path.join(output_dir, "poses_bounds.npy"), poses_bounds)
    print(f"✓ Saved poses_bounds.npy with shape {poses_bounds.shape}")

    # Backproject points to 3D
    print(f"\n🔄 Backprojecting 3D points from depth maps...")
    disps = disps.contiguous()
    index = torch.arange(len(images_downsampled), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])

    colors = images_downsampled[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0

    # Filter points
    print(f"🔄 Filtering 3D points (threshold={filter_thresh}, count={filter_count})...")
    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    # points_np[:, 2] *= -1
    colors_np = np.clip(colors[mask].cpu().numpy(), 0, 1)

    # Subsample points if max_points is specified
    if max_points is not None and len(points_np) > max_points:
        print(f"⚠ Subsampling from {len(points_np)} to {max_points} points for memory efficiency...")
        indices = np.random.choice(len(points_np), max_points, replace=False)
        points_np = points_np[indices]
        colors_np = colors_np[indices]

    print(f"✓ Generated {len(points_np)} filtered 3D points")

    # Create point tracks by projecting into images (for COLMAP visualization)
    if create_tracks:
        points3D_dict, images_observations = project_points_to_images(
            points_np, colors_np, poses_list, [fx, fy, cx, cy], width, height
        )
        
        # Update images_dict with 2D observations
        for img_id in images_dict:
            images_dict[img_id]["xys"] = images_observations[img_id]["xys"]
            images_dict[img_id]["point3D_ids"] = images_observations[img_id]["point3D_ids"]
    else:
        # Create COLMAP points3D dictionary without tracks
        points3D_dict = {}
        for i, (p, c) in enumerate(zip(points_np, colors_np)):
            points3D_dict[i + 1] = {
                "xyz": p,
                "rgb": (c * 255).astype(np.uint8),
                "error": 1.0,
                "track": []
            }

    # Write COLMAP files
    print(f"\n💾 Writing COLMAP {output_format.upper()} files to sparse/0_txt/...")
    
    file_ext = f".{output_format}"
    cameras_file = os.path.join(sparse_dir, f"cameras{file_ext}")
    images_file = os.path.join(sparse_dir, f"images{file_ext}")
    points3D_file = os.path.join(sparse_dir, f"points3D{file_ext}")
    
    if output_format == "txt":
        write_cameras_text(cameras, cameras_file)
        print(f"✓ Written cameras.txt")
        
        write_images_text(images_dict, images_file)
        print(f"✓ Written images.txt")
        
        write_points3D_text(points3D_dict, points3D_file)
        print(f"✓ Written points3D.txt")
    else:
        write_cameras_binary(cameras, cameras_file)
        print(f"✓ Written cameras.bin")
        
        write_images_binary(images_dict, images_file)
        print(f"✓ Written images.bin")
        
        write_points3D_binary(points3D_dict, points3D_file)
        print(f"✓ Written points3D.bin")

    print(f"\n✅ Conversion complete!")
    print(f"📂 Mip-NeRF 360_v2 format saved to: {os.path.abspath(output_dir)}")
    print(f"\n📋 Summary:")
    print(f"   • Cameras: 1")
    print(f"   • Images: {num_images}")
    print(f"   • Image resolutions: 1x")
    for factor in downsample_factors:
        print(f"   • Image resolutions: {factor}x ({width//factor}x{height//factor})")
    print(f"   • 3D Points: {len(points_np)}")
    print(f"   • Poses bounds: {poses_bounds.shape}")
    print(f"\n🔧 Ready for:")
    print(f"   1. Mip-NeRF 360 / Gaussian Splatting / other NeRF methods")
    print(f"   2. COLMAP GUI: colmap gui --import_path {os.path.abspath(sparse_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DROID-SLAM to complete Mip-NeRF 360_v2 format")
    parser.add_argument("filename", type=str, help="Path to DROID-SLAM reconstruction file (e.g., bench.pth)")
    parser.add_argument("--output", type=str, default="colmap_output", 
                        help="Output directory for Mip-NeRF 360_v2 format (default: colmap_output)")
    parser.add_argument("--filter_threshold", type=float, default=0.005,
                        help="Threshold for depth filtering (default: 0.005)")
    parser.add_argument("--filter_count", type=int, default=3,
                        help="Minimum view count for 3D points (default: 3)")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Maximum number of 3D points to save (default: None, keep all)")
    parser.add_argument("--downsample", nargs="+", type=int, default=[2, 4, 8],
                        help="Downsampling factors for images (default: 2 4 8)")
    parser.add_argument("--near", type=float, default=0.1,
                        help="Near plane for poses_bounds.npy (default: 0.1)")
    parser.add_argument("--far", type=float, default=100.0,
                        help="Far plane for poses_bounds.npy (default: 100.0)")
    parser.add_argument("--format", type=str, choices=["txt", "bin"], default="txt",
                        help="Output format for COLMAP files: txt or bin (default: txt)")
    parser.add_argument("--no-tracks", action="store_true",
                        help="Don't create point tracks (faster but points won't show in COLMAP GUI)")
    args = parser.parse_args()

    save_reconstruction_360v2(
        args.filename, 
        args.filter_threshold, 
        args.filter_count, 
        output_dir=args.output, 
        max_points=args.max_points,
        downsample_factors=args.downsample,
        near=args.near,
        far=args.far,
        output_format=args.format,
        create_tracks=not args.no_tracks
    )