import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse
import torch.nn.functional as F

from droid import Droid
from droid_async import DroidAsync

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(rgb_dir, calib, stride=1, depth_dir=None):
    """Image generator for mono or RGB-D mode with timestamp matching"""

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    rgb_list = sorted(os.listdir(rgb_dir))[::stride]

    if depth_dir:
        depth_list = sorted(os.listdir(depth_dir))[::stride]

        # Extract timestamps without extensions
        rgb_times = [os.path.splitext(f)[0] for f in rgb_list]
        depth_times = [os.path.splitext(f)[0] for f in depth_list]

        # Find matching timestamps
        common_times = sorted(zip(rgb_times, depth_times), key=lambda x: x[0])

        # Create dicts for fast lookup
        rgb_map = {t: f for t, f in zip(rgb_times, rgb_list)}
        depth_map = {t: f for t, f in zip(depth_times, depth_list)}

        print(f"[INFO] Matched {len(common_times)} RGB-Depth pairs out of "
              f"{len(rgb_list)} RGB and {len(depth_list)} depth frames.")

        for idx, (rgb_ts, depth_ts) in enumerate(common_times):
            rgb_file = rgb_map[rgb_ts]
            depth_file = depth_map[depth_ts]

            image = cv2.imread(os.path.join(rgb_dir, rgb_file))
            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1 - h1 % 8, :w1 - w1 % 8]
            image_torch = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            depth = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_ANYDEPTH) / 256.0
            depth = cv2.resize(depth, (w1, h1))
            depth = depth[:h1 - h1 % 8, :w1 - w1 % 8]
            depth_torch = torch.as_tensor(depth)

            yield idx, image_torch[None], depth_torch, intrinsics

    else:
        # Mono mode (no depth matching needed)
        for idx, rgb_file in enumerate(rgb_list):
            image = cv2.imread(os.path.join(rgb_dir, rgb_file))
            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1 - h1 % 8, :w1 - w1 % 8]
            image_torch = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield idx, image_torch[None], intrinsics


def save_reconstruction(droid, save_path):
    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir", type=str, required=True, help="Path to RGB image directory")
    parser.add_argument("--depth_dir", type=str, help="Path to depth image directory (optional for RGB-D mode)")
    parser.add_argument("--calib", type=str, required=True, help="Path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="Starting frame index")
    parser.add_argument("--stride", default=1, type=int, help="Frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")

    parser.add_argument("--reconstruction_path", help="Path to save reconstruction .pth file")

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    args.stereo = False
    droid = None

    # Enable high resolution depth saving
    if args.reconstruction_path:
        args.upsample = True

    tstamps = []
    if args.depth_dir:
        # RGB-D mode
        for (t, image, depth, intrinsics) in tqdm(image_stream(args.rgb_dir, args.calib, args.stride, depth_dir=args.depth_dir)):
            if t < args.t0:
                continue
            if not args.disable_vis:
                show_image(image[0])
            if droid is None:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = DroidAsync(args) if args.asynchronous else Droid(args)
            droid.track(t, image, depth, intrinsics=intrinsics)
    else:
        # Monocular mode
        for (t, image, intrinsics) in tqdm(image_stream(args.rgb_dir, args.calib, args.stride)):
            if t < args.t0:
                continue
            if not args.disable_vis:
                show_image(image[0])
            if droid is None:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = DroidAsync(args) if args.asynchronous else Droid(args)
            droid.track(t, image, intrinsics=intrinsics)

    print("[INFO] Tracking complete. Terminating...")

    # traj_est = droid.terminate(image_stream(args.rgb_dir, args.calib, args.stride))
    droid.terminate(image_stream(args.rgb_dir, args.calib, args.stride))

    if args.reconstruction_path:
        save_reconstruction(droid, args.reconstruction_path)