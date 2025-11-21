# import sys
# sys.path.append('droid_slam')

# from tqdm import tqdm
# import numpy as np
# import torch
# import lietorch
# import cv2
# import os
# import glob 
# import time
# import argparse

# from torch.multiprocessing import Process
# from droid import Droid
# from droid_async import DroidAsync

# import torch.nn.functional as F


# def show_image(image):
#     image = image.permute(1, 2, 0).cpu().numpy()
#     cv2.imshow('image', image / 255.0)
#     cv2.waitKey(1)

# def image_stream(imagedir, calib, stride):
#     """ image generator """

#     calib = np.loadtxt(calib, delimiter=" ")
#     fx, fy, cx, cy = calib[:4]

#     K = np.eye(3)
#     K[0,0] = fx
#     K[0,2] = cx
#     K[1,1] = fy
#     K[1,2] = cy

#     image_list = sorted(os.listdir(imagedir))[::stride]

#     for t, imfile in enumerate(image_list):
#         image = cv2.imread(os.path.join(imagedir, imfile))
#         if len(calib) > 4:
#             image = cv2.undistort(image, K, calib[4:])

#         h0, w0, _ = image.shape
#         h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
#         w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

#         image = cv2.resize(image, (w1, h1))
#         image = image[:h1-h1%8, :w1-w1%8]
#         image = torch.as_tensor(image).permute(2, 0, 1)

#         intrinsics = torch.as_tensor([fx, fy, cx, cy])
#         intrinsics[0::2] *= (w1 / w0)
#         intrinsics[1::2] *= (h1 / h0)

#         yield t, image[None], intrinsics


# def save_reconstruction(droid, save_path):

#     if hasattr(droid, "video2"):
#         video = droid.video2
#     else:
#         video = droid.video

#     t = video.counter.value
#     save_data = {
#         "tstamps": video.tstamp[:t].cpu(),
#         "images": video.images[:t].cpu(),
#         "disps": video.disps_up[:t].cpu(),
#         "poses": video.poses[:t].cpu(),
#         "intrinsics": video.intrinsics[:t].cpu()
#     }

#     torch.save(save_data, save_path)

# def save_traj_tum(traj_est, save_path="trajectory.txt"):
#     """
#     Save trajectory in TUM RGB-D format:
#     timestamp tx ty tz qx qy qz qw
#     """
#     poses = traj_est
#     N = poses.shape[0]

#     # If GPU tensor -> move to CPU
#     if torch.is_tensor(poses):
#         poses = poses.cpu().numpy()

#     with open(save_path, "w") as f:
#         for i in range(N):
#             # use frame index / fps as timestamp if real timestamps not available
#             t = i
#             tx, ty, tz, qx, qy, qz, qw = poses[i]
#             f.write(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

#     print(f"Saved TUM trajectory to {save_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--imagedir", type=str, help="path to image directory")
#     parser.add_argument("--calib", type=str, help="path to calibration file")
#     parser.add_argument("--t0", default=0, type=int, help="starting frame")
#     parser.add_argument("--stride", default=3, type=int, help="frame stride")

#     parser.add_argument("--weights", default="droid.pth")
#     parser.add_argument("--buffer", type=int, default=512)
#     parser.add_argument("--image_size", default=[240, 320])
#     parser.add_argument("--disable_vis", action="store_true")

#     parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
#     parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
#     parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
#     parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
#     parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
#     parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
#     parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
#     parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

#     parser.add_argument("--backend_thresh", type=float, default=22.0)
#     parser.add_argument("--backend_radius", type=int, default=2)
#     parser.add_argument("--backend_nms", type=int, default=3)
#     parser.add_argument("--upsample", action="store_true")
#     parser.add_argument("--asynchronous", action="store_true")
#     parser.add_argument("--frontend_device", type=str, default="cuda")
#     parser.add_argument("--backend_device", type=str, default="cuda")
    
#     parser.add_argument("--reconstruction_path", help="path to saved reconstruction")

#     parser.add_argument("--traj_path", type=str, default="trajectory.txt", help="path to save TUM-format trajectory")
#     args = parser.parse_args()

#     args.stereo = False
#     torch.multiprocessing.set_start_method('spawn')

#     droid = None

#     # need high resolution depths
#     if args.reconstruction_path is not None:
#         args.upsample = True

#     tstamps = []
#     for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
#         if t < args.t0:
#             continue

#         if not args.disable_vis:
#             show_image(image[0])

#         if droid is None:
#             args.image_size = [image.shape[2], image.shape[3]]
#             droid = DroidAsync(args) if args.asynchronous else Droid(args)
        
#         droid.track(t, image, intrinsics=intrinsics)

#     traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
#     print(traj_est)
#     print(traj_est.shape)
#     save_traj_tum(traj_est, args.traj_path)
    
#     if args.reconstruction_path is not None:
#         save_reconstruction(droid, args.reconstruction_path)

import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream(imagedir, calib, stride):
    """Image generator yielding (t, image, intrinsics)"""
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, save_path):
    video = droid.video2 if hasattr(droid, "video2") else droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)


def save_traj_tum(traj_est, image_dir, stride, save_path="trajectory.txt"):
    """Save trajectory in TUM RGB-D format."""
    image_list = sorted(os.listdir(image_dir))[::stride]
    poses = traj_est

    if torch.is_tensor(poses):
        poses = poses.cpu().numpy()

    N = min(len(image_list), poses.shape[0])
    with open(save_path, "w") as f:
        for i in range(N):
            tstamp = os.path.splitext(image_list[i])[0]
            tx, ty, tz, qx, qy, qz, qw = poses[i]
            f.write(f"{tstamp} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    print(f"✅ Saved TUM trajectory to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

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

    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--traj_path", type=str, default="trajectory.txt", help="path to save TUM-format trajectory")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None
    if args.reconstruction_path is not None:
        args.upsample = True

    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)

        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    print("Trajectory shape:", traj_est.shape)

    save_traj_tum(traj_est, args.imagedir, args.stride, args.traj_path)

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
