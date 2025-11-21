import torch
import numpy as np
import os
from gsplat import GaussianRenderer
from colmap_utils import load_colmap_points, load_colmap_cameras

# Config
DATA_DIR = "bench_colmap"
LR = 1e-3
N_ITERS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load COLMAP data
    points3D = load_colmap_points(os.path.join(DATA_DIR, "points3D.txt"))
    cameras = load_colmap_cameras(os.path.join(DATA_DIR, "cameras.txt"))

    print(f"Loaded {points3D.shape[0]} points and {len(cameras)} cameras")

    # Convert to torch tensors
    means = torch.tensor(points3D[:, :3], dtype=torch.float32, device=DEVICE, requires_grad=True)
    colors = torch.tensor(points3D[:, 3:6], dtype=torch.float32, device=DEVICE, requires_grad=True)
    scales = torch.ones((points3D.shape[0], 1), dtype=torch.float32, device=DEVICE, requires_grad=True) * 0.01
    opacities = torch.ones((points3D.shape[0], 1), dtype=torch.float32, device=DEVICE, requires_grad=True) * 0.8

    # Make sure these are LEAF tensors
    means = torch.nn.Parameter(means)
    colors = torch.nn.Parameter(colors)
    scales = torch.nn.Parameter(scales)
    opacities = torch.nn.Parameter(opacities)

    # Renderer
    renderer = GaussianRenderer(device=DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(
        [means, colors, scales, opacities],
        lr=LR
    )

    # Dummy training loop (replace with actual loss computation)
    for i in range(N_ITERS):
        optimizer.zero_grad()

        # Render scene (example; adjust depending on gsplat API)
        img = renderer.render(means, colors, scales, opacities, cameras)

        # Example dummy loss (just to make optimization valid)
        loss = img.mean()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"[Iter {i}] Loss = {loss.item():.6f}")

    print("Training complete!")

if __name__ == "__main__":
    main()