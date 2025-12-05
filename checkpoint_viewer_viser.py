import argparse
import numpy as np
import torch
import time
from pathlib import Path
import glob
import torch.nn.functional as F

# Try to import the viewer from the simple_trainer location
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from gsplat_viewer import GsplatViewer, GsplatRenderTabState
    from nerfview import CameraState
    GSPLAT_VIEWER_AVAILABLE = True
    print("✓ gsplat_viewer loaded successfully")
except ImportError as e:
    print(f"Warning: gsplat_viewer not available: {e}")
    GSPLAT_VIEWER_AVAILABLE = False

import viser
from gsplat.rendering import rasterization


def load_checkpoint(ckpt_path):
    """Load a checkpoint file."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    step = ckpt['step']
    splats = ckpt['splats']
    
    print(f"Checkpoint at step: {step}")
    print(f"Number of Gaussians: {len(splats['means'])}")
    
    return step, splats


def view_checkpoint_with_gsplat_viewer(ckpt_path, port=8080):
    """View checkpoint using the official gsplat viewer."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    step, splats_dict = load_checkpoint(ckpt_path)
    
    # Move to device
    for key in splats_dict:
        splats_dict[key] = splats_dict[key].to(device)
    
    print(f"\nSetting up viewer...")
    print(f"Device: {device}")
    print(f"Gaussians: {len(splats_dict['means']):,}")
    
    # Create viser server
    server = viser.ViserServer(port=port, verbose=False)
    
    # Define render function for gsplat viewer
    def render_fn(camera_state: CameraState, render_tab_state: GsplatRenderTabState):
        """Render function called by gsplat viewer."""
        
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        
        # Get camera parameters
        c2w = camera_state.c2w  # [4, 4]
        K = camera_state.get_K((width, height))  # [3, 3]
        
        # Convert to torch
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        
        # Prepare colors
        colors = torch.cat([splats_dict['sh0'], splats_dict['shN']], 1)  # [N, K, 3]
        
        # Render
        with torch.no_grad():
            renders, alphas, info = rasterization(
                means=splats_dict['means'],
                quats=F.normalize(splats_dict['quats'], dim=-1),
                scales=torch.exp(splats_dict['scales']),
                opacities=torch.sigmoid(splats_dict['opacities']),
                colors=colors,
                viewmats=torch.linalg.inv(c2w.unsqueeze(0)),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                sh_degree=min(render_tab_state.max_sh_degree, 3),
                radius_clip=render_tab_state.radius_clip,
                backgrounds=torch.tensor([render_tab_state.backgrounds], device=device) / 255.0,
            )
        
        # Update stats
        render_tab_state.total_gs_count = len(splats_dict['means'])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()
        
        # Return RGB render
        render = renders[0, ..., 0:3].clamp(0, 1).cpu().numpy()
        return render
    
    # Create gsplat viewer
    viewer = GsplatViewer(
        server=server,
        render_fn=render_fn,
        output_dir=Path(ckpt_path).parent.parent,
        mode="rendering",
    )
    
    print("\n" + "="*60)
    print(f"Gsplat Viewer running at: http://localhost:{port}")
    print("="*60)
    print(f"Checkpoint: Step {step}")
    print(f"Gaussians: {len(splats_dict['means']):,}")
    print("\nThe viewer is now rendering Gaussians!")
    print("Move the camera to see the scene.")
    print("Press Ctrl+C to exit.\n")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def view_checkpoint_basic(ckpt_path, port=8080, server=None):
    """Basic viewer without gsplat_viewer."""
    from gsplat.rendering import rasterization
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    step, splats = load_checkpoint(ckpt_path)
    
    # Move to device
    for key in splats:
        splats[key] = splats[key].to(device)
    
    # Create server if not provided
    if server is None:
        server = viser.ViserServer(port=port, verbose=False)
    
    # Calculate scene bounds for camera
    points = splats['means'].cpu().numpy()
    center = points.mean(axis=0)
    
    # Add text info
    with server.gui.add_folder("Checkpoint Info"):
        server.gui.add_text(
            "Info",
            initial_value=(
                f"Step: {step}\n"
                f"Gaussians: {len(splats['means']):,}\n"
                f"Use gsplat_viewer for interactive rendering\n"
                f"Install: pip install git+https://github.com/nerfstudio-project/gsplat.git"
            ),
        )
    
    # Add a simple point cloud visualization as fallback
    # Downsample for display
    max_points = 100000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        display_points = points[indices]
    else:
        display_points = points
    
    # Get colors from SH
    C0 = 0.28209479177387814
    colors = splats['sh0'].cpu().numpy()[:, 0, :]
    colors = colors * C0 + 0.5
    colors = np.clip(colors, 0, 1)
    
    if len(points) > max_points:
        colors = colors[indices]
    
    server.scene.add_point_cloud(
        name="/gaussians",
        points=display_points,
        colors=colors,
        point_size=0.01,
    )
    
    print("\n" + "="*60)
    print(f"Basic Viewer running at: http://localhost:{port}")
    print("="*60)
    print(f"Checkpoint: Step {step}")
    print(f"Gaussians: {len(splats['means']):,}")
    print("\nNote: This is a point cloud visualization")
    print("For proper Gaussian rendering, install gsplat_viewer:")
    print("  pip install git+https://github.com/nerfstudio-project/gsplat.git")
    print("\nPress Ctrl+C to exit.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def compare_checkpoints(ckpt_dir, port=8080):
    """Compare multiple checkpoints."""
    
    # Find all checkpoints
    ckpt_files = sorted(glob.glob(f"{ckpt_dir}/ckpt_*.pt"))
    
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        return
    
    print(f"Found {len(ckpt_files)} checkpoints:")
    for i, ckpt_file in enumerate(ckpt_files):
        ckpt = torch.load(ckpt_file, map_location='cpu')
        step = ckpt['step']
        num_gs = len(ckpt['splats']['means'])
        print(f"  [{i}] Step {step:6d}: {num_gs:,} Gaussians")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load initial checkpoint
    current_idx = [len(ckpt_files) - 1]  # Start with last checkpoint
    step, splats_dict = load_checkpoint(ckpt_files[current_idx[0]])
    for key in splats_dict:
        splats_dict[key] = splats_dict[key].to(device)
    
    current_splats = [splats_dict]
    current_step = [step]
    
    # Create server
    server = viser.ViserServer(port=port, verbose=False)
    
    # Define render function
    def render_fn(camera_state: CameraState, render_tab_state: GsplatRenderTabState):
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
        
        colors = torch.cat([current_splats[0]['sh0'], current_splats[0]['shN']], 1)
        
        with torch.no_grad():
            renders, alphas, info = rasterization(
                means=current_splats[0]['means'],
                quats=F.normalize(current_splats[0]['quats'], dim=-1),
                scales=torch.exp(current_splats[0]['scales']),
                opacities=torch.sigmoid(current_splats[0]['opacities']),
                colors=colors,
                viewmats=torch.linalg.inv(c2w.unsqueeze(0)),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                sh_degree=min(render_tab_state.max_sh_degree, 3),
                radius_clip=render_tab_state.radius_clip,
                backgrounds=torch.tensor([render_tab_state.backgrounds], device=device) / 255.0,
            )
        
        render_tab_state.total_gs_count = len(current_splats[0]['means'])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()
        
        return renders[0, ..., 0:3].clamp(0, 1).cpu().numpy()
    
    # Create viewer
    viewer = GsplatViewer(
        server=server,
        render_fn=render_fn,
        output_dir=Path(ckpt_dir).parent,
        mode="rendering",
    )
    
    # Add checkpoint selector
    with server.gui.add_folder("Checkpoint Selector"):
        checkpoint_dropdown = server.gui.add_dropdown(
            "Select Checkpoint",
            options=[Path(f).stem for f in ckpt_files],
            initial_value=Path(ckpt_files[current_idx[0]]).stem,
        )
        
        info_text = server.gui.add_text(
            "Info",
            initial_value=f"Step: {step}\nGaussians: {len(splats_dict['means']):,}",
        )
        
        @checkpoint_dropdown.on_update
        def _(event):
            idx = [Path(f).stem for f in ckpt_files].index(event.target.value)
            print(f"Loading checkpoint {idx} (step {torch.load(ckpt_files[idx])['step']})...")
            
            step, new_splats = load_checkpoint(ckpt_files[idx])
            for key in new_splats:
                new_splats[key] = new_splats[key].to(device)
            
            current_splats[0] = new_splats
            current_step[0] = step
            current_idx[0] = idx
            
            info_text.value = f"Step: {step}\nGaussians: {len(new_splats['means']):,}"
            print(f"Loaded checkpoint at step {step}")
    
    print("\n" + "="*60)
    print(f"Checkpoint Comparison Viewer running at: http://localhost:{port}")
    print("="*60)
    print(f"Current: Step {step} with {len(splats_dict['means']):,} Gaussians")
    print("\nUse the dropdown to switch between checkpoints")
    print("Press Ctrl+C to exit.\n")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def print_checkpoint_stats(ckpt_dir):
    """Print statistics for all checkpoints."""
    ckpt_files = sorted(glob.glob(f"{ckpt_dir}/ckpt_*.pt"))
    
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        return
    
    print("\n" + "="*80)
    print(f"{'Step':<10} {'Gaussians':<15} {'Mean Scale':<15} {'Mean Opacity':<15}")
    print("="*80)
    
    for ckpt_file in ckpt_files:
        ckpt = torch.load(ckpt_file, map_location='cpu')
        step = ckpt['step']
        splats = ckpt['splats']
        
        num_gs = len(splats['means'])
        mean_scale = torch.exp(splats['scales']).mean().item()
        mean_opacity = torch.sigmoid(splats['opacities']).mean().item()
        
        print(f"{step:<10} {num_gs:<15,} {mean_scale:<15.6f} {mean_opacity:<15.6f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="View training checkpoints with Gaussian Splatting")
    parser.add_argument(
        "command",
        choices=["view", "compare", "stats"],
        help="Command to run"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (for 'view')"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="results/output/ckpts",
        help="Directory containing checkpoints (for 'compare' and 'stats')"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viser server"
    )
    
    args = parser.parse_args()
    
    if args.command == "view":
        if not args.checkpoint:
            print("Error: --checkpoint required for 'view' command")
            return
        
        if GSPLAT_VIEWER_AVAILABLE:
            view_checkpoint_with_gsplat_viewer(args.checkpoint, args.port)
        else:
            view_checkpoint_basic(args.checkpoint, args.port)
    
    elif args.command == "compare":
        compare_checkpoints(args.ckpt_dir, args.port)
    
    elif args.command == "stats":
        print_checkpoint_stats(args.ckpt_dir)


if __name__ == "__main__":
    main()