import numpy as np
import viser
import time
from pathlib import Path
import struct
from plyfile import PlyData
import torch
import torch.nn.functional as F


def sh2rgb(sh_coeffs):
    """Convert spherical harmonics to RGB colors."""
    # SH degree 0 (DC component)
    C0 = 0.28209479177387814
    color = sh_coeffs[..., 0] * C0 + 0.5
    return torch.clamp(color, 0.0, 1.0)


def read_gaussian_ply(ply_path):
    """Read Gaussian Splatting PLY file."""
    plydata = PlyData.read(ply_path)
    
    # Extract vertex data
    vertices = plydata['vertex']
    
    print(f"Available properties: {vertices.data.dtype.names}")
    
    # Get positions
    positions = np.stack([
        np.asarray(vertices['x']),
        np.asarray(vertices['y']),
        np.asarray(vertices['z'])
    ], axis=1)
    
    # Get scales (log scale)
    try:
        scales = np.stack([
            np.asarray(vertices['scale_0']),
            np.asarray(vertices['scale_1']),
            np.asarray(vertices['scale_2'])
        ], axis=1)
    except:
        scales = np.ones_like(positions) * 0.01
    
    # Get rotations (quaternions)
    try:
        rotations = np.stack([
            np.asarray(vertices['rot_0']),
            np.asarray(vertices['rot_1']),
            np.asarray(vertices['rot_2']),
            np.asarray(vertices['rot_3'])
        ], axis=1)
    except:
        rotations = np.array([[1, 0, 0, 0]] * len(positions))
    
    # Get opacity
    try:
        opacity = np.asarray(vertices['opacity'])
    except:
        opacity = np.ones(len(positions))
    
    # Get spherical harmonics (colors)
    sh_names = [name for name in vertices.data.dtype.names if name.startswith('f_dc_')]
    
    if sh_names:
        # Extract DC component (first SH coefficient for RGB)
        try:
            sh_dc = np.stack([
                np.asarray(vertices['f_dc_0']),
                np.asarray(vertices['f_dc_1']),
                np.asarray(vertices['f_dc_2'])
            ], axis=1)
            
            # Convert SH to RGB - the DC component is stored as sh * C0
            # To get RGB we need: RGB = SH_DC * C0 + 0.5
            C0 = 0.28209479177387814
            colors = sh_dc * C0 + 0.5
            colors = np.clip(colors, 0, 1)
            
            print(f"Color range from SH: [{colors.min():.3f}, {colors.max():.3f}]")
        except Exception as e:
            print(f"Error reading SH: {e}")
            colors = np.ones_like(positions) * 0.5
    else:
        # Try standard RGB format
        try:
            colors = np.stack([
                np.asarray(vertices['red']),
                np.asarray(vertices['green']),
                np.asarray(vertices['blue'])
            ], axis=1) / 255.0
            print(f"Color range from RGB: [{colors.min():.3f}, {colors.max():.3f}]")
        except:
            # Try normalized RGB
            try:
                colors = np.stack([
                    np.asarray(vertices['r']),
                    np.asarray(vertices['g']),
                    np.asarray(vertices['b'])
                ], axis=1)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                print(f"Color range from r,g,b: [{colors.min():.3f}, {colors.max():.3f}]")
            except:
                print("Warning: Could not find color data, using gray")
                colors = np.ones_like(positions) * 0.5
    
    print(f"Loaded {len(positions)} Gaussians")
    print(f"Position range: [{positions.min():.2f}, {positions.max():.2f}]")
    print(f"Scale range: [{np.exp(scales).min():.4f}, {np.exp(scales).max():.4f}]")
    print(f"Opacity (raw) range: [{opacity.min():.4f}, {opacity.max():.4f}]")
    print(f"Color range: [{colors.min():.3f}, {colors.max():.3f}]")
    print(f"Color mean: [{colors.mean(axis=0)}]")
    
    # Check if colors are all very dark
    if colors.max() < 0.1:
        print("\n⚠️  WARNING: Colors are very dark!")
        print("This might be because:")
        print("1. The SH coefficients are stored differently")
        print("2. The point cloud hasn't been trained yet")
        print("3. The conversion formula needs adjustment")
        print("\nTrying alternative color extraction...")
        
        # Try without the SH conversion
        try:
            sh_dc = np.stack([
                np.asarray(vertices['f_dc_0']),
                np.asarray(vertices['f_dc_1']),
                np.asarray(vertices['f_dc_2'])
            ], axis=1)
            # Try sigmoid instead
            colors = 1.0 / (1.0 + np.exp(-sh_dc))
            print(f"Alternative color range: [{colors.min():.3f}, {colors.max():.3f}]")
        except:
            pass
    
    return {
        'positions': positions,
        'scales': scales,
        'rotations': rotations,
        'opacity': opacity,
        'colors': colors
    }


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    # Normalize quaternion
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = np.zeros((*q.shape[:-1], 3, 3))
    
    R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    R[..., 1, 2] = 2 * (y * z - w * x)
    
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R


def visualize_gaussians(ply_path, port=8080, max_gaussians=None, show_ellipsoids=False):
    """
    Visualize Gaussian Splatting PLY file with viser.
    
    Args:
        ply_path: Path to the PLY file
        port: Port for the viser server
        max_gaussians: Maximum number of Gaussians to display
        show_ellipsoids: Whether to render Gaussians as ellipsoids (slow for many Gaussians)
    """
    # Load Gaussian PLY file
    print(f"Loading Gaussians from: {ply_path}")
    gaussians = read_gaussian_ply(ply_path)
    
    positions = gaussians['positions']
    colors = gaussians['colors']
    scales = np.exp(gaussians['scales'])  # Convert from log scale
    rotations = gaussians['rotations']
    opacity = 1.0 / (1.0 + np.exp(-gaussians['opacity']))  # Sigmoid
    
    # Filter by opacity
    opacity_threshold = 0.01  # Lower threshold to see more
    valid_mask = opacity > opacity_threshold
    
    print(f"\nFiltering: {valid_mask.sum()} / {len(opacity)} Gaussians pass opacity threshold > {opacity_threshold}")
    
    positions = positions[valid_mask]
    colors = colors[valid_mask]
    scales = scales[valid_mask]
    rotations = rotations[valid_mask]
    opacity = opacity[valid_mask]
    
    print(f"After opacity filtering: {len(positions)} Gaussians")
    
    # If colors are still very dark, boost them
    if colors.max() < 0.3:
        print("⚠️  Boosting color brightness...")
        colors = np.clip(colors * 3.0, 0, 1)  # Boost brightness
        print(f"Boosted color range: [{colors.min():.3f}, {colors.max():.3f}]")
    
    # Downsample if needed
    if max_gaussians is not None and len(positions) > max_gaussians:
        print(f"Downsampling from {len(positions)} to {max_gaussians} Gaussians")
        indices = np.random.choice(len(positions), max_gaussians, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        scales = scales[indices]
        rotations = rotations[indices]
        opacity = opacity[indices]
    
    # Start viser server
    print(f"\nStarting viser server on port {port}...")
    server = viser.ViserServer(port=port, verbose=False)
    
    # Render mode state
    render_state = {'mode': 'points', 'point_size': 0.01, 'ellipsoid_scale': 1.0}
    
    def update_visualization():
        # Clear previous visualization
        server.scene.reset()
        
        if render_state['mode'] == 'points':
            # Render as point cloud (fast)
            server.scene.add_point_cloud(
                name="/gaussians",
                points=positions,
                colors=colors * opacity[:, None],  # Apply opacity to color
                point_size=render_state['point_size'],
            )
        elif render_state['mode'] == 'ellipsoids':
            # Render as ellipsoids (slow but more accurate)
            print("Rendering ellipsoids (this may take a moment)...")
            
            # Only show subset for ellipsoids
            max_ellipsoids = min(1000, len(positions))
            indices = np.random.choice(len(positions), max_ellipsoids, replace=False)
            
            for i in indices:
                # Get rotation matrix
                R = quaternion_to_rotation_matrix(rotations[i:i+1])[0]
                
                # Create ellipsoid
                scale = scales[i] * render_state['ellipsoid_scale']
                color_with_opacity = tuple(colors[i] * opacity[i])
                
                server.scene.add_icosphere(
                    name=f"/gaussians/ellipsoid_{i}",
                    radius=np.mean(scale),  # Average radius
                    position=positions[i],
                    color=color_with_opacity,
                )
            
            print(f"Rendered {max_ellipsoids} ellipsoids")
    
    # Initial visualization
    update_visualization()
    
    # Add GUI controls
    with server.gui.add_folder("Gaussian Controls"):
        render_mode = server.gui.add_dropdown(
            "Render Mode",
            options=["points", "ellipsoids"],
            initial_value="points",
        )
        
        @render_mode.on_update
        def _(event):
            render_state['mode'] = event.target.value
            update_visualization()
        
        point_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=0.01,
        )
        
        @point_size_slider.on_update
        def _(event):
            render_state['point_size'] = event.target.value
            if render_state['mode'] == 'points':
                update_visualization()
        
        ellipsoid_scale_slider = server.gui.add_slider(
            "Ellipsoid Scale",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )
        
        @ellipsoid_scale_slider.on_update
        def _(event):
            render_state['ellipsoid_scale'] = event.target.value
            if render_state['mode'] == 'ellipsoids':
                update_visualization()
        
        # Statistics
        server.gui.add_text(
            "Statistics",
            initial_value=(
                f"Total Gaussians: {len(positions):,}\n"
                f"Avg Scale: {scales.mean():.4f}\n"
                f"Avg Opacity: {opacity.mean():.4f}"
            ),
        )
    
    # Print information
    print("\n" + "="*60)
    print(f"Viser viewer is running at: http://localhost:{port}")
    print("="*60)
    print(f"\nGaussian statistics:")
    print(f"  - Total Gaussians: {len(positions):,}")
    print(f"  - Average scale: {scales.mean():.4f}")
    print(f"  - Average opacity: {opacity.mean():.4f}")
    print(f"  - Center: [{positions.mean(axis=0)[0]:.2f}, {positions.mean(axis=0)[1]:.2f}, {positions.mean(axis=0)[2]:.2f}]")
    print(f"\nControls:")
    print(f"  - Left click + drag: Rotate")
    print(f"  - Right click + drag: Pan")
    print(f"  - Scroll: Zoom")
    print(f"  - Switch between 'points' and 'ellipsoids' mode")
    print(f"\nPress Ctrl+C to exit.\n")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Gaussian Splatting PLY with viser")
    parser.add_argument(
        "ply_path",
        type=str,
        help="Path to the Gaussian Splatting PLY file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viser server (default: 8080)"
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=100000,
        help="Maximum number of Gaussians to display (default: 100000)"
    )
    parser.add_argument(
        "--show-ellipsoids",
        action="store_true",
        help="Show Gaussians as ellipsoids (slower)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.ply_path).exists():
        print(f"Error: File not found: {args.ply_path}")
        return
    
    # Visualize
    visualize_gaussians(
        args.ply_path,
        port=args.port,
        max_gaussians=args.max_gaussians,
        show_ellipsoids=args.show_ellipsoids
    )


if __name__ == "__main__":
    main()