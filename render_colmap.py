import sys
import numpy as np
import pyvista as pv

def qvec2rotmat(q):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])

def load_points3D(file):
    points = []
    with open(file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            data = line.strip().split()
            if len(data) < 7:
                continue
            X, Y, Z = map(float, data[1:4])
            R, G, B = map(float, data[4:7])
            points.append([X, Y, Z, R, G, B])
    return np.array(points)

def load_images(file):
    images = []
    with open(file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            data = line.strip().split()
            if len(data) < 10:
                continue
            q = np.array(list(map(float, data[1:5])))
            t = np.array(list(map(float, data[5:8])))
            name = data[9]
            images.append({"q": q, "t": t, "name": name})
    return images

def create_camera_frustum(center, R, scale=0.1):
    """Create a simple camera frustum as a pyramid mesh."""
    pts = np.array([
        [ scale,  scale, scale*2],
        [-scale,  scale, scale*2],
        [-scale, -scale, scale*2],
        [ scale, -scale, scale*2],
        [ 0, 0, 0]  # apex
    ])
    pts = (R @ pts.T).T + center
    faces = [
        [4, 0, 1, 2, 3],  # base
        [3, 4, 0, 1],     # sides
        [3, 4, 1, 2],
        [3, 4, 2, 3],
        [3, 4, 3, 0]
    ]
    faces_flat = np.hstack(faces)
    return pv.PolyData(pts, faces_flat)

def main(path, frustum_scale=0.2, frustum_color="red", frustum_opacity=0.5):
    points = load_points3D(f"{path}/points3D.txt")
    images = load_images(f"{path}/images.txt")

    print(f"Loaded {len(points)} points and {len(images)} cameras.")

    # Point cloud
    pc = pv.PolyData(points[:, :3])
    pc["RGB"] = points[:, 3:6]  

    # PyVista plotter
    plotter = pv.Plotter()
    plotter.add_points(pc, scalars="RGB", rgb=True, point_size=3)

    # Add camera frustums
    for img in images:
        R = qvec2rotmat(img["q"])
        center = img["t"]  # use original COLMAP camera position
        frustum = create_camera_frustum(center, R, scale=frustum_scale)
        plotter.add_mesh(frustum, color=frustum_color, opacity=frustum_opacity)

    plotter.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: render_colmap.py /path/to/colmap_model [frustum_scale] [frustum_color] [frustum_opacity]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    color = sys.argv[3] if len(sys.argv) > 3 else "red"
    opacity = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    main(model_path, frustum_scale=scale, frustum_color=color, frustum_opacity=opacity)