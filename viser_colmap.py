import viser
import numpy as np
from pathlib import Path
import cv2
import time
from scipy.spatial.transform import Rotation as R

# -------------------
# Utility functions
# -------------------

def read_points3D(path):
    """Load COLMAP points3D.txt"""
    pts, colors = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            vals = line.strip().split()
            if len(vals) < 8:
                continue
            x, y, z = map(float, vals[1:4])
            r, g, b = map(float, vals[4:7])
            pts.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(pts), np.array(colors)


def read_images(path):
    """Load COLMAP images.txt"""
    images = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = line.strip().split()
            if len(vals) < 10:
                continue
            image_id = int(vals[0])
            qw, qx, qy, qz = map(float, vals[1:5])
            tx, ty, tz = map(float, vals[5:8])
            image_name = vals[9]
            images.append({
                "id": image_id,
                "q": np.array([qw, qx, qy, qz]),
                "t": np.array([tx, ty, tz]),
                "name": image_name
            })
    return images


# -------------------
# Main
# -------------------

path = Path("bench_colmap")

# Load point cloud
points, colors = read_points3D(path / "points3D.txt")

# Load images and poses
images = read_images(path / "images.txt")

# Start viser server
server = viser.ViserServer(port=8080)

# Add point cloud
server.scene.add_point_cloud(
    name="colmap_points",
    points=points,
    colors=colors,
    point_size=0.002
)

# Add cameras with image textures
image_dir = path / "images"
for img_info in images:
    R_world_cam = R.from_quat(img_info["q"]).as_matrix()
    t_world_cam = img_info["t"]

    # Convert to 4x4 transform
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R_world_cam
    T_world_cam[:3, 3] = t_world_cam

    # Load image texture
    img_path = image_dir / img_info["name"]
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Add a camera frustum with the image
    server.scene.add_camera_frustum(
        name=f"cam_{img_info['id']}",
        fov=45.0,
        aspect=img.shape[1] / img.shape[0],
        scale=0.1,  # adjust camera size in scene
        wxyz=img_info["q"],
        position=img_info["t"],
        image=img,
    )

print("✅ Open http://localhost:8080 to explore your COLMAP model!")

# Keep alive
while True:
    time.sleep(1)