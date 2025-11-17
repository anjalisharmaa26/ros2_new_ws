import json
import numpy as np
import open3d as o3d
import os

# ---------- CONFIG ----------
pcd_file = "demo/data/kitti/000008.bin"
pred_file = "outputs/preds/000008.json"
output_image = "outputs/open3d_render.png"
# ----------------------------

os.makedirs("outputs", exist_ok=True)

# Load point cloud
points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.paint_uniform_color([0.5,0.5,0.5])  # gray points

# Load bounding boxes
with open(pred_file) as f:
    data = json.load(f)

bboxes = []
for det in data.get("boxes_3d", []):
    x,y,z,dx,dy,dz,yaw = det
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = [x,y,z]
    bbox.extent = [dx,dy,dz]
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])
    bbox.R = R
    bbox.color = (1,0,0)  # red boxes
    bboxes.append(bbox)

# Offscreen rendering
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)  # no GUI
vis.add_geometry(pcd)
for bbox in bboxes:
    vis.add_geometry(bbox)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image(output_image)
vis.destroy_window()

print(f"Rendered image saved at: {output_image}")
