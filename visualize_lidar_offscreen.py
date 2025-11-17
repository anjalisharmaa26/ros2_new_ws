import json
import numpy as np
import open3d as o3d
import sys

# Arguments
pcd_file = sys.argv[1]   # LiDAR .bin
json_file = sys.argv[2]  # prediction JSON
out_file = sys.argv[3]   # output PNG

# Load point cloud
pc = np.fromfile(pcd_file, dtype=np.float32).reshape(-1,4)[:,:3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)

# Load prediction JSON
with open(json_file, 'r') as f:
    data = json.load(f)

boxes = []

# Loop through 3D bounding boxes
for i, box in enumerate(data["bboxes_3d"]):
    # box = [x, y, z, dx, dy, dz, yaw]
    center = box[:3]
    dx, dy, dz = box[3:6]
    yaw = box[6]

    # Open3D oriented bounding box
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0,0,yaw])
    obb = o3d.geometry.OrientedBoundingBox(center, R, [dx, dy, dz])
    obb.color = (0.6, 0, 0.8)  # purple
    boxes.append(obb)

# Offscreen render
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(pcd)
for b in boxes:
    vis.add_geometry(b)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image(out_file)
vis.destroy_window()
print(f"Saved Open3D visualization to {out_file}")
