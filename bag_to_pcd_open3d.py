import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import os
from rclpy.serialization import deserialize_message

# --- Configuration ---
bag_path = "/home/anjali26/bags/rosbag2_2025_07_31-17_24_16_0.db3"
topic_name = "/zed/zed_node/point_cloud/cloud_registered"  # <-- change if your topic is different
output_dir = "/home/anjali26/pcd_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Initialize ROS2 bag reader ---
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions('', '')
reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

frame = 0
print(f"Extracting from topic: {topic_name}")

# --- Read and convert each frame ---
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == topic_name:
        msg = deserialize_message(data, PointCloud2)

        # Convert PointCloud2 to XYZ numpy array
        points = np.array([
            [x, y, z]
            for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ])

        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            save_path = os.path.join(output_dir, f"frame_{frame:04d}.pcd")
            o3d.io.write_point_cloud(save_path, pcd)
            print(f" Saved frame_{frame:04d}.pcd ({len(points)} points)")
            frame += 1

print(f"\n All done! Saved {frame} frames to {output_dir}")

