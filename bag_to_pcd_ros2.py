import os
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sensor_msgs.msg
import rosbag2_py
import pcl
import numpy as np
from sensor_msgs_py import point_cloud2

bag_path = "/home/anjali26/bags/rosbag2_2025_07_31-17_24_16_0.db3"
topic_name = "/velodyne_points"
output_dir = "/home/anjali26/pcd_outputs"

os.makedirs(output_dir, exist_ok=True)

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions('', '')
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}
msg_type = get_message(type_map[topic_name])

frame = 0
while reader.has_next():
    (topic, data, t) = reader.read_next()
    if topic == topic_name:
        msg = deserialize_message(data, msg_type)
        cloud_points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if len(cloud_points) == 0:
            continue
        np_points = np.array(cloud_points, dtype=np.float32)
        cloud = pcl.PointCloud()
        cloud.from_array(np_points)
        filename = os.path.join(output_dir, f"frame_{frame:05d}.pcd")
        pcl.save(cloud, filename)
        print(f"Saved: {filename}")
        frame += 1

print(" Conversion complete!")
