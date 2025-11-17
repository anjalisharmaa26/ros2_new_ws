from mmdet3d.apis import init_model, inference_detector
import os
import numpy as np

# -------------------- USER SETTINGS --------------------
config_file = '/home/anjali26/mmdetection3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py'
checkpoint_file = '/home/anjali26/mmdetection3d/checkpoints/second_kitti_fp16_car.pth'
pointcloud_folder = '/home/anjali26/mmdetection3d/data/rosbag_points/'  # folder with .bin frames
output_folder = './results'
device = 'cuda:0'
# -------------------------------------------------------

# Create results folder
os.makedirs(output_folder, exist_ok=True)

# Initialize model
model = init_model(config_file, checkpoint_file, device=device)

# Loop through all point cloud files
pointcloud_files = sorted([f for f in os.listdir(pointcloud_folder) if f.endswith('.bin')])

for pc_file in pointcloud_files:
    pc_path = os.path.join(pointcloud_folder, pc_file)
    
    # Run inference
    results = inference_detector(model, pc_path)
    
    # Save results as .ply
    out_file = os.path.join(output_folder, pc_file.replace('.bin', '_result.ply'))
    model.show(results, pc_path, out_file=out_file, show=False)
    
    print(f"Inference done for {pc_file}, saved at {out_file}")

print("All files processed! Check the ./results folder for visualization.")
