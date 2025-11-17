from mmdet3d.apis import init_model, inference_detector
import os

# Paths
config_file = '/home/anjali26/mmdetection3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py'
checkpoint_file = '/home/anjali26/mmdetection3d/checkpoints/second_kitti_fp16_car.pth'
pointcloud_file = '/home/anjali26/mmdetection3d/data/kitti/testing/velodyne/000001.bin'  # replace with your .bin file
output_folder = './results'

# Check if point cloud file exists
if not os.path.exists(pointcloud_file):
    raise FileNotFoundError(f"Point cloud file not found: {pointcloud_file}\n"
                            "Make sure your .bin file is in the correct path.")

# Create results folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Run inference
results = inference_detector(model, pointcloud_file)

# Save results as .ply using the model's built-in function
out_file = os.path.join(output_folder, '000001_result.ply')
model.show_results(results, out_file=out_file, show=False)  # note: show_results replaces show_result

print(f"Inference done! Check the results in: {out_file}")

