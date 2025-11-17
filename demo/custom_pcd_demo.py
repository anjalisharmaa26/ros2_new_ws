import os
from mmcv import Config
from mmdet3d.apis import inference_model, show_result_meshlab
from mmdet3d.models import build_model
import torch

# -----------------------------
# Config and checkpoint paths
# -----------------------------
config_file = 'configs/pointpillars/hv_pointpillars_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint_file = 'checkpoints/pointpillars_kitti_car.pth'

# -----------------------------
# Load config and remove test_cfg in outer scope
# -----------------------------
cfg = Config.fromfile(config_file)
cfg.model.test_cfg = None  # Avoid the double test_cfg issue

# -----------------------------
# Build model manually
# -----------------------------
device = 'cpu'
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
model.CLASSES = ('Car',)
model = model.to(device)
model.eval()

# -----------------------------
# Directory containing PCDs
# -----------------------------
pcd_dir = os.path.expanduser('data/custom_pcd')
output_dir = os.path.expanduser('data/custom_pcd_results')
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Run inference on all PCDs
# -----------------------------
pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])

for pcd_file in pcd_files:
    pcd_path = os.path.join(pcd_dir, pcd_file)
    print(f'Processing {pcd_path} ...')

    # Inference
    result, data = inference_model(model, pcd_path)

    # Save visualization using MeshLab
    out_file = os.path.join(output_dir, pcd_file.replace('.pcd', '_result.ply'))
    show_result_meshlab(
        model,
        pcd_path,
        result,
        out_file=out_file,
        show=False
    )

print('✅ All PCDs processed! Results saved in', output_dir)
