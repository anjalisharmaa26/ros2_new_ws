import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pickle  # for MMDetection3D prediction outputs (if using pickle)
import os

# -----------------------------
# 1️⃣ Load KITTI LiDAR .bin file
# -----------------------------
def load_kitti_bin(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # x, y, z

# -----------------------------
# 2️⃣ Load MMDetection3D predicted boxes
# -----------------------------
def load_mmdet3d_boxes(pred_file):
    """
    Load predictions from MMDetection3D inference.
    Example: saved as a pickle or .npy file containing:
        [{'boxes_3d': ndarray(N, 7), 'scores_3d': ndarray(N,), 'labels_3d': ndarray(N,)}]
    """
    if pred_file.endswith(".pkl"):
        with open(pred_file, "rb") as f:
            preds = pickle.load(f)
    elif pred_file.endswith(".npy"):
        preds = np.load(pred_file, allow_pickle=True)
    else:
        raise ValueError("Unsupported file type: use .pkl or .npy")
    
    bboxes = []
    for obj in preds:
        boxes_3d = obj["boxes_3d"]  # (N, 7) -> x, y, z, dx, dy, dz, yaw
        for box in boxes_3d:
            x, y, z, dx, dy, dz, yaw = box
            # Create AxisAlignedBoundingBox (approximation)
            bbox = o3d.geometry.OrientedBoundingBox(
                center=[x, y, z],
                R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0, 0, yaw]),
                extent=[dx, dy, dz]
            )
            bbox.color = (1, 0, 1)  # purple
            bboxes.append(bbox)
    return bboxes

# -----------------------------
# 3️⃣ Main visualization
# -----------------------------
def visualize_lidar_with_boxes(lidar_file, pred_file=None, save_path="lidar_output.png"):
    # Load point cloud
    points = load_kitti_bin(lidar_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 1, 1])  # white points

    # Load prediction boxes if provided
    bboxes = []
    if pred_file and os.path.exists(pred_file):
        bboxes = load_mmdet3d_boxes(pred_file)
    else:
        print("No prediction file found, using example boxes")
        # Example static boxes if prediction not provided
        bboxes.append(o3d.geometry.AxisAlignedBoundingBox([0, -1, -1], [4, 1, 1]))
        bboxes.append(o3d.geometry.AxisAlignedBoundingBox([5, 2, -1], [8, 4, 1]))
        for bbox in bboxes:
            bbox.color = (1, 0, 1)

    # Offscreen rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    for bbox in bboxes:
        vis.add_geometry(bbox)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()

    # Show with matplotlib
    img = plt.imread(save_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    print(f"Saved output image: {save_path}")

# -----------------------------
# 4️⃣ Run the visualization
# -----------------------------
if __name__ == "__main__":
    lidar_file = "demo/data/kitti/000008.bin"  # <-- change to your .bin file
    pred_file = "demo/data/kitti/000008_pred.pkl"  # <-- change to your model output
    visualize_lidar_with_boxes(lidar_file, pred_file)
