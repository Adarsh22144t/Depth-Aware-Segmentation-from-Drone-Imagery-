
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # DPT_Hybrid


# Read image

def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth map {path}")
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return depth.astype(np.float32)


# Depth estimation

def estimate_depth(image, crop_size=512):
    H, W = image.shape[:2]
    crop_img = image[:min(crop_size, H), :min(crop_size, W), :]

    input_batch = transform(crop_img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=crop_img.shape[:2],
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Normalize depth to 0-1
    pred_min, pred_max = prediction.min(), prediction.max()
    depth_normalized = (prediction - pred_min) / (pred_max - pred_min + 1e-6)
    return depth_normalized


# Metrics

def compute_metrics(pred, gt):
    rmse = mean_squared_error(gt, pred, squared=False)
    mae = mean_absolute_error(gt, pred)
    mse = mean_squared_error(gt, pred)
    corr = np.corrcoef(gt.flatten(), pred.flatten())[0,1]
    return {"RMSE": rmse, "MAE": mae, "MSE": mse, "Correlation": corr}

def evaluate_test_folder(test_dir, crop_size=512, print_every=10):
    test_dir = Path(test_dir)
    image_paths = sorted(test_dir.glob("neighbourhood/*/image/*.png"))

    metrics_list = []
    for idx, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
        # Corresponding depth map
        depth_path = img_path.parent.parent / "depth" / img_path.name
        if not depth_path.exists():
            print(f"Depth map missing for {img_path}, skipping.")
            continue

        # Read image and ground truth
        img = read_rgb(img_path)
        gt_depth = read_depth(depth_path)
        gt_depth = gt_depth[:crop_size, :crop_size] 

        # Normalize GT to 0-1
        gt_min, gt_max = gt_depth.min(), gt_depth.max()
        gt_norm = (gt_depth - gt_min) / (gt_max - gt_min + 1e-6)

        # Estimate depth
        pred_depth = estimate_depth(img, crop_size=crop_size)

        # Compute metrics
        metrics = compute_metrics(pred_depth, gt_norm)
        metrics_list.append(metrics)

        if (idx + 1) % print_every == 0:
            avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
            print(f"After {idx+1} images: {avg_metrics}")

    # Final metrics
    if metrics_list:
        final_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        print("\nFinal Metrics on Test Set:")
        for k, v in final_metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    test_folder = "/Users/sadik2/main_project/test"  
    evaluate_test_folder(test_folder)
