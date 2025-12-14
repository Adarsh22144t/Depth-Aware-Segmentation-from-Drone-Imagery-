
import os
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Metrics

def compute_errors(gt, pred):
    
    thresh = np.maximum(gt / pred, pred / gt)
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25**2).mean()
    delta3 = (thresh < 1.25**3).mean()
    rmse   = np.sqrt(((gt - pred) ** 2).mean())
    mae    = np.mean(np.abs(gt - pred))
    abs_rel= np.mean(np.abs(gt - pred) / gt)
    return rmse, mae, abs_rel, delta1, delta2, delta3


# Dataset loader

def collect_test_pairs(test_dir):
    test_dir = Path(test_dir)
    pairs = []
    for img_path in test_dir.rglob("image/*"):
        # corresponding depth path
        depth_path = img_path.parent.parent / "depth" / img_path.name
        if depth_path.exists():
            pairs.append((img_path, depth_path))
    return sorted(pairs)

# Main evaluation
def main(test_dir, depth_scale=1000.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DPT_Hybrid model
    print("Loading MiDaS DPT_Hybrid...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform 

    # Collect test pairs
    pairs = collect_test_pairs(test_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No image/depth pairs found in {test_dir}")
    print(f"Found {len(pairs)} test examples.")

    all_rmse, all_mae, all_absrel, all_delta1, all_delta2, all_delta3 = [], [], [], [], [], []

    for img_path, depth_path in tqdm(pairs, desc="Evaluating"):
        # Load input image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img).to(device)

        with torch.no_grad():
            pred_depth = midas(input_tensor)
            if pred_depth.dim() == 3:
                pred_depth = pred_depth.unsqueeze(1)
            elif pred_depth.dim() == 4 and pred_depth.shape[1] != 1:
                pred_depth = pred_depth.mean(dim=1, keepdim=True)
            # resize to original image size
            pred_depth = F.interpolate(pred_depth, size=img.shape[:2], mode="bilinear", align_corners=False)
            pred_depth = pred_depth.squeeze().cpu().numpy()

        # Load ground truth depth 
        gt_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if gt_depth.ndim == 3:
            gt_depth = gt_depth[:, :, 0]
        gt_depth = gt_depth.astype(np.float32) / depth_scale  # convert to meters

        # Scale prediction 
        scale = np.median(gt_depth[gt_depth > 0]) / (np.median(pred_depth) + 1e-6)
        pred_depth = pred_depth * scale

        # Remove invalid GT pixels
        mask = gt_depth > 0
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # Compute metrics
        rmse, mae, absrel, d1, d2, d3 = compute_errors(gt_depth, pred_depth)
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_absrel.append(absrel)
        all_delta1.append(d1)
        all_delta2.append(d2)
        all_delta3.append(d3)

    # Report averaged metrics
    print("\nEvaluation results:")
    print(f"RMSE   : {np.mean(all_rmse):.4f} m")
    print(f"MAE    : {np.mean(all_mae):.4f} m")
    print(f"AbsRel : {np.mean(all_absrel):.4f}")
    print(f"Delta1 : {np.mean(all_delta1):.4f}")
    print(f"Delta2 : {np.mean(all_delta2):.4f}")
    print(f"Delta3 : {np.mean(all_delta3):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Divide GT values to convert 16-bit PNG to meters")
    args = parser.parse_args()
    main(args.test_dir, args.depth_scale)
