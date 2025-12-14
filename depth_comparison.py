
import os
from pathlib import Path
import random
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Dataset loader

def collect_test_pairs(test_dir):
    """Collect (image_path, depth_path) pairs recursively."""
    test_dir = Path(test_dir)
    pairs = []
    for img_path in test_dir.rglob("image/*"):
        depth_path = img_path.parent.parent / "depth" / img_path.name
        if depth_path.exists():
            pairs.append((img_path, depth_path))
    return sorted(pairs)


# Depth prediction function

def estimate_depth(midas, transform, img, device):
    img_cv = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_cv).to(device)
    with torch.no_grad():
        pred_depth = midas(input_tensor)
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        elif pred_depth.dim() == 4 and pred_depth.shape[1] != 1:
            pred_depth = pred_depth.mean(dim=1, keepdim=True)
        pred_depth = F.interpolate(pred_depth, size=img_cv.shape[:2], mode="bilinear", align_corners=False)
        pred_depth = pred_depth.squeeze().cpu().numpy()
    return pred_depth


# Visualization 

def visualize_comparison(images, dpt_large_maps, dpt_hybrid_maps, gt_maps, titles=None):
    n = len(images)
    plt.figure(figsize=(16, 4*n))
    for i in range(n):
        # Input image
        plt.subplot(n,4,i*4+1)
        plt.imshow(images[i])
        plt.title("Input")
        plt.axis("off")

        # DPT-Large
        plt.subplot(n,4,i*4+2)
        plt.imshow(dpt_large_maps[i], cmap='plasma')
        plt.title("DPT-Large")
        plt.axis("off")

        # DPT-Hybrid
        plt.subplot(n,4,i*4+3)
        plt.imshow(dpt_hybrid_maps[i], cmap='plasma')
        plt.title("DPT-Hybrid")
        plt.axis("off")

        # Ground truth 
        plt.subplot(n,4,i*4+4)
        plt.imshow(gt_maps[i], cmap='plasma_r')  
        plt.title("Ground Truth")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main(test_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MiDaS models
    print("Loading DPT-Large...")
    midas_large = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform_large = midas_transforms.dpt_transform

    print("Loading DPT-Hybrid...")
    midas_hybrid = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device).eval()
    transform_hybrid = midas_transforms.dpt_transform

    # Collect dataset
    pairs = collect_test_pairs(test_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No image/depth pairs found in {test_dir}")

    print(f"Found {len(pairs)} samples, selecting 5 randomly...")
    selected_pairs = random.sample(pairs, min(5, len(pairs)))

    images, dpt_large_maps, dpt_hybrid_maps, gt_maps = [], [], [], []

    for img_path, depth_path in selected_pairs:
        # Input image
        img_cv = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        images.append(img_cv)

        # DPT-Large prediction
        pred_large = estimate_depth(midas_large, transform_large, img_path, device)
        # scale to GT median
        gt_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        scale = np.median(gt_depth) / (np.median(pred_large) + 1e-6)
        pred_large *= scale
        dpt_large_maps.append(pred_large)

        # DPT-Hybrid prediction
        pred_hybrid = estimate_depth(midas_hybrid, transform_hybrid, img_path, device)
        pred_hybrid *= scale
        dpt_hybrid_maps.append(pred_hybrid)

        # Ground truth
        if gt_depth.ndim == 3:
            gt_depth = gt_depth[:,:,0]
        gt_maps.append(gt_depth)

    # Visualization
    visualize_comparison(images, dpt_large_maps, dpt_hybrid_maps, gt_maps, titles=None)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.test_dir)
