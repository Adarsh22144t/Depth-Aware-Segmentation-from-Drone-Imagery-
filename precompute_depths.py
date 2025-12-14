import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Paths
ROOT = "/Users/sadik2/main_project/train"  # parent folder containing scenes
train_dir = "/Users/sadik2/main_project/train"  
OUT_DIR = "/Users/sadik2/main_project/depth_precomputed"
os.makedirs(OUT_DIR, exist_ok=True)

# Load MiDaS (small model!)
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # or "MiDaS_small" for even lighter
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.default_transform

device = torch.device("cpu")
midas.to(device)

print("Precomputing depth maps...")
for scene in sorted(os.listdir(train_dir)):
    scene_path = os.path.join(train_dir, scene)
    if not os.path.isdir(scene_path):  # ✅ Skip files like .DS_Store
        continue

    for sub in sorted(os.listdir(scene_path)):
        sub_path = os.path.join(scene_path, sub)
        if not os.path.isdir(sub_path):  # ✅ Skip any non-folder
            continue

        image_dir = os.path.join(sub_path, "image")
        depth_out_dir = os.path.join(sub_path, "depth")

        os.makedirs(depth_out_dir, exist_ok=True)

        for img_file in sorted(os.listdir(image_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(image_dir, img_file)
            output_path = os.path.join(depth_out_dir, img_file)

            if os.path.exists(output_path):
                continue

            # Load and predict depth
            img = cv2.imread(img_path)
            img_input = transform({"image": img})["image"].to(device)
            with torch.no_grad():
                prediction = model.forward(img_input.unsqueeze(0))
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            cv2.imwrite(output_path, depth.astype(np.uint8))

