#!/usr/bin/env python3
"""
inference_unet_denoiser.py

Loads trained U-Net denoiser, runs inference on sample test images,
and displays Original, Noisy, and Cleaned versions.
"""

import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from UNet_train import UNet  # <-- import your lightweight UNet

# ----------------------------
# Settings
# ----------------------------
TEST_DIR = "/Users/sadik2/main_project/test"   # <- set your test directory
CHECKPOINT_PATH = "./checkpoints/latest_checkpoint.pth"
NUM_SAMPLES = 5
NOISE_STD = 0.08
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------


# Preprocessing
to_tensor = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()


# ----------------------------
# Load model
# ----------------------------
def load_model(checkpoint_path, device=DEVICE):
    model = UNet(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[checkpoint] Loaded model from {checkpoint_path}")
    return model


# ----------------------------
# Collect sample test images
# ----------------------------
def collect_test_images(test_dir, num_samples=5):
    image_paths = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg"):
        image_paths.extend(list(Path(test_dir).rglob(ext)))
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under {test_dir}")
    return random.sample(image_paths, min(num_samples, len(image_paths)))


# ----------------------------
# Inference + Visualization
# ----------------------------
def run_inference():
    from pathlib import Path
    model = load_model(CHECKPOINT_PATH)

    image_paths = collect_test_images(TEST_DIR, NUM_SAMPLES)

    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(12, 4 * NUM_SAMPLES))
    if NUM_SAMPLES == 1:
        axes = [axes]  # handle single row

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        clean = to_tensor(img).unsqueeze(0).to(DEVICE)

        # Add synthetic noise
        noise = torch.randn_like(clean) * NOISE_STD
        noisy = torch.clamp(clean + noise, 0.0, 1.0)

        # Run through model
        with torch.no_grad():
            denoised = model(noisy)

        # Convert to images
        clean_img = to_pil(clean.squeeze(0).cpu())
        noisy_img = to_pil(noisy.squeeze(0).cpu())
        denoised_img = to_pil(denoised.squeeze(0).cpu())

        # Plot
        axes[idx][0].imshow(clean_img)
        axes[idx][0].set_title("Original")
        axes[idx][0].axis("off")

        axes[idx][1].imshow(noisy_img)
        axes[idx][1].set_title("Noisy")
        axes[idx][1].axis("off")

        axes[idx][2].imshow(denoised_img)
        axes[idx][2].set_title("Cleaned")
        axes[idx][2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_inference()
