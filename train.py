#!/usr/bin/env python3
"""
train_unet_denoiser.py

End-to-end training script for a U-Net denoising autoencoder with PSNR + SSIM metrics.
"""

import os
import csv
import math
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from pytorch_msssim import ssim  # <-- SSIM metric

# ----------------------------
# User settings (edit these)
# ----------------------------
TRAIN_DIR = "/Users/sadik2/main_project/val"  # <- set this to the parent 'train' folder
BATCH_SIZE = 2
NUM_WORKERS = 0
LR = 1e-4
NUM_EPOCHS = 5
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_FREQ_EPOCHS = 1
RESUME_CHECKPOINT = None
NOISE_STD = 0.08
IMAGE_SIZE = (256, 256)  # width, height (cropped)
SEED = 42
SAVE_METRICS_CSV = "metrics.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

torch.manual_seed(SEED)
random.seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ----------------------------
# Collect all .png images
# ----------------------------
def collect_image_paths(train_dir):
    train_dir = Path(train_dir)
    image_paths = []
    for top in ["neighbourhood", "park"]:
        base = train_dir / top
        if not base.exists():
            continue
        for img_dir in base.rglob("image"):
            for ext in ("*.png", "*.PNG"):
                for p in img_dir.glob(ext):
                    image_paths.append(str(p.resolve()))
    return sorted(set(image_paths))


# ----------------------------
# Dataset
# ----------------------------
class DenoiseDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), noise_std=0.08, transforms_aug=None):
        self.image_paths = image_paths
        self.noise_std = noise_std
        self.w, self.h = image_size
        self.to_tensor = transforms.Compose([
            transforms.CenterCrop((self.h, self.w)),  # crop instead of resize
            transforms.ToTensor()
        ])
        self.aug = transforms_aug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.aug:
            img = self.aug(img)
        clean = self.to_tensor(img)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean, p


# ----------------------------
# U-Net model
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        rev_features = features[::-1]
        up_in = features[-1] * 2
        for f in rev_features:
            self.ups.append(nn.ConvTranspose2d(up_in, f, 2, stride=2))
            self.ups.append(DoubleConv(up_in, f))
            up_in = f
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        out = x
        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
            out = self.pool(out)
        out = self.bottleneck(out)
        skip_connections = skip_connections[::-1]
        up_idx = 0
        for i in range(0, len(self.ups), 2):
            trans = self.ups[i]
            double = self.ups[i + 1]
            out = trans(out)
            skip = skip_connections[up_idx]
            up_idx += 1
            if skip.shape[2:] != out.shape[2:]:
                h_min = (skip.shape[2] - out.shape[2]) // 2
                w_min = (skip.shape[3] - out.shape[3]) // 2
                skip = skip[:, :, h_min:h_min + out.shape[2], w_min:w_min + out.shape[3]]
            out = torch.cat([skip, out], dim=1)
            out = double(out)
        return torch.sigmoid(self.final_conv(out))


# ----------------------------
# Metrics
# ----------------------------
def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10((max_val ** 2) / mse)


def ssim_metric(pred, target):
    return ssim(pred, target, data_range=1.0, size_average=True).item()


# ----------------------------
# Checkpoint utils
# ----------------------------
def save_checkpoint(state, checkpoint_dir, epoch):
    filename = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save(state, filename)
    latest = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(state, latest)
    print(f"[checkpoint] saved: {filename}")


def load_checkpoint(path, model, optimizer=None, device=DEVICE):
    print(f"[checkpoint] Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    return start_epoch


# ----------------------------
# Training loop
# ----------------------------
def train():
    image_paths = collect_image_paths(TRAIN_DIR)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under {TRAIN_DIR}.")
    print(f"Found {len(image_paths)} images.")

    dataset = DenoiseDataset(image_paths, image_size=IMAGE_SIZE, noise_std=NOISE_STD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(in_channels=3, out_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    start_epoch = 1
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        start_epoch = load_checkpoint(RESUME_CHECKPOINT, model, optimizer, DEVICE)
        print(f"Resuming from epoch {start_epoch}")

    metrics_path = os.path.join(CHECKPOINT_DIR, SAVE_METRICS_CSV)
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "epoch", "train_loss", "avg_psnr", "avg_ssim"])

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, epoch_psnr, epoch_ssim = 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")
        for noisy, clean, _ in pbar:
            noisy, clean = noisy.to(DEVICE).float(), clean.to(DEVICE).float()

            optimizer.zero_grad()
            out = model(noisy)
            loss = mse_loss(out, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            b_loss = loss.item()
            b_psnr = psnr(out.detach().cpu(), clean.detach().cpu())
            b_ssim = ssim_metric(out, clean)

            epoch_loss += b_loss
            epoch_psnr += b_psnr
            epoch_ssim += b_ssim
            n_batches += 1
            pbar.set_postfix({"loss": f"{b_loss:.4f}", "psnr": f"{b_psnr:.2f}", "ssim": f"{b_ssim:.4f}"})

        avg_loss = epoch_loss / n_batches
        avg_psnr = epoch_psnr / n_batches
        avg_ssim = epoch_ssim / n_batches
        print(f"[epoch {epoch}] loss={avg_loss:.6f}, psnr={avg_psnr:.2f}, ssim={avg_ssim:.4f}")

        scheduler.step(avg_loss)

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.utcnow().isoformat(), epoch, f"{avg_loss:.6f}", f"{avg_psnr:.4f}", f"{avg_ssim:.4f}"])

        if epoch % CHECKPOINT_FREQ_EPOCHS == 0 or epoch == NUM_EPOCHS:
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {"train_dir": TRAIN_DIR, "image_size": IMAGE_SIZE, "noise_std": NOISE_STD},
            }
            save_checkpoint(state, CHECKPOINT_DIR, epoch)

    print("Training finished.")


if __name__ == "__main__":
    train()
