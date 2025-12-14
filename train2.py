#!/usr/bin/env python3
"""
train_denoiser_stage1.py

Tiny, modular training + evaluation script for Stage 1 (denoising).
- CPU-friendly tiny U-Net
- On-the-fly noisy data generation (Gaussian + optional motion blur)
- Checkpointing and resume support
- Baseline comparisons: OpenCV fastNlMeans + optional DnCNN from torch.hub
- Saves sample outputs & metrics.csv

Adapt paths and hyperparams via CLI args.
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from pytorch_msssim import ssim as ssim_fn

# Optional: OpenCV (for baseline)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def l2_distance(a, b):
    return ((a - b) ** 2).mean()

def psnr_torch(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10((max_val ** 2) / mse)

def ssim_torch(pred, target):
    with torch.no_grad():
        # pytorch_msssim expects NCHW in [0,1]
        val = ssim_fn(pred, target, data_range=1.0, size_average=True).item()
    return val

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

# ----------------------------
# Collect images (generalized)
# ----------------------------
def collect_image_paths(parent_dir, exts=("png","jpg","jpeg","PNG","JPG")):
    p = Path(parent_dir)
    if not p.exists():
        raise FileNotFoundError(f"{parent_dir} not found")
    imgs = []
    for e in exts:
        imgs.extend(list(p.rglob(f"*.{e}")))
    imgs = [str(x.resolve()) for x in sorted(set(imgs))]
    return imgs

# ----------------------------
# Dataset
# ----------------------------
class DenoiseDataset(Dataset):
    """
    - Loads full-size images (1280x720), performs random crop (e.g., 256x256).
    - Produces (noisy, clean) pairs where noisy is clean + synthetic noise.
    - Supports Gaussian noise and optional motion blur augmentation.
    """
    def __init__(self, image_paths, crop_size=(256,256), noise_std=0.08, motion_blur_prob=0.0, transforms_extra=None):
        self.paths = image_paths
        self.crop_w, self.crop_h = crop_size
        self.noise_std = float(noise_std)
        self.motion_blur_prob = float(motion_blur_prob)
        self.transforms_extra = transforms_extra

        # convert to tensors later
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def random_crop(self, pil_img):
        W, H = pil_img.size
        if W < self.crop_w or H < self.crop_h:
            # resize preserving aspect ratio
            pil_img = pil_img.resize((max(self.crop_w, W), max(self.crop_h, H)), Image.BILINEAR)
            W, H = pil_img.size
        left = random.randint(0, W - self.crop_w)
        top = random.randint(0, H - self.crop_h)
        return pil_img.crop((left, top, left + self.crop_w, top + self.crop_h))

    def apply_motion_blur(self, pil_img, degree=8, angle=0):
        # simple motion blur using PIL by convolving with a line kernel
        # degree: length of the blur
        # angle: angle in degrees
        # approximate using ImageFilter.GaussianBlur after rotate/back for speed
        try:
            img = pil_img.rotate(angle, resample=Image.BILINEAR)
            img = img.filter(ImageFilter.GaussianBlur(radius=degree/6.0))
            img = img.rotate(-angle, resample=Image.BILINEAR)
            return img
        except Exception:
            return pil_img

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.random_crop(img)

        if self.transforms_extra:
            img = self.transforms_extra(img)

        clean = self.to_tensor(img)  # [C,H,W] in [0,1]

        # optionally apply motion blur before adding noise
        if random.random() < self.motion_blur_prob:
            angle = random.uniform(-15, 15)
            degree = random.uniform(1, 4)
            img_blur = self.apply_motion_blur(img, degree=int(degree), angle=angle)
            clean = self.to_tensor(img_blur)

        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)

        return noisy, clean, p

# ----------------------------
# Tiny U-Net model (CPU friendly)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=32):
        super().__init__()
        # tiny encoder 3 levels
        self.d1 = DoubleConv(in_channels, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        # bottleneck
        self.b = DoubleConv(base*4, base*8)
        # decoder
        self.t2 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.u2 = DoubleConv(base*8, base*4)
        self.t1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.u1 = DoubleConv(base*4, base*2)
        # final conv
        self.final = nn.Conv2d(base*2, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.d1(x)
        x = self.p1(s1)
        s2 = self.d2(x)
        x = self.p2(s2)
        s3 = self.d3(x)
        x = self.b(s3)
        x = self.t2(x)
        # If sizes mismatch, center-crop skip to match
        if x.shape[2:] != s3.shape[2:]:
            s3 = center_crop_like(s3, x)
        x = torch.cat([s3, x], dim=1)
        x = self.u2(x)
        x = self.t1(x)
        if x.shape[2:] != s2.shape[2:]:
            s2 = center_crop_like(s2, x)
        x = torch.cat([s2, x], dim=1)
        x = self.u1(x)
        # combine with s1 via conv (no transpose to keep small)
        if x.shape[2:] != s1.shape[2:]:
            s1 = center_crop_like(s1, x)
        x = torch.cat([s1, x], dim=1)  # channels: base + base*2
        # reduce channels
        x = nn.Conv2d(x.shape[1], 64, kernel_size=3, padding=1).to(x.device)(x)
        x = nn.ReLU(inplace=True)(x)
        out = torch.sigmoid(self.final(x))
        return out

def center_crop_like(src, tgt):
    _, _, hs, ws = tgt.shape
    _, _, hs2, ws2 = src.shape
    top = (hs2 - hs) // 2
    left = (ws2 - ws) // 2
    return src[..., top:top+hs, left:left+ws]

# ----------------------------
# Checkpoint utilities
# ----------------------------
def save_checkpoint(state, ckpt_dir, epoch, keep_latest=True):
    ensure_dir(ckpt_dir)
    fname = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save(state, fname)
    if keep_latest:
        latest = os.path.join(ckpt_dir, "latest_checkpoint.pth")
        torch.save(state, latest)
    print(f"[checkpoint] saved epoch {epoch} -> {fname}")

def load_checkpoint_if_exists(ckpt_dir, model, optimizer=None, device='cpu'):
    latest = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    if os.path.exists(latest):
        print(f"[checkpoint] loading {latest}")
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model_state"])
        if optimizer is not None and "optimizer_state" in ck:
            optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck.get("epoch", 0) + 1
        return start_epoch
    return 1

# ----------------------------
# Baseline denoisers
# ----------------------------
def baseline_opencv_fastnlm(img_np_uint8):
    """
    img_np_uint8: HxWx3 uint8 BGR (OpenCV style)
    returns: denoised image as HxWx3 uint8 BGR
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV not installed")
    # fastNlMeansDenoisingColored parameters: adjust for speed/quality
    h = 10  # filter strength
    templateWindowSize = 7
    searchWindowSize = 21
    out = cv2.fastNlMeansDenoisingColored(img_np_uint8, None, h, h, templateWindowSize, searchWindowSize)
    return out

def baseline_dncnn_torchhub(img_tensor):
    """
    Try to load a DnCNN from torch.hub (if available). If not, raise Exception.
    Input: image tensor float [0,1], shape [C,H,W] or [1,C,H,W]
    Output: denoised tensor same shape
    """
    # attempted hub repo often used: 'cszn/DnCNN' or 'rongzhao/DnCNN' - may or may not be present.
    # We'll attempt a common path but wrap in try/except.
    try:
        # repo and model name may vary; this is best-effort and non-fatal if not found
        model = torch.hub.load('cszn/DnCNN', 'dncnn_50')  # may fail
        model.eval()
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
            if inp.device != next(model.parameters()).device:
                inp = inp.to(next(model.parameters()).device)
            out = model(inp)
            # DnCNN usually predicts noise, so denoised = inp - out
            den = inp - out
            den = torch.clamp(den, 0.0, 1.0)
            return den.squeeze(0).cpu()
    except Exception as e:
        raise RuntimeError(f"Could not load DnCNN from torch.hub: {e}")

# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.force_cpu) else "cpu")
        set_seed(cfg.seed)
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.checkpoint_dir)
        ensure_dir(cfg.samples_dir)

        self.image_paths = collect_image_paths(cfg.data_dir)
        if len(self.image_paths) == 0:
            raise RuntimeError("No images found in data_dir.")
        print(f"[data] found {len(self.image_paths)} images")

        self.dataset = DenoiseDataset(self.image_paths,
                                      crop_size=(cfg.crop_w, cfg.crop_h),
                                      noise_std=cfg.noise_std,
                                      motion_blur_prob=cfg.motion_blur_prob)

        self.loader = DataLoader(self.dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=cfg.num_workers,
                                 pin_memory=False)

        # model, optimizer, scheduler
        self.model = TinyUNet(in_channels=3, out_channels=3, base=cfg.base_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=0.5, patience=5, verbose=True)

        # resume if exists
        self.start_epoch = load_checkpoint_if_exists(cfg.checkpoint_dir, self.model, self.optimizer, device=self.device)
        print(f"[train] starting epoch: {self.start_epoch}")

        # metrics CSV
        self.metrics_csv = os.path.join(cfg.output_dir, "metrics.csv")
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_utc", "epoch", "train_loss", "avg_psnr", "avg_ssim"])

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        n_batches = 0

        pbar = tqdm(self.loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", unit="batch")
        for noisy, clean, paths in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(noisy)
            loss = nn.functional.mse_loss(out, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                out_cpu = out.detach().cpu()
                clean_cpu = clean.detach().cpu()
                batch_psnr = psnr_torch(out_cpu, clean_cpu)
                batch_ssim = ssim_torch(out_cpu, clean_cpu)

            running_loss += loss.item()
            running_psnr += batch_psnr
            running_ssim += batch_ssim
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.6f}", "psnr": f"{batch_psnr:.2f}", "ssim": f"{batch_ssim:.4f}"})

        avg_loss = running_loss / max(1, n_batches)
        avg_psnr = running_psnr / max(1, n_batches)
        avg_ssim = running_ssim / max(1, n_batches)

        # scheduler step
        self.scheduler.step(avg_loss)

        # save metrics
        with open(self.metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.utcnow().isoformat(), epoch, f"{avg_loss:.6f}", f"{avg_psnr:.4f}", f"{avg_ssim:.4f}"])

        print(f"[epoch {epoch}] loss={avg_loss:.6f}, psnr={avg_psnr:.2f}, ssim={avg_ssim:.4f}")

        return avg_loss

    def save_sample_outputs(self, epoch, num_samples=4):
        self.model.eval()
        ensure_dir(self.cfg.samples_dir)
        saved = 0
        with torch.no_grad():
            for noisy, clean, paths in self.loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                out = self.model(noisy)
                # save first num_samples from this batch
                for i in range(min(noisy.size(0), num_samples - saved)):
                    n = noisy[i].cpu()
                    c = clean[i].cpu()
                    o = out[i].cpu()
                    # stack for visualization: noisy | denoised | clean
                    vis = torch.cat([n, o, c], dim=2)  # concat horizontally if H,W same -> careful: we want H,W same
                    # convert to PIL
                    vis_np = (vis.numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
                    out_path = os.path.join(self.cfg.samples_dir, f"epoch{epoch}_sample{saved}.png")
                    Image.fromarray(vis_np).save(out_path)
                    saved += 1
                    if saved >= num_samples:
                        return

    def save_checkpoint(self, epoch):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": vars(self.cfg)
        }
        save_checkpoint(state, self.cfg.checkpoint_dir, epoch)

    def evaluate_baselines(self, num_images=8):
        """
        Run baseline denoisers (OpenCV & optional DnCNN) on a small set of validation images
        and compute avg PSNR/SSIM vs ground truth.
        """
        print("[eval] running baselines on a small sample")
        samples = []
        # take up to num_images images from dataset (first N)
        for i in range(min(num_images, len(self.dataset))):
            noisy, clean, p = self.dataset[i]
            samples.append((noisy, clean, p))

        results = {}

        # Our model
        self.model.eval()
        with torch.no_grad():
            psnrs, ssims = [], []
            for noisy, clean, _ in samples:
                noisy_t = noisy.to(self.device).unsqueeze(0)
                out = self.model(noisy_t).squeeze(0).cpu()
                psnrs.append(psnr_torch(out, clean))
                ssims.append(ssim_torch(out.unsqueeze(0), clean.unsqueeze(0)))
            results["tiny_unet"] = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}
        print(f"[eval] tiny_unet -> PSNR {results['tiny_unet']['psnr']:.3f}, SSIM {results['tiny_unet']['ssim']:.4f}")

        # OpenCV baseline if available
        if OPENCV_AVAILABLE:
            psnrs, ssims = [], []
            for noisy, clean, _ in samples:
                # convert noisy to uint8 BGR for OpenCV
                np_noisy = (noisy.numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
                bgr = cv2.cvtColor(np_noisy, cv2.COLOR_RGB2BGR)
                den_bgr = baseline_opencv_fastnlm(bgr)
                den_rgb = cv2.cvtColor(den_bgr, cv2.COLOR_BGR2RGB)
                den_t = torch.from_numpy(den_rgb.astype(np.float32) / 255.0).permute(2,0,1)
                psnrs.append(psnr_torch(den_t, clean))
                ssims.append(ssim_torch(den_t.unsqueeze(0), clean.unsqueeze(0)))
            results["opencv_fastnlm"] = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}
            print(f"[eval] opencv_fastnlm -> PSNR {results['opencv_fastnlm']['psnr']:.3f}, SSIM {results['opencv_fastnlm']['ssim']:.4f}")
        else:
            print("[eval] OpenCV not available; skipping fastNlMeans baseline")

        # DnCNN if available (best-effort)
        try:
            psnrs, ssims = [], []
            for noisy, clean, _ in samples:
                # run DnCNN hub model on CPU (this may be slow)
                den = baseline_dncnn_torchhub(noisy)
                psnrs.append(psnr_torch(den, clean))
                ssims.append(ssim_torch(den.unsqueeze(0), clean.unsqueeze(0)))
            results["dncnn_hub"] = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}
            print(f"[eval] dncnn_hub -> PSNR {results['dncnn_hub']['psnr']:.3f}, SSIM {results['dncnn_hub']['ssim']:.4f}")
        except Exception as e:
            print(f"[eval] DnCNN hub baseline not available: {e}")

        # save baseline results CSV
        csvp = os.path.join(self.cfg.output_dir, "baseline_results.csv")
        with open(csvp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "psnr", "ssim"])
            for k,v in results.items():
                writer.writerow([k, v["psnr"], v["ssim"]])
        print(f"[eval] saved baseline results -> {csvp}")

    def run(self):
        try:
            for epoch in range(self.start_epoch, self.cfg.epochs + 1):
                t0 = time.time()
                avg_loss = self.train_one_epoch(epoch)
                self.save_sample_outputs(epoch, num_samples=self.cfg.num_sample_outputs)
                self.save_checkpoint(epoch)
                t1 = time.time()
                print(f"[epoch {epoch}] took {t1 - t0:.1f}s")
        except KeyboardInterrupt:
            print("[train] interrupted by user -- saving checkpoint")
            # save a last checkpoint
            self.save_checkpoint(epoch)
            raise

        # final evaluation (baselines)
        self.evaluate_baselines()

# ----------------------------
# Config object
# ----------------------------
class Config:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        self.samples_dir = os.path.join(args.output_dir, "samples")
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.noise_std = args.noise_std
        self.crop_w = args.crop_w
        self.crop_h = args.crop_h
        self.motion_blur_prob = args.motion_blur_prob
        self.seed = args.seed
        self.base_channels = args.base_channels
        self.force_cpu = args.force_cpu
        self.num_sample_outputs = args.num_samples
        # other params can be added here

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Tiny UNet denoiser training (Stage 1)")
    p.add_argument("--data_dir", type=str, default="./data", help="parent folder with images (recursively searched)")
    p.add_argument("--output_dir", type=str, default="./denoise_output", help="where checkpoints, samples, metrics are stored")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--noise_std", type=float, default=0.08)
    p.add_argument("--crop_w", type=int, default=256)
    p.add_argument("--crop_h", type=int, default=256)
    p.add_argument("--motion_blur_prob", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=32, help="base channel width for tiny UNet")
    p.add_argument("--num_samples", type=int, default=4, help="how many sample images to save per epoch")
    p.add_argument("--force_cpu", action="store_true", help="force CPU even if CUDA available")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(args)
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.checkpoint_dir)
    ensure_dir(cfg.samples_dir)

    print("CONFIG")
    for k,v in vars(cfg).items():
        print(f"  {k}: {v}")

    trainer = Trainer(cfg)
    trainer.run()
    print("DONE")

if __name__ == "__main__":
    main()
