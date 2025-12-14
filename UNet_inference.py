import argparse
import os
import time
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

# metrics for SSIM/PSNR
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk


IMAGE_SIZE = (256, 256)  
NOISE_STD_DEFAULT = 0.08
VALID_EXTS = (".png", ".PNG", ".jpg", ".jpeg", ".bmp")
DTYPE = torch.float32


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if identity.shape[1] != out.shape[1]:
         
            conv1x1 = nn.Conv2d(identity.shape[1], out.shape[1], 1).to(out.device)
            identity = conv1x1(identity)
        out += identity
        out = self.relu(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down path
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up path
        rev_features = features[::-1]
        up_in = features[-1] * 2
        for f in rev_features:
            self.ups.append(nn.ConvTranspose2d(up_in, f, 2, stride=2))
            self.ups.append(DoubleConv(up_in, f))
            up_in = f
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

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
# Helpers
# ----------------------------
def collect_image_paths(test_dir):
    """
    First try same layout as training: /test/<neighbourhood|park>/**/image/*.png
    If that yields nothing, fallback to recursive search for images in test_dir.
    Returns list of string paths.
    """
    test_dir = Path(test_dir)
    image_paths = []


    for top in ["neighbourhood", "park"]:
        base = test_dir / top
        if not base.exists():
            continue
        for img_dir in base.rglob("image"):
            for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.bmp"):
                for p in img_dir.glob(ext):
                    image_paths.append(str(p.resolve()))

    image_paths = sorted(set(image_paths))
    if len(image_paths) > 0:
        return image_paths

    
    for ext in VALID_EXTS:
        for p in test_dir.rglob(f"*{ext}"):
            image_paths.append(str(p.resolve()))
    image_paths = sorted(set(image_paths))
    return image_paths


# ----------------------------
# Preprocess 
# ----------------------------
def preprocess_pil(img: Image.Image, size):
    w, h = size
    if img.width < w or img.height < h:
        img = img.resize((w, h), Image.BILINEAR)
    else:
        left = (img.width - w) // 2
        top = (img.height - h) // 2
        img = img.crop((left, top, left + w, top + h))
    to_tensor = T.ToTensor()
    return to_tensor(img)  


# ----------------------------
# Checkpoint loader
# ----------------------------
def load_model_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(ckpt, dict):
        
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
        
            sample_keys = list(ckpt.keys())
            if sample_keys and isinstance(sample_keys[0], str) and "." in sample_keys[0]:
                state = ckpt
            else:
                raise RuntimeError("Checkpoint dict doesn't contain recognizable model state. Keys: " + ", ".join(list(ckpt.keys())))
    else:
        raise RuntimeError("Checkpoint format not recognized")

    # load model
    model.load_state_dict(state)
    return model


# ----------------------------
# Metrics
# ----------------------------
def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    
    try:
        return float(ssim_sk(a, b, channel_axis=-1, data_range=1.0))
    except TypeError:
        
        return float(ssim_sk(a, b, multichannel=True, data_range=1.0))


def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(psnr_sk(a, b, data_range=1.0))


# ----------------------------
# Main evaluation function
# ----------------------------
def evaluate_all(test_dir, checkpoint_path, image_size=IMAGE_SIZE, noise_std=NOISE_STD_DEFAULT, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"[info] device: {device}")

    # build model 
    model = UNet().to(device)
    # load checkpoint
    print(f"[info] Loading checkpoint: {checkpoint_path}")
    load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    # model stats
    num_params = sum(p.numel() for p in model.parameters())
    approx_mb = num_params * 4 / (1024 ** 2)
    print(f"[info] Model params: {num_params:,} (~{approx_mb:.2f} MB)")

    # gather images
    image_paths = collect_image_paths(test_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under {test_dir}.")
    print(f"[info] Found {len(image_paths)} images for evaluation.")

    mse_list = []
    ssim_list = []
    psnr_list = []
    time_list = []
    skipped = 0

    for i, p in enumerate(image_paths, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[warn] Skipping unreadable file {p}: {e}")
            skipped += 1
            continue

        # preprocess
        clean_t = preprocess_pil(img, image_size).to(device=device, dtype=DTYPE)  # [C,H,W]
        # add synthetic noise like training
        noise = torch.randn_like(clean_t) * noise_std
        noisy_t = torch.clamp(clean_t + noise, 0.0, 1.0).unsqueeze(0)  # [1,C,H,W]

        # inference and timing
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out_t = model(noisy_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
        infer_time = t1 - t0
        time_list.append(infer_time)

        # convert to numpy 
        out_np = out_t.squeeze(0).cpu().permute(1, 2, 0).numpy()
        clean_np = clean_t.cpu().permute(1, 2, 0).numpy()
        out_np = np.clip(out_np, 0.0, 1.0)
        clean_np = np.clip(clean_np, 0.0, 1.0)

        # metrics
        mse_v = mse_np(clean_np, out_np)
        ssim_v = ssim_np(clean_np, out_np)
        psnr_v = psnr_np(clean_np, out_np)

        mse_list.append(mse_v)
        ssim_list.append(ssim_v)
        psnr_list.append(psnr_v)

        if (i % 50 == 0) or (i == len(image_paths)):
            print(f"[{i}/{len(image_paths)}] mse={mse_v:.6f} ssim={ssim_v:.4f} psnr={psnr_v:.2f} time={infer_time*1000:.1f} ms  {Path(p).name}")

    # summary
    n = len(mse_list)
    if n == 0:
        raise RuntimeError("No valid images were processed.")
    avg_mse = float(np.mean(mse_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_psnr = float(np.mean(psnr_list))
    avg_time = float(np.mean(time_list))

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Images evaluated: {n}  (skipped: {skipped})")
    print(f"Avg MSE : {avg_mse:.6f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg PSNR: {avg_psnr:.2f} dB")
    print(f"Avg inference time/image: {avg_time:.4f} s  ({avg_time*1000:.1f} ms)")
    print(f"Model params: {num_params:,} (~{approx_mb:.2f} MB)")
    print("==============================\n")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNet denoiser checkpoint on a test folder")
    parser.add_argument("--test_dir", required=True, help="Path to test folder (can be same layout used in training)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth saved by training")
    parser.add_argument("--image_size", type=int, nargs=2, default=IMAGE_SIZE, help="(w h) crop/resize size")
    parser.add_argument("--noise_std", type=float, default=NOISE_STD_DEFAULT, help="Gaussian noise std to add to clean images")
    args = parser.parse_args()

    evaluate_all(args.test_dir, args.checkpoint, image_size=tuple(args.image_size), noise_std=args.noise_std)
