import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

# ----------------------------
# User settings
# ----------------------------
CHECKPOINT_PATH = "./checkpoints/checkpoint_epoch6.pth"  
TEST_DIR = "/Users/sadik2/main_project/test"
NUM_SAMPLES = 5
IMAGE_SIZE = (256, 256)
NOISE_STD = 0.08
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------
class DenoiseDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), noise_std=0.08):
        self.image_paths = image_paths
        self.noise_std = noise_std
        self.w, self.h = image_size
        self.to_tensor = transforms.Compose([
            transforms.CenterCrop((self.h, self.w)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        clean = self.to_tensor(img)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean, p

# ----------------------------
# Collect images
# ----------------------------
def collect_image_paths(test_dir):
    test_dir = Path(test_dir)
    image_paths = []
    for top in ["neighbourhood", "park"]:
        base = test_dir / top
        if not base.exists():
            continue
        for img_dir in base.rglob("image"):
            for ext in ("*.png", "*.PNG"):
                for p in img_dir.glob(ext):
                    image_paths.append(str(p.resolve()))
    return sorted(image_paths)

# ----------------------------
# UNet model 
# ----------------------------
import torch.nn as nn

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
            identity = nn.Conv2d(identity.shape[1], out.shape[1], 1).to(out.device)(identity)
        out += identity
        out = self.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        self.pool = nn.MaxPool2d(2,2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        rev_features = features[::-1]
        up_in = features[-1]*2
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
            double = self.ups[i+1]
            out = trans(out)
            skip = skip_connections[up_idx]
            up_idx +=1
            if skip.shape[2:] != out.shape[2:]:
                h_min = (skip.shape[2]-out.shape[2])//2
                w_min = (skip.shape[3]-out.shape[3])//2
                skip = skip[:,:,h_min:h_min+out.shape[2], w_min:w_min+out.shape[3]]
            out = torch.cat([skip, out], dim=1)
            out = double(out)
        return torch.sigmoid(self.final_conv(out))

# ----------------------------
# Load checkpoint
# ----------------------------
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[demo] Loaded checkpoint from {checkpoint_path}")
    return model

# ----------------------------
# Display samples
# ----------------------------
def show_samples():
    image_paths = collect_image_paths(TEST_DIR)
    if len(image_paths) == 0:
        print("No images found in test folder.")
        return
    dataset = DenoiseDataset(image_paths, image_size=IMAGE_SIZE, noise_std=NOISE_STD)
    model = UNet().to(DEVICE)
    model = load_checkpoint(model, CHECKPOINT_PATH, DEVICE)
    sampled_idxs = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    for idx in sampled_idxs:
        noisy, clean, path = dataset[idx]
        noisy_batch = noisy.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(noisy_batch).cpu().squeeze(0)
        # Display
        fig, axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].imshow(clean.permute(1,2,0).numpy())
        axs[0].set_title("Original")
        axs[1].imshow(noisy.permute(1,2,0).numpy())
        axs[1].set_title("Noisy")
        axs[2].imshow(out.permute(1,2,0).numpy())
        axs[2].set_title("Cleaned")
        for ax in axs: ax.axis("off")
        plt.show()

if __name__ == "__main__":
    show_samples()
