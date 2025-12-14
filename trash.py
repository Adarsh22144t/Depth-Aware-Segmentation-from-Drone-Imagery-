#!/usr/bin/env python3
"""
segformer_eval_b0.py

Evaluate SegFormer-B0 on DDOS dataset and visualize predictions vs ground truth.
"""

import argparse, json, random
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--checkpoint_dir", type=str, required=True)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--crop_size", type=int, default=512)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--label_map", type=str, default=None)
parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
args = parser.parse_args()

TRAIN_DIR = Path(args.train_dir)
CHECKPOINT_DIR = Path(args.checkpoint_dir)
device = torch.device(args.device)

# -------------------------
# Collect (image, mask) pairs
# -------------------------
def collect_pairs(root):
    root = Path(root)
    pairs = []
    for top in ["neighbourhood", "park"]:
        base = root / top
        if not base.exists(): continue
        for img_dir in base.rglob("image"):
            seg_dir = img_dir.parent / "segmentation"
            if not seg_dir.exists(): continue
            for ext in ("*.png","*.PNG","*.jpg","*.jpeg"):
                for p in img_dir.glob(ext):
                    seg_p = seg_dir / p.name
                    if seg_p.exists(): pairs.append((str(p), str(seg_p)))
    return sorted(pairs)

pairs = collect_pairs(TRAIN_DIR)
if len(pairs) == 0:
    raise RuntimeError(f"No image/mask pairs under {TRAIN_DIR}")
print(f"Found {len(pairs)} examples")

# -------------------------
# Label mapping
# -------------------------
if args.label_map:
    with open(args.label_map, "r") as f:
        label_map = json.load(f)
    label_map = {int(k): int(v) for k,v in label_map.items()}
    print("Loaded label map:", label_map)
else:
    # infer from sample
    sample = pairs[:min(200, len(pairs))]
    unique_vals = set()
    for _, seg_path in sample:
        m = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if m.ndim == 3: m = m[:,:,0]
        unique_vals.update(np.unique(m).tolist())
    unique_vals = sorted(list(unique_vals))
    print("Found values:", unique_vals)
    label_map = {v:i%args.num_classes for i,v in enumerate(unique_vals)}
    pprint(label_map)

# -------------------------
# Dataset
# -------------------------
class DDOSDataset(Dataset):
    def __init__(self, pairs, crop_size=512, label_map=None):
        self.pairs = pairs
        self.crop_size = crop_size
        self.label_map = label_map or {}

    def __len__(self): return len(self.pairs)

    def _map_labels(self, mask): 
        mapped = np.zeros_like(mask,dtype=np.int64)
        for k,v in self.label_map.items():
            mapped[mask==k]=v
        return mapped

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(img_path,cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg.ndim==3: seg = seg[:,:,0]
        mask_mapped = self._map_labels(seg)
        return Image.fromarray(img), Image.fromarray(mask_mapped.astype(np.uint8))

dataset = DDOSDataset(pairs, crop_size=args.crop_size, label_map=label_map)

# -------------------------
# Load model
# -------------------------
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)

# Replace classifier
in_ch = model.decode_head.classifier.in_channels
model.decode_head.classifier = torch.nn.Conv2d(in_ch, args.num_classes, 1)
model.to(device)
model.eval()

# Load latest checkpoint
def latest_checkpoint(dir_):
    files = list(Path(dir_).glob("checkpoint_*.pth"))
    if not files: return None
    return sorted(files,key=lambda p:int(p.stem.split("_")[1]))[-1]

latest = latest_checkpoint(CHECKPOINT_DIR)
if latest:
    ck = torch.load(latest, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    print(f"Loaded checkpoint: {latest}")

# Processor
processor = SegformerImageProcessor.from_pretrained(model_name)

# -------------------------
# Colors for visualization
# -------------------------
colors = np.array([
    [0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],
    [255,0,255],[0,255,255],[128,128,128],[128,0,0],[0,128,0]
], dtype=np.uint8)

# -------------------------
# Show 5 random predictions
# -------------------------
for _ in range(args.num_samples):
    img, mask = random.choice(dataset)
    enc = processor(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        pred = out.logits.argmax(1).squeeze().cpu().numpy()
    
    # Resize pred to original size
    pred_resized = cv2.resize(pred.astype(np.uint8), (img.width, img.height), interpolation=cv2.INTER_NEAREST)
    mask_np = np.array(mask)

    # Map class indices to colors
    pred_color = colors[pred_resized]
    mask_color = colors[mask_np]

    # Concatenate original image, prediction, and ground truth for display
    orig_np = np.array(img)
    combined = np.concatenate([orig_np, pred_color, mask_color], axis=1)
    cv2.imshow("Original | Prediction | Ground Truth", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

cv2.destroyAllWindows()
