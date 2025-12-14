#!/usr/bin/env python3
import random
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

# Configuration
TEST_DIR = "/Users/sadik2/main_project/test"
CHECKPOINT_DIR = "./checkpoints_ddos"
CROP_SIZE = 512
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP_FILE = None


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

# Dataset
class DDOSDataset(Dataset):
    def __init__(self, pairs, crop_size=512, label_map=None):
        self.pairs = pairs
        self.crop_size = crop_size
        self.label_map = label_map or {}

    def __len__(self): return len(self.pairs)

    def _map_labels(self, mask):
        mapped = np.zeros_like(mask,dtype=np.int64)
        for k,v in self.label_map.items(): mapped[mask==k]=v
        return mapped

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg.ndim==3: seg = seg[:,:,0]
        h,w = img.shape[:2]
        if h!=self.crop_size or w!=self.crop_size:
            img = cv2.resize(img,(self.crop_size,self.crop_size),cv2.INTER_LINEAR)
            seg = cv2.resize(seg,(self.crop_size,self.crop_size),cv2.INTER_NEAREST)
        mask_mapped = self._map_labels(seg)
        return Image.fromarray(img), mask_mapped

# Load dataset
pairs = collect_pairs(TEST_DIR)
if len(pairs)==0:
    raise RuntimeError(f"No test data found in {TEST_DIR}")

if LABEL_MAP_FILE:
    with open(LABEL_MAP_FILE,"r") as f:
        label_map = json.load(f)
        label_map = {int(k): int(v) for k,v in label_map.items()}
else:
    sample = pairs[:min(200,len(pairs))]
    unique_vals = set()
    for _, seg_path in sample:
        m = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if m.ndim==3: m = m[:,:,0]
        unique_vals.update(np.unique(m).tolist())
    unique_vals = sorted(list(unique_vals))
    label_map = {v:i%NUM_CLASSES for i,v in enumerate(unique_vals)}

dataset = DDOSDataset(pairs, crop_size=CROP_SIZE, label_map=label_map)
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Load base and fine-tuned models
def latest_checkpoint(dir_):
    files = list(Path(dir_).glob("checkpoint_*.pth"))
    if not files: return None
    return sorted(files,key=lambda p:int(p.stem.split("_")[1]))[-1]

# Base SegFormer 
base_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
).to(DEVICE)
base_model.eval()

# Fine-tuned SegFormer
ft_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512", ignore_mismatched_sizes=True
)

# Replace classifier for NUM_CLASSES
in_ch = ft_model.decode_head.classifier.in_channels
ft_model.decode_head.classifier = nn.Conv2d(in_ch, NUM_CLASSES, 1)
nn.init.normal_(ft_model.decode_head.classifier.weight, 0, 0.02)
if ft_model.decode_head.classifier.bias is not None:
    nn.init.zeros_(ft_model.decode_head.classifier.bias)

ft_model.config.num_labels = NUM_CLASSES
ft_model.config.id2label = {i:str(i) for i in range(NUM_CLASSES)}
ft_model.config.label2id = {str(i):i for i in range(NUM_CLASSES)}

# Load checkpoint
ckpt = latest_checkpoint(CHECKPOINT_DIR)
if ckpt:
    state = torch.load(ckpt,map_location=DEVICE)
    ft_model.load_state_dict(state["model_state_dict"])
    print(f"Loaded fine-tuned checkpoint: {ckpt}")
else:
    print("No checkpoint found, using base model as fine-tuned")

ft_model.to(DEVICE)
ft_model.eval()

# Visualization of 5 random samples
samples = random.sample(range(len(dataset)), min(5,len(dataset)))

plt.figure(figsize=(24, 5*len(samples)))

def colorize(mask):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        colored[mask==cls] = np.array([int(255*(cls/NUM_CLASSES)), 0, int(255*(1-cls/NUM_CLASSES))])
    return colored

for idx, i in enumerate(samples):
    img, label = dataset[i]
    enc = processor(images=[img], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Base model prediction
        base_logits = base_model(**enc).logits
        base_pred = F.interpolate(base_logits, size=label.shape, mode="bilinear", align_corners=False)
        base_pred = base_pred.argmax(1).squeeze().cpu().numpy()

        # Fine-tuned model prediction
        ft_logits = ft_model(**enc).logits
        ft_pred = F.interpolate(ft_logits, size=label.shape, mode="bilinear", align_corners=False)
        ft_pred = ft_pred.argmax(1).squeeze().cpu().numpy()

    # Colorize
    base_colored = colorize(base_pred)
    ft_colored = colorize(ft_pred)
    label_colored = colorize(label)

    # Plot
    plt.subplot(len(samples), 4, idx*4 + 1)
    plt.imshow(np.array(img))
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(len(samples), 4, idx*4 + 2)
    plt.imshow(base_colored)
    plt.title("Base SegFormer")
    plt.axis("off")

    plt.subplot(len(samples), 4, idx*4 + 3)
    plt.imshow(ft_colored)
    plt.title("Fine-tuned SegFormer")
    plt.axis("off")

    plt.subplot(len(samples), 4, idx*4 + 4)
    plt.imshow(label_colored)
    plt.title("Ground Truth")
    plt.axis("off")

plt.tight_layout()
plt.show()
