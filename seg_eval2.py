#!/usr/bin/env python3
import random
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import matplotlib.pyplot as plt

# Configuration
TEST_DIR = "/Users/sadik2/main_project/test" 
BATCH_SIZE = 1
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
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def _map_labels(self, mask):
        mapped = np.zeros_like(mask,dtype=np.int64)
        for k,v in self.label_map.items():
            mapped[mask==k] = v
        return mapped

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg.ndim==3:
            seg = seg[:,:,0]
        h,w = img.shape[:2]
        if h!=self.crop_size or w!=self.crop_size:
            img = cv2.resize(img,(self.crop_size,self.crop_size),cv2.INTER_LINEAR)
            seg = cv2.resize(seg,(self.crop_size,self.crop_size),cv2.INTER_NEAREST)
        mask_mapped = self._map_labels(seg)
        img_tensor = self.tf(Image.fromarray(img))
        return img_tensor, mask_mapped


# Load dataset
pairs = collect_pairs(TEST_DIR)
if len(pairs) == 0:
    raise RuntimeError(f"No test data found in {TEST_DIR}")

# Label map
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
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# Load DeepLabV3 model
model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)

in_ch = model.classifier[-1].in_channels
model.classifier[-1] = torch.nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1)
model.to(DEVICE)
model.eval()


# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)["out"]  
        outputs = F.interpolate(outputs, size=(CROP_SIZE, CROP_SIZE),
                                mode="bilinear", align_corners=False)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(masks.numpy())

all_preds = np.concatenate(all_preds, axis=0).ravel()
all_labels = np.concatenate(all_labels, axis=0).ravel()

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
iou = jaccard_score(all_labels, all_preds, average='weighted')

print(f"DeepLabV3 Evaluation on Test Set:")
print(f"Accuracy: {acc*100:.3f}%")
print(f"Weighted F1-score: {f1:.3f}")
print(f"Weighted IoU: {iou:.3f}")
