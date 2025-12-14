
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
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import matplotlib.pyplot as plt
import torch.nn as nn


# Configuration
TEST_DIR = "/Users/sadik2/main_project/test"  # Test dataset
BATCH_SIZE = 1
CROP_SIZE = 512
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP_FILE = None


# Dataset
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

class DDOSDataset(Dataset):
    def __init__(self, pairs, crop_size=512, label_map=None):
        self.pairs = pairs
        self.crop_size = crop_size
        self.label_map = label_map or {}
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.pairs)
    def _map_labels(self, mask):
        mapped = np.zeros_like(mask,dtype=np.int64)
        for k,v in self.label_map.items(): mapped[mask==k] = v
        return mapped
    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg.ndim==3: seg=seg[:,:,0]
        h,w = img.shape[:2]
        if h!=self.crop_size or w!=self.crop_size:
            img = cv2.resize(img,(self.crop_size,self.crop_size),cv2.INTER_LINEAR)
            seg = cv2.resize(seg,(self.crop_size,self.crop_size),cv2.INTER_NEAREST)
        mask_mapped = self._map_labels(seg)
        img_tensor = self.tf(Image.fromarray(img))
        return img_tensor, mask_mapped, np.array(img)


# Load dataset
pairs = collect_pairs(TEST_DIR)
if len(pairs)==0: raise RuntimeError(f"No test data found in {TEST_DIR}")

if LABEL_MAP_FILE:
    with open(LABEL_MAP_FILE,"r") as f:
        label_map = json.load(f)
        label_map = {int(k): int(v) for k,v in label_map.items()}
else:
    sample = pairs[:min(200,len(pairs))]
    unique_vals = set()
    for _, seg_path in sample:
        m = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if m.ndim==3: m=m[:,:,0]
        unique_vals.update(np.unique(m).tolist())
    unique_vals = sorted(list(unique_vals))
    label_map = {v:i%NUM_CLASSES for i,v in enumerate(unique_vals)}

dataset = DDOSDataset(pairs, crop_size=CROP_SIZE, label_map=label_map)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# Load models
# Base SegFormer
segformer_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_model_name, ignore_mismatched_sizes=True)
in_ch = segformer.decode_head.classifier.in_channels
segformer.decode_head.classifier = nn.Conv2d(in_ch, NUM_CLASSES, 1)
segformer.config.num_labels = NUM_CLASSES
segformer.config.id2label = {i:str(i) for i in range(NUM_CLASSES)}
segformer.config.label2id = {str(i):i for i in range(NUM_CLASSES)}
segformer.to(DEVICE)
segformer.eval()
processor = SegformerImageProcessor.from_pretrained(segformer_model_name)

# DeepLabV3
deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
in_ch = deeplab.classifier[-1].in_channels
deeplab.classifier[-1] = nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1)
deeplab.to(DEVICE)
deeplab.eval()

# Visualization
def colorize(mask):
    colored = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        colored[mask==cls] = np.array([int(255*(cls/NUM_CLASSES)),0,int(255*(1-cls/NUM_CLASSES))])
    return colored

samples = random.sample(range(len(dataset)), min(5,len(dataset)))
plt.figure(figsize=(20, 5*len(samples)))

for idx, i in enumerate(samples):
    img_tensor, label, img_np = dataset[i]
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # SegFormer prediction
    with torch.no_grad():
        enc = processor(images=[Image.fromarray(img_np)], return_tensors="pt").to(DEVICE)
        out_sf = segformer(**enc)
        logits_sf = F.interpolate(out_sf.logits, size=(CROP_SIZE,CROP_SIZE), mode="bilinear", align_corners=False)
        pred_sf = logits_sf.argmax(1).squeeze().cpu().numpy()

    # DeepLabV3 prediction
    with torch.no_grad():
        out_dl = deeplab(img_tensor)["out"]
        out_dl = F.interpolate(out_dl, size=(CROP_SIZE,CROP_SIZE), mode="bilinear", align_corners=False)
        pred_dl = out_dl.argmax(1).squeeze().cpu().numpy()

    # Color maps
    pred_sf_col = colorize(pred_sf)
    pred_dl_col = colorize(pred_dl)
    label_col = colorize(label)

    # Plot
    plt.subplot(len(samples),4,idx*4+1)
    plt.imshow(img_np)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(len(samples),4,idx*4+2)
    plt.imshow(pred_sf_col)
    plt.title("SegFormer Prediction")
    plt.axis("off")

    plt.subplot(len(samples),4,idx*4+3)
    plt.imshow(pred_dl_col)
    plt.title("DeepLabV3 Prediction")
    plt.axis("off")

    plt.subplot(len(samples),4,idx*4+4)
    plt.imshow(label_col)
    plt.title("Ground Truth")
    plt.axis("off")

plt.tight_layout()
plt.show()
