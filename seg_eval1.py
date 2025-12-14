
import os, json, random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from tqdm import tqdm
import torch.nn.functional as F


TEST_DIR = "/Users/sadik2/main_project/test"      
CHECKPOINT_DIR =  "/Users/sadik2/main_project/checkpoints_ddos/checkpoint_1000.pth"# 
BATCH_SIZE = 2
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

    def __len__(self): return len(self.pairs)

    def _map_labels(self, mask):
        mapped = np.zeros_like(mask,dtype=np.int64)
        for k,v in self.label_map.items(): mapped[mask==k]=v
        return mapped

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path,cv2.IMREAD_UNCHANGED)
        if seg.ndim==3: seg = seg[:,:,0]
        h,w = img.shape[:2]
        if h!=self.crop_size or w!=self.crop_size:
            img = cv2.resize(img,(self.crop_size,self.crop_size),cv2.INTER_LINEAR)
            seg = cv2.resize(seg,(self.crop_size,self.crop_size),cv2.INTER_NEAREST)
        mask_mapped = self._map_labels(seg)
        return Image.fromarray(img), mask_mapped

def collate_batch(batch, processor):
    images = [b[0] for b in batch]
    masks = [b[1] for b in batch]
    enc = processor(images=images, return_tensors="pt")
    labels = torch.stack([torch.from_numpy(m).long() for m in masks], dim=0)
    return {"pixel_values": enc["pixel_values"], "labels": labels}

# Load dataset
pairs = collect_pairs(TEST_DIR)
if len(pairs)==0: raise RuntimeError(f"No test data found in {TEST_DIR}")

# Label map
if LABEL_MAP_FILE:
    with open(LABEL_MAP_FILE,"r") as f: label_map = json.load(f)
    label_map = {int(k): int(v) for k,v in label_map.items()}
else:
    sample = pairs[:min(100,len(pairs))]
    unique_vals = set()
    for _, seg_path in sample:
        m = cv2.imread(seg_path,cv2.IMREAD_UNCHANGED)
        if m.ndim==3: m=m[:,:,0]
        unique_vals.update(np.unique(m).tolist())
    unique_vals = sorted(list(unique_vals))
    label_map = {v:i%NUM_CLASSES for i,v in enumerate(unique_vals)}

dataset = DDOSDataset(pairs, crop_size=CROP_SIZE, label_map=label_map)
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b, processor))

# Load model
def latest_checkpoint(dir_):
    files = list(Path(dir_).glob("checkpoint_*.pth"))
    if not files: return None
    return sorted(files,key=lambda p:int(p.stem.split("_")[1]))[-1]

model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)

ckpt = latest_checkpoint(CHECKPOINT_DIR)
if ckpt:
    state = torch.load(ckpt,map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt}")
else:
    print("No checkpoint found, using base model")

model.to(DEVICE)
model.eval()

# Evaluation
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  

        # Resize logits to match label size
        logits_resized = F.interpolate(logits, size=labels.shape[1:], mode="bilinear", align_corners=False)
        preds = logits_resized.argmax(1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0).ravel()
all_labels = np.concatenate(all_labels, axis=0).ravel()

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
iou = jaccard_score(all_labels, all_preds, average="macro")

print(f"\nSegFormer Evaluation on Test Set:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"Mean IoU: {iou*100:.2f}%")
