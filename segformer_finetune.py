

import argparse, json, math, random
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    get_cosine_schedule_with_warmup,
)

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
parser.add_argument("--checkpoint_every_batches", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--crop_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--label_map", type=str, default=None)
parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
args = parser.parse_args()

TRAIN_DIR = Path(args.train_dir)
CHECKPOINT_DIR = Path(args.checkpoint_dir)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)


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
if len(pairs)==0: raise RuntimeError(f"No image/mask pairs under {TRAIN_DIR}")
print(f"Found {len(pairs)} examples")

# Label mapping
if args.label_map:
    with open(args.label_map,"r") as f: label_map = json.load(f)
    label_map = {int(k): int(v) for k,v in label_map.items()}
    print("Loaded label map:", label_map)
else:
    print("Inferring label map...")
    sample = pairs[:min(200,len(pairs))]
    unique_vals = set()
    for _, seg_path in sample:
        m = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if m.ndim==3: m = m[:,:,0]
        unique_vals.update(np.unique(m).tolist())
    unique_vals = sorted(list(unique_vals))
    print("Found values:", unique_vals)
    label_map = {v:i%args.num_classes for i,v in enumerate(unique_vals)}
    pprint(label_map)

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

    def _random_crop(self,img,mask):
        H,W = img.shape[:2]
        ch,cw = min(self.crop_size,H), min(self.crop_size,W)
        y = np.random.randint(0,H-ch+1) if H>ch else 0
        x = np.random.randint(0,W-cw+1) if W>cw else 0
        img_c = img[y:y+ch,x:x+cw]; mask_c = mask[y:y+ch,x:x+cw]
        if img_c.shape[0]!=self.crop_size or img_c.shape[1]!=self.crop_size:
            img_c = cv2.resize(img_c,(self.crop_size,self.crop_size),cv2.INTER_LINEAR)
            mask_c = cv2.resize(mask_c,(self.crop_size,self.crop_size),cv2.INTER_NEAREST)
        return img_c, mask_c

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(img_path,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path,cv2.IMREAD_UNCHANGED)
        if seg.ndim==3: seg=seg[:,:,0]
        img_c, mask_c = self._random_crop(img,seg)
        mask_mapped = self._map_labels(mask_c)
        return Image.fromarray(img_c), Image.fromarray(mask_mapped.astype(np.uint8))

# Collate
def collate_batch(batch, processor):
    images = [b[0] for b in batch]
    masks = [np.array(b[1],dtype=np.int64) for b in batch]
    enc = processor(images=images, return_tensors="pt")
    B,C,H,W = enc["pixel_values"].shape
    resized_masks = []
    for m in masks:
        m_res = cv2.resize(m.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
        resized_masks.append(torch.from_numpy(m_res).long())
    labels = torch.stack(resized_masks,dim=0)
    return {"pixel_values": enc["pixel_values"], "labels": labels}

# Model
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
print("Loading model:", model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)

#  classifier
in_ch = model.decode_head.classifier.in_channels
model.decode_head.classifier = nn.Conv2d(in_ch, args.num_classes, 1)
nn.init.normal_(model.decode_head.classifier.weight,0,0.02)
if model.decode_head.classifier.bias is not None: nn.init.zeros_(model.decode_head.classifier.bias)

model.config.num_labels=args.num_classes
model.config.id2label={i:str(i) for i in range(args.num_classes)}
model.config.label2id={str(i):i for i in range(args.num_classes)}
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
total_steps = math.ceil(len(pairs)/args.batch_size)*args.num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss(ignore_index=255)
processor = SegformerImageProcessor.from_pretrained(model_name)

# Resume checkpoint
def latest_checkpoint(dir_):
    files = list(Path(dir_).glob("checkpoint_*.pth"))
    if not files: return None
    return sorted(files,key=lambda p:int(p.stem.split("_")[1]))[-1]

start_epoch, global_step = 0,0
latest = latest_checkpoint(CHECKPOINT_DIR)
if latest:
    ck = torch.load(latest,map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    scheduler.load_state_dict(ck["scheduler_state_dict"])
    start_epoch = ck.get("epoch",0)
    global_step = ck.get("global_step",0)
    print(f"Resumed from epoch {start_epoch}, step {global_step}")


# Dataloader
dataset = DDOSDataset(pairs,crop_size=args.crop_size,label_map=label_map)
loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
                    collate_fn=lambda b: collate_batch(b,processor))

#training
print("Starting training...")
model.train()
for epoch in range(start_epoch, args.num_epochs):
    running_loss = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{args.num_epochs}")
    for batch_idx, batch in pbar:
        global_step += 1
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss_value = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss_value

        if global_step % 10 == 0:
            avg_loss = running_loss / 10.0
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'step': global_step})
            running_loss = 0.0

        # Save checkpoint 
        if global_step % args.checkpoint_every_batches == 0:
            ckpt_path = CHECKPOINT_DIR / f"checkpoint_{global_step}.pth"
            torch.save({
                "global_step": global_step,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # Display random sample 
            model.eval()
            with torch.no_grad():
                img_sample, mask_sample = random.choice(dataset)
                enc_sample = processor(images=[img_sample], return_tensors="pt").to(device)
                out = model(**enc_sample)
                pred = out.logits.argmax(1).squeeze().cpu().numpy()
                
                # Resize prediction 
                pred_resized = cv2.resize(pred.astype(np.uint8),
                                          (img_sample.width, img_sample.height),
                                          interpolation=cv2.INTER_NEAREST)
                
                overlay = np.array(img_sample).copy()
                overlay[pred_resized > 0] = [255, 0, 0]  # red overlay for non-bg
                cv2.imshow("Sample Prediction", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            model.train()

print("Training finished.")
