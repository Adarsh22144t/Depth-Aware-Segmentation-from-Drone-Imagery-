
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import matplotlib.pyplot as plt
import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = "/Users/sadik2/main_project/test/neighbourhood/4/image"
LABEL_DIR = IMAGE_DIR  
NUM_SAMPLES = 5
NUM_CLASSES = 10
SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
SEG_CHECKPOINT = "./checkpoints_ddos/checkpoint_1000.pth"

# Class Colors
CLASS_COLORS = np.array([
    [0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],
    [255,0,255],[0,255,255],[128,128,0],[128,0,128],[0,128,128]
], dtype=np.uint8)

def mask_to_color(mask):
    return CLASS_COLORS[mask % len(CLASS_COLORS)]

# Load Models
# Fine-tuned model ( RGB+Depth)
ft_model = SegformerForSemanticSegmentation.from_pretrained(
    SEG_MODEL_NAME, ignore_mismatched_sizes=True
)
in_ch = ft_model.decode_head.classifier.in_channels
ft_model.decode_head.classifier = nn.Conv2d(in_ch, NUM_CLASSES, 1)
ckpt = torch.load(SEG_CHECKPOINT, map_location=DEVICE)
ft_model.load_state_dict(ckpt["model_state_dict"])
ft_model.to(DEVICE).eval()

# Base model (3-channel)
base_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_NAME)
base_model.to(DEVICE).eval()

# Processor
processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_NAME)

# Helpers
def overlay_mask(image_pil, mask):
    
    img_np = np.array(image_pil)
    h, w = img_np.shape[:2]

    mask_color = mask_to_color(mask)
    if mask_color.shape[0] != h or mask_color.shape[1] != w:
        mask_color = cv2.resize(mask_color, (w, h), interpolation=cv2.INTER_NEAREST)

    if mask_color.ndim == 2:
        mask_color = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2RGB)
    elif mask_color.shape[2] == 1:
        mask_color = np.repeat(mask_color, 3, axis=2)

    if img_np.dtype != np.uint8:
        img_np = (img_np*255).astype(np.uint8)
    if mask_color.dtype != np.uint8:
        mask_color = mask_color.astype(np.uint8)

    overlay = cv2.addWeighted(img_np, 0.7, mask_color, 0.3, 0)
    return overlay

def segment_model(image_pil, model):
    img_resized = image_pil.resize((512,512))
    inputs = processor(images=[img_resized], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    mask = torch.argmax(F.interpolate(logits, size=(image_pil.height, image_pil.width),
                                      mode="bilinear", align_corners=False), dim=1).squeeze().cpu().numpy()
    return mask

# Select random samples
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
selected_files = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

# Visualization
plt.figure(figsize=(24, 5*len(selected_files)))

for idx, file in enumerate(selected_files):
    img_path = os.path.join(IMAGE_DIR, file)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(file)[0]+".png")
    image = Image.open(img_path).convert("RGB")
    gt_mask = np.array(Image.open(label_path).convert("L"), dtype=np.uint8)

    base_mask = segment_model(image, base_model)
    ft_mask = segment_model(image, ft_model)

    base_overlay = overlay_mask(image, base_mask)
    ft_overlay = overlay_mask(image, ft_mask)
    gt_overlay = overlay_mask(image, gt_mask)

    # Plotting
    plt.subplot(len(selected_files), 4, idx*4 + 1)
    plt.imshow(image); plt.title("Input Image"); plt.axis("off")
    plt.subplot(len(selected_files), 4, idx*4 + 2)
    plt.imshow(base_overlay); plt.title("3-channel Model"); plt.axis("off")
    plt.subplot(len(selected_files), 4, idx*4 + 3)
    plt.imshow(ft_overlay); plt.title("RGB+Depth Model"); plt.axis("off")
    plt.subplot(len(selected_files), 4, idx*4 + 4)
    plt.imshow(gt_overlay); plt.title("Ground Truth"); plt.axis("off")

plt.tight_layout()
plt.show()
