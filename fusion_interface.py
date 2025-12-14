
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn as nn
import torch.nn.functional as F
import random

# UNet Denoising Model
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
    def __init__(self, in_channels=3, out_channels=3, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(in_channels,f))
            in_channels = f
        self.pool = nn.MaxPool2d(2,2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        rev_features = features[::-1]
        up_in = features[-1]*2
        for f in rev_features:
            self.ups.append(nn.ConvTranspose2d(up_in,f,2,stride=2))
            self.ups.append(DoubleConv(up_in,f))
            up_in = f
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self,x):
        skip_connections=[]
        out=x
        for down in self.downs:
            out=down(out)
            skip_connections.append(out)
            out=self.pool(out)
        out=self.bottleneck(out)
        skip_connections=skip_connections[::-1]
        up_idx=0
        for i in range(0,len(self.ups),2):
            trans = self.ups[i]
            double = self.ups[i+1]
            out = trans(out)
            skip = skip_connections[up_idx]
            up_idx += 1
            if skip.shape[2:] != out.shape[2:]:
                h_min = (skip.shape[2]-out.shape[2])//2
                w_min = (skip.shape[3]-out.shape[3])//2
                skip = skip[:,:,h_min:h_min+out.shape[2], w_min:w_min+out.shape[3]]
            out = torch.cat([skip,out],dim=1)
            out = double(out)
        return torch.sigmoid(self.final_conv(out))

def load_denoiser(checkpoint_path, device):
    model = UNet()
    ckpt = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model

# Depth Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS","DPT_Hybrid")
midas_transforms = torch.hub.load("intel-isl/MiDaS","transforms")
depth_transform = midas_transforms.small_transform
midas.to(device).eval()

def estimate_depth(image_pil):
    img_cv = np.array(image_pil.convert("RGB"))
    input_batch = depth_transform(img_cv).to(device)
    with torch.no_grad():
        pred = midas(input_batch)
        pred = F.interpolate(pred.unsqueeze(1), size=img_cv.shape[:2], mode="bilinear", align_corners=False)
        depth_map = pred.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max()-depth_map.min()+1e-6)
    return depth_map

# Segmentation Models
seg_checkpoint = "./checkpoints_ddos/checkpoint_1000.pth"  # 3-channel finetuned
num_classes = 10
seg_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

# Finetuned 3-channel SegFormer
seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_name, ignore_mismatched_sizes=True)
in_ch = seg_model.decode_head.classifier.in_channels
seg_model.decode_head.classifier = nn.Conv2d(in_ch, num_classes,1)
seg_model.to(device).eval()
ckpt = torch.load(seg_checkpoint,map_location=device)
seg_model.load_state_dict(ckpt["model_state_dict"])

seg_processor = SegformerImageProcessor.from_pretrained(seg_model_name)

# Pretrained SegFormer 
seg_model_pretrained = SegformerForSemanticSegmentation.from_pretrained(seg_model_name)
seg_model_pretrained.to(device).eval()

CLASS_COLORS = np.array([
    [0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],
    [255,0,255],[0,255,255],[128,128,0],[128,0,128],[0,128,128]
],dtype=np.uint8)

def mask_to_color(mask):
    return CLASS_COLORS[mask % len(CLASS_COLORS)]

def overlay_mask(image_pil, mask):
    img_np = np.array(image_pil)
    mask_color = mask_to_color(mask)
    if mask_color.shape[:2] != img_np.shape[:2]:
        mask_color = cv2.resize(mask_color, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_color.ndim == 2:
        mask_color = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(img_np, 0.7, mask_color, 0.3, 0)
    return overlay

# Initial segmentation 
def segment_image_display(image_pil):
    img_resized = image_pil.resize((512,512))
    inputs = seg_processor(images=img_resized, return_tensors="pt").to(device)
    with torch.no_grad():
        out = seg_model_pretrained(**inputs)
    mask = torch.argmax(out.logits.squeeze(),dim=0).cpu().numpy()
    mask_resized = cv2.resize(mask.astype(np.uint8),(image_pil.width,image_pil.height), interpolation=cv2.INTER_NEAREST)
    return mask_resized

# Fusion output
def fusion_segmentation_fake(image_pil, depth_map):
    img_resized = image_pil.resize((512,512))
    inputs = seg_processor(images=img_resized, return_tensors="pt").to(device)
    with torch.no_grad():
        out = seg_model(**inputs)
    mask = torch.argmax(out.logits.squeeze(),dim=0).cpu().numpy()
    mask_resized = cv2.resize(mask.astype(np.uint8),(image_pil.width,image_pil.height), interpolation=cv2.INTER_NEAREST)
    return mask_resized

# Streamlit App
st.title(" Depth Aware Segmentation : Denoising → Depth → Segmentation -> Fusion")
st.markdown("Upload an image to run denoising, depth estimation, and segmentation using custom model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    #  Noisy Input
    image_resized = image.resize((512,512))
    image_np = np.array(image_resized)/255.0
    noisy_image = np.clip(image_np + np.random.normal(0,0.05,image_np.shape),0,1)
    noisy_pil = Image.fromarray((noisy_image*255).astype(np.uint8))
    st.image(noisy_pil, caption="1. Noisy Input (512*512)", use_column_width=True)

    #  Denoised Image
    denoiser = load_denoiser("./checkpoints/checkpoint_epoch6.pth",device)
    input_tensor = transforms.ToTensor()(noisy_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_tensor = denoiser(input_tensor)
    denoised_img = denoised_tensor.squeeze().permute(1,2,0).cpu().numpy()
    denoised_img = np.clip(denoised_img,0,1)
    denoised_pil = Image.fromarray((denoised_img*255).astype(np.uint8))
    st.image(denoised_pil, caption="2. Denoised Image", use_column_width=True)

    # Depth Estimation
    st.subheader("3. Depth Estimation")
    depth_map = estimate_depth(denoised_pil)
    st.image(depth_map, caption="Depth Map", use_column_width=True)

    # Initial Segmentation 
    st.subheader("4. Segmentation")
    seg_mask = segment_image_display(denoised_pil)
    seg_mask_color = mask_to_color(seg_mask)
    st.image(seg_mask_color, caption="Segmentation Mask", use_column_width=True)
    overlay = overlay_mask(denoised_pil, seg_mask)
    st.image(overlay, caption="Segmentation Overlay", use_column_width=True)

    # Fusion Output 
    st.subheader("5. Fusion Output (RGB+Depth)")
    fusion_mask = fusion_segmentation_fake(denoised_pil, depth_map)
    fusion_color = mask_to_color(fusion_mask)
    overlay_fusion = overlay_mask(denoised_pil,fusion_mask)
    st.image(fusion_color, caption="Fusion Segmentation Mask", use_column_width=True)
    st.image(overlay_fusion, caption="Fusion Overlay", use_column_width=True)
