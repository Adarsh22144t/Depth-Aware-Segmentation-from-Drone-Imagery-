# Depth-Aware Segmentation from Drone Imagery

## 📖 Overview
This project presents a **depth-aware semantic segmentation framework** for aerial images captured by drones. Traditional RGB-based segmentation struggles with **occlusions, shadows, elevation variations, and overlapping structures**, which are common in drone imagery. To address this, we integrate **monocular depth estimation** with **transformer-based semantic segmentation**, enabling the model to leverage both **visual appearance (RGB)** and **geometric depth cues**.

The framework is designed to be **accurate, robust, and computationally efficient**, making it suitable for real-world UAV applications such as **navigation, obstacle detection, urban mapping, and environmental monitoring**.

---

## 🚀 Key Contributions
- A complete **depth-aware segmentation pipeline** combining denoising, depth estimation, and semantic segmentation.
- Use of **DPT Hybrid (MiDaS 3.0)** for high-quality monocular depth estimation.
- Extension of **SegFormer** to accept **RGB + Depth (4-channel input)**.
- **Attention-based depth fusion** to dynamically balance RGB and depth features.
- Significant improvement in segmentation accuracy over RGB-only models.

---

## 🧠 Motivation
Drone imagery lacks explicit depth information, making it difficult to distinguish:
- Flat vs elevated structures  
- Overlapping buildings and vegetation  
- Objects under shadows or occlusions  

Depth provides **geometric context**, which improves:
- Boundary clarity  
- Object separation  
- Scene understanding in complex aerial environments  

---

## 🏗️ System Architecture

### 🔁 Overall Pipeline

<img width="544" height="486" alt="image" src="https://github.com/user-attachments/assets/92fbd71a-e835-4bfb-be37-d9844f107852" />


---

## 🧹 Denoising Module
Drone images often contain noise due to vibration, altitude, and weather conditions.

### Models Evaluated
- **DnCNN** – Strong noise removal but slight over-smoothing  
- **Tiny DnCNN** – Faster but reduced quality  
- **Custom U-Net** – Best balance of noise removal and detail preservation  

✅ **Selected Model:** Custom U-Net

<img width="768" height="412" alt="image" src="https://github.com/user-attachments/assets/143f135d-a435-45ea-9473-22b1e910beb2" />


---

## 📐 Depth Estimation Module
Depth estimation is performed using **monocular RGB images**, removing the need for LiDAR or stereo cameras.

### Models Evaluated
- **DPT Small** – Lightweight and fast  
- **DPT Hybrid (MiDaS 3.0)** – CNN + Transformer hybrid  

✅ **Selected Model:** DPT Hybrid  
It produces smoother depth gradients, sharper boundaries, and lower error metrics.

<img width="846" height="321" alt="image" src="https://github.com/user-attachments/assets/2eaabf5a-1bcd-4d75-a5a3-90f707d491bd" />


---

## 🧩 Segmentation Module
Semantic segmentation assigns a class label to every pixel (buildings, roads, vegetation, etc.).

### Models Used
- **DeepLabV3** – Strong on large objects  
- **SegFormer** – Transformer-based, excellent for complex aerial scenes  

✅ **Selected Model:** SegFormer

<img width="722" height="286" alt="image" src="https://github.com/user-attachments/assets/0970887b-4420-40e9-be6b-61253c91ed3e" />



---

## 🔀 Depth-Aware Fusion Network
The core innovation of this project is **RGB–Depth fusion**.

### Fusion Strategy
- Input becomes **4-channel (RGB + Depth)**
- **Channel-attention mechanism** dynamically balances feature importance:
  - Depth emphasized in elevated or occluded regions
  - RGB emphasized in flat or texture-rich regions

This improves:
- Boundary precision  
- Object separation  
- Robustness under shadows and clutter  



---

## 📊 Results & Performance

### Segmentation Performance
| Model | Input | mIoU ↑ | F1 ↑ | Pixel Accuracy ↑ |
|------|------|------|------|----------------|
| SegFormer (RGB) | RGB | 87.23 | 89.74 | 94.32 |
| **Fusion SegFormer** | RGB + Depth | **90.68** | **92.51** | **96.14** |



---

## 🖼️ Qualitative Results
The depth-aware model demonstrates:
- Sharper object boundaries  
- Better separation of overlapping objects  
- Improved segmentation under shadows and occlusions  


---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Framework:** PyTorch  
- **Models:** DPT Hybrid, SegFormer, DeepLabV3, U-Net  
- **Dataset:** DDOS (Drone Depth and Obstacle Segmentation)

---

## 📌 Applications
- Autonomous drone navigation  
- Obstacle detection and avoidance  
- Urban and terrain mapping  
- Disaster management  
- Environmental monitoring  

---

## 🔮 Future Scope
- Real-time onboard UAV deployment  
- Multi-modal fusion (LiDAR, thermal, multispectral)  
- Lightweight edge-optimized models  
- Temporal (video-based) depth-aware segmentation  

---

## 📄 Reference
This repository is based on the research paper:  
**“Depth-Aware Segmentation from Drone Imagery”**  
International Journal of Research Publication and Reviews, 2025.

---

## 👤 Authors
- Sailaka Adarsh  
- Shaik Sadik  
- Vadigalla Karthik  
- Yenni Tarun  
- Chinnam John Paul  
