import argparse, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk

IMAGE_SIZE = (256,256)
NOISE_STD_DEFAULT=0.08
VALID_EXTS = (".png",".jpg",".jpeg",".bmp")
DTYPE=torch.float32

def collect_image_paths(test_dir):
    return sorted(f for ext in VALID_EXTS for f in Path(test_dir).rglob(f"*{ext}"))

def preprocess(img, size):
    img = img.resize(size)
    return T.ToTensor()(img)

def mse_np(a,b): return float(np.mean((a-b)**2))
def ssim_np(a,b):
    try: return float(ssim_sk(a,b,channel_axis=-1,data_range=1.0))
    except TypeError: return float(ssim_sk(a,b,multichannel=True,data_range=1.0))
def psnr_np(a,b): return float(psnr_sk(a,b,data_range=1.0))

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super().__init__()
        layers = [nn.Conv2d(channels,64,3,padding=1), nn.ReLU(True)]
        for _ in range(num_of_layers-2):
            layers += [nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True)]
        layers.append(nn.Conv2d(64,channels,3,padding=1,bias=False))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return torch.clamp(x - self.net(x),0.0,1.0)

class TinyUNet(nn.Module):
    def __init__(self, in_ch=3,out_ch=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_ch,32,3,padding=1), nn.ReLU(),
                                 nn.Conv2d(32,64,3,padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(64,32,3,padding=1), nn.ReLU(),
                                 nn.Conv2d(32,out_ch,3,padding=1), nn.Sigmoid())
    def forward(self,x):
        x1 = self.enc(x)
        return self.dec(x1)

def evaluate_model(model,name,image_paths,device):
    print(f"\n=== {name} ===")
    model.to(device).eval()
    mse_list,ssim_list,psnr_list,time_list=[],[],[],[]
    for i,p in enumerate(image_paths,1):
        img = Image.open(p).convert("RGB")
        clean_t = preprocess(img, IMAGE_SIZE).to(device=device,dtype=DTYPE).unsqueeze(0)
        noise = torch.randn_like(clean_t)*NOISE_STD_DEFAULT
        noisy_t = torch.clamp(clean_t + noise,0.0,1.0)
        with torch.no_grad():
            t0=time.time()
            out = model(noisy_t)
            t1=time.time()
        out_np = out.squeeze(0).cpu().permute(1,2,0).numpy()
        clean_np = clean_t.squeeze(0).cpu().permute(1,2,0).numpy()
        mse_list.append(mse_np(clean_np,out_np))
        ssim_list.append(ssim_np(clean_np,out_np))
        psnr_list.append(psnr_np(clean_np,out_np))
        time_list.append(t1-t0)
        if i%50==0 or i==len(image_paths):
            print(f"{i}/{len(image_paths)} done")
    print(f"\n{name} Avg MSE:{np.mean(mse_list):.6f}, Avg SSIM:{np.mean(ssim_list):.4f}, Avg PSNR:{np.mean(psnr_list):.2f}, Avg time/image:{np.mean(time_list)*1000:.1f}ms")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_dir",required=True)
    parser.add_argument("--max_images", type=int, default=300, help="Maximum number of images to evaluate")
    args=parser.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths=collect_image_paths(args.test_dir)
    
    # Limit to first N images
    image_paths = image_paths[:args.max_images]
    
    print(f"Found {len(image_paths)} images. Evaluating {len(image_paths)} images only.")

    evaluate_model(DnCNN(),"DnCNN",image_paths,device)
    evaluate_model(TinyUNet(),"TinyUNet",image_paths,device)

