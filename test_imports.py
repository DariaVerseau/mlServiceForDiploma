import torch
import torchvision
from fastapi import FastAPI
import cv2
from PIL import Image

print("✅ Все зависимости работают!")
print(f"PyTorch: {torch.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")