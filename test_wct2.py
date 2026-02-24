import sys
sys.path.append('WCT2')

from transfer import load_model
import torch

# Укажи правильный путь к весам
checkpoint_path = "WCT2/model_checkpoints/wave_encoder_cat5_l4.pth"
device = torch.device('cpu')

try:
    model = load_model(checkpoint_path, device)
    print("✅ Модель WCT2 успешно загружена!")
except Exception as e:
    print(f"❌ Ошибка: {e}")