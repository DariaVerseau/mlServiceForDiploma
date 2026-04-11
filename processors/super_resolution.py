import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np
from PIL import Image
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Кэшируем модели для скорости
_models = {}

def get_model_path(scale: int) -> str:
    """Возвращает путь к файлу модели в зависимости от окружения"""
    
    # Пути для Docker-контейнера (приоритет)
    docker_paths = {
        2: "/app/weights/realesrgan/RealESRGAN_x2plus.pth",
        4: "/app/weights/realesrgan/RealESRGAN_x4plus.pth"
    }
    
    # Проверяем, работаем ли в Docker
    if os.path.exists(docker_paths[scale]):
        return docker_paths[scale]
    
    # Пути для локальной разработки (Windows)
    local_paths = {
        2: "Real-ESRGAN/weights/RealESRGAN_x2plus.pth",
        4: "Real-ESRGAN/weights/RealESRGAN_x4plus.pth"
    }
    
    return local_paths[scale]

def get_upscaler(scale: int):
    if scale not in _models:
        if scale not in [2, 4]:
            raise ValueError("Only scale=2 or 4 supported")
        
        model_path = get_model_path(scale)
        
        # Проверяем существование файла
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Available weights in /app/weights/realesrgan/: {os.listdir('/app/weights/realesrgan/') if os.path.exists('/app/weights/realesrgan/') else 'directory not found'}"
            )
        
        print(f"Loading model for scale {scale}x from {model_path}")
        
        # Создаём архитектуру сети
        if scale == 2:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        else:  # scale == 4
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=170,  # оптимально для GTX 1650
            tile_pad=10,
            pre_pad=0,
            half=False,  # GTX 1650 не поддерживает fp16
            device=device
        )
        _models[scale] = upsampler
        print(f"Model for scale {scale}x loaded successfully")
    
    return _models[scale]

def upscale_image(image_bytes: bytes, scale: int = 4) -> bytes:
    # Загрузка изображения
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(img)
    
    print(f"Processing image of size: {img_array.shape}, scale: {scale}x")

    # Выбор модели
    upsampler = get_upscaler(scale)

    # Обработка
    output, _ = upsampler.enhance(img_array, outscale=scale)

    # Конвертация обратно в байты
    output_img = Image.fromarray(output)
    output_bytes = io.BytesIO()
    output_img.save(output_bytes, format='JPEG', quality=95)
    output_bytes.seek(0)
    return output_bytes.getvalue()