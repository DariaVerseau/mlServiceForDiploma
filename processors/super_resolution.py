
# processors/super_resolution.py
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np
from PIL import Image
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Кэшируем модели для скорости
_models = {}

def get_upscaler(scale):
    if scale not in _models:
        if scale == 2:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = 'Real-ESRGAN\weights\RealESRGAN_x2plus.pth'
        elif scale == 4:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = 'Real-ESRGAN\weights\RealESRGAN_x4plus.pth'
        else:
            raise ValueError("Only scale=2 or 4 supported")

        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=170,  # оптимально для GTX 1650
            tile_pad=10,
            pre_pad=0,
            half=False,  # GTX 1650 не поддерживает TensorFloat32
            device=device
        )
        _models[scale] = upsampler
    return _models[scale]

def upscale_image(image_bytes, scale=4):
    # Загрузка изображения
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(img)

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
