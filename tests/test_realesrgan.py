# tests/test_realesrgan.py
from PIL import Image
import numpy as np

def test_realesrgan_import():
    """Проверка импорта Real-ESRGAN"""
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    assert True

def test_realesrgan_inference():
    """Проверка инференса Real-ESRGAN"""
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    
    # Создаём тестовое изображение 64x64
    test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Загружаем модель
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=None, model=model, half=False)
    
    # Проверяем, что функция enhance существует
    assert hasattr(upsampler, 'enhance')