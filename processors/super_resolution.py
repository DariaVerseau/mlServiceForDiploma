import torch
import numpy as np
from PIL import Image
import io
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def get_optimal_scale_and_model(original_size, requested_scale=4):
    """Определяет оптимальный scale и модель в зависимости от исходного размера"""
    max_dim = max(original_size)
    
    # Ограничение максимального выходного размера (4K)
    MAX_OUTPUT_SIZE = 3840
    
    if max_dim <= 512:
        # Маленькие изображения - можно использовать запрошенный scale
        optimal_scale = min(requested_scale, MAX_OUTPUT_SIZE // max_dim)
        return optimal_scale, 'x2plus' if optimal_scale == 2 else 'x4plus'
    elif max_dim <= 1024:
        # Средние изображения - максимум scale=2
        optimal_scale = min(2, MAX_OUTPUT_SIZE // max_dim)
        return optimal_scale, 'x2plus'
    else:
        # Большие изображения - только улучшение качества без увеличения
        return 1, 'x2plus'

def process_large_image(image_array, upsampler, scale):
    """Обработка очень больших изображений по частям"""
    h, w = image_array.shape[:2]
    tile_size = 512
    overlap = 64
    
    output_h, output_w = h * scale, w * scale
    output_img = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image_array[y:y_end, x:x_end]
            
            try:
                enhanced_tile, _ = upsampler.enhance(tile, outscale=scale)
            except Exception as e:
                print(f"Ошибка плитки ({x}, {y}): {e}")
                enhanced_tile = np.array(Image.fromarray(tile).resize(
                    (tile.shape[1] * scale, tile.shape[0] * scale),
                    Image.Resampling.LANCZOS
                ))
            
            out_y, out_x = y * scale, x * scale
            out_y_end = out_y + enhanced_tile.shape[0]
            out_x_end = out_x + enhanced_tile.shape[1]
            output_img[out_y:out_y_end, out_x:out_x_end] = enhanced_tile
    
    return output_img

def upscale_image(image_bytes, scale=4):
    """Умная обработка изображений любого размера с оптимальным качеством"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {device}")
        
        input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_array = np.array(input_image)
        original_size = input_image.size
        
        # Определение оптимального scale
        optimal_scale, model_type = get_optimal_scale_and_model(original_size, scale)
        print(f"Исходный размер: {original_size}, Запрошенный scale: {scale}, Оптимальный scale: {optimal_scale}")
        
        # Выбор модели
        if model_type == 'x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        else:  # x2plus
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        
        # Обработка без увеличения (только улучшение качества)
        if optimal_scale == 1:
            print("Режим улучшения качества без изменения размера")
            # Для scale=1 используем x2plus модель с последующим уменьшением
            temp_upsampler = RealESRGANer(
                scale=2,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                device=device
            )
            temp_output, _ = temp_upsampler.enhance(input_array, outscale=2)
            # Уменьшаем обратно до оригинального размера
            output_img = np.array(Image.fromarray(temp_output).resize(
                original_size, Image.Resampling.LANCZOS
            ))
        else:
            # Обычная обработка с оптимальным scale
            max_size = max(input_array.shape[:2])
            
            if max_size <= 600:
                # Маленькие - максимальное качество
                print(f"Обработка целиком (размер: {max_size}px)")
                upsampler = RealESRGANer(
                    scale=optimal_scale,
                    model_path=model_path,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    device=device
                )
                output_img, _ = upsampler.enhance(input_array, outscale=optimal_scale)
                
            elif max_size <= 1200:
                # Средние - стандартное разбиение
                print(f"Стандартное разбиение (размер: {max_size}px)")
                upsampler = RealESRGANer(
                    scale=optimal_scale,
                    model_path=model_path,
                    model=model,
                    tile=512,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    device=device
                )
                output_img, _ = upsampler.enhance(input_array, outscale=optimal_scale)
                
            else:
                # Большие - ручное разбиение
                print(f"Ручное разбиение (размер: {max_size}px)")
                upsampler = RealESRGANer(
                    scale=optimal_scale,
                    model_path=model_path,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    device=device
                )
                output_img = process_large_image(input_array, upsampler, optimal_scale)
        
        output_bytes = io.BytesIO()
        Image.fromarray(output_img).save(output_bytes, format='JPEG', quality=95)
        return output_bytes.getvalue()
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return image_bytes