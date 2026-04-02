import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

# Проверка устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Загрузка модели
def load_stable_diffusion():
    """Загружает модель Stable Diffusion"""
    print("Loading Stable Diffusion model...")
    
    # Используем базовую модель (меньше потребление памяти)
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Отключаем для упрощения
    )
    
    pipe = pipe.to(device)
    print("✓ Stable Diffusion model loaded")
    
    return pipe

# Генерация изображения
def generate_image(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=7.5):
    """
    Генерирует изображение по текстовому описанию
    
    Args:
        prompt: текстовое описание желаемого изображения
        negative_prompt: что НЕ должно быть на изображении
        num_inference_steps: количество шагов диффузии
        guidance_scale: сила следования промпту (1-20)
    """
    # Загрузка модели
    pipe = load_stable_diffusion()
    
    # Генерация
    print(f"Generating image with prompt: '{prompt}'")
    print(f"Negative prompt: '{negative_prompt}'")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=512,
        width=512
    ).images[0]
    
    print("✓ Image generated successfully")
    
    return image

# Интеграция в сервис
def process_stable_diffusion(prompt, negative_prompt="", num_steps=20, guidance=7.5):
    """Функция для интеграции в сервис"""
    output_pil = generate_image(prompt, negative_prompt, num_steps, guidance)
    
    # Конвертация в байты
    output_bytes = io.BytesIO()
    output_pil.save(output_bytes, format='JPEG')
    output_bytes.seek(0)
    return output_bytes.getvalue()