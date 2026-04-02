import torch
import os
import sys
import io
import tempfile
from PIL import Image
import numpy as np

# Добавляем путь к репозиторию TeleStyle
sys.path.append('./AI-TeleStyle')

class TeleStyleTransfer:
    """TeleStyle для переноса стиля (2026) — официальная реализация"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TeleStyle model on {self.device}...")
        
        # Импорт модулей из репозитория TeleStyle
        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
        from huggingface_hub import hf_hub_download
        
        self.QwenImagePipeline = QwenImagePipeline
        self.ModelConfig = ModelConfig
        self.hf_hub_download = hf_hub_download
        
        # Загрузка моделей
        self._load_models()
        print("✓ TeleStyle model loaded successfully")
    
    def _load_models(self):
        """Загрузка весов модели через официальный пайплайн"""
        # Создаем пайплайн
        self.pipe = self.QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device="cuda" if self.device.type == "cuda" else "cpu",
            model_configs=[
                self.ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source='huggingface',
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"
                ),
                self.ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source='huggingface',
                    origin_file_pattern="text_encoder/model*.safetensors"
                ),
                self.ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source='huggingface',
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors"
                ),
            ],
            tokenizer_config=None,
            processor_config=self.ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2509",
                download_source='huggingface',
                origin_file_pattern="processor/"
            ),
        )
        
        # Скачиваем веса LoRA (если ещё не скачаны)
        weights_dir = "./AI-TeleStyle/weights"
        os.makedirs(weights_dir, exist_ok=True)
        
        # Веса для стиля
        telestyle_path = os.path.join(weights_dir, "diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors")
        if not os.path.exists(telestyle_path):
            print("Downloading TeleStyle weights (this may take a few minutes)...")
            telestyle_path = self.hf_hub_download(
                repo_id="Tele-AI/TeleStyle",
                filename="weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors",
                local_dir=weights_dir
            )
        
        # Веса для ускорения (4 шага)
        lightning_path = os.path.join(weights_dir, "diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
        if not os.path.exists(lightning_path):
            print("Downloading Lightning weights (this may take a few minutes)...")
            lightning_path = self.hf_hub_download(
                repo_id="Tele-AI/TeleStyle",
                filename="weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                local_dir=weights_dir
            )
        
        # Загрузка LoRA-весов
        print("Loading LoRA weights...")
        self.pipe.load_lora(self.pipe.dit, telestyle_path)
        self.pipe.load_lora(self.pipe.dit, lightning_path)
        print("✓ LoRA weights loaded")
    
    def transfer_style(self, content_bytes, style_bytes, num_inference_steps=4):
        """
        Применение стиля через официальный TeleStyle
        
        Args:
            content_bytes: байты контентного изображения
            style_bytes: байты стилевого изображения
            num_inference_steps: количество шагов диффузии (4 по умолчанию)
        
        Returns:
            PIL Image: стилизованное изображение
        """
        # Сохраняем байты во временные файлы (требуется для пайплайна)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as content_file:
            content_file.write(content_bytes)
            content_path = content_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as style_file:
            style_file.write(style_bytes)
            style_path = style_file.name
        
        try:
            # Загрузка изображений для определения размеров
            content_img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
            style_img = Image.open(io.BytesIO(style_bytes)).convert("RGB")
            original_w, original_h = content_img.size
            
            # Определение размера для обработки (максимум 1024 для GTX 1650)
            max_edge = 768  # Уменьшаем до 768 для безопасности на 4 ГБ GPU
            min_edge = min(original_w, original_h)
            scale = max_edge / min_edge if min_edge > max_edge else 1.0
            
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            # Промпт для переноса стиля (официальный из репозитория)
            prompt = 'Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'
            
            # Выполнение инференса
            print(f"Applying TeleStyle with {num_inference_steps} steps...")
            with torch.no_grad():
                result_image = self.pipe(
                    prompt,
                    edit_image=[
                        content_img.resize((new_w, new_h)),
                        style_img.resize((max_edge, max_edge))
                    ],
                    seed=42,
                    num_inference_steps=num_inference_steps,
                    height=new_h,
                    width=new_w,
                    edit_image_auto_resize=False,
                    cfg_scale=1.0
                )
            
            # Восстановление оригинального размера
            result_image = result_image.resize((original_w, original_h), Image.Resampling.LANCZOS)
            
            return result_image
        
        finally:
            # Очистка временных файлов
            for path in [content_path, style_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def process(self, content_bytes, style_bytes, num_steps=4):
        """Функция для интеграции в сервис"""
        output_pil = self.transfer_style(content_bytes, style_bytes, num_steps)
        
        # Конвертация в байты
        output_bytes = io.BytesIO()
        output_pil.save(output_bytes, format='JPEG', quality=95)
        output_bytes.seek(0)
        return output_bytes.getvalue()

# Глобальный экземпляр модели (для кэширования)
_telestyle_instance = None

def get_telestyle_model():
    """Получение или создание экземпляра модели"""
    global _telestyle_instance
    if _telestyle_instance is None:
        _telestyle_instance = TeleStyleTransfer()
    return _telestyle_instance

def process_telestyle(content_bytes, style_bytes, num_steps=4):
    """Функция для интеграции в сервис"""
    model = get_telestyle_model()
    return model.process(content_bytes, style_bytes, num_steps)