import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image 
import io

import warnings
# Отключаем предупреждения от torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Проверка GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ GPU недоступен, используется CPU")

def tv_loss(img):
    """Total Variation Loss для сглаживания результата"""
    h_tv = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()
    w_tv = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()
    return (h_tv + w_tv) * 1e-6

# Устройство 
device = torch.device("cpu") # пока без GPU

def load_image(image_bytes, size=384): #рекомендуется 256 или 384 для слабых/средних систем
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0).to(device)  
    return tensor

def gram_matrix(input):
    """Вычисляет Gram-матрицу для style loss"""
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input    
    
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()   
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input    

def get_model_and_losses(cnn, content_img, style_img):
    """Создает модель с функциями потерь"""
    cnn = cnn.features[:21].eval() # До conv4_2

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential()
    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace = False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            model.add_module(f'content_loss{i}', ContentLoss(target))

        if name in style_layers:
            target_feature = model(style_img).detach()
            model.add_module(f'style_loss_{i}', StyleLoss(target_feature))             

    return model

def run_style_transfer(content_bytes, style_bytes, num_steps=150):
    """Основная функция переноса стиля"""
    
    try: 
        # Загрузка изображений (оба одинакового размера для обучения)
        content_img = load_image(content_bytes, size=384)
        style_img = load_image(style_bytes, size=384)
        
        # Сохраняем оригинальный размер content для финального результата
        original_content = Image.open(io.BytesIO(content_bytes)).convert('RGB')
        original_size = original_content.size  # (width, height)
    
        # Загрузка VGG19 (загрузка модели на GPU)
        weights = VGG19_Weights.IMAGENET1K_V1
        cnn = models.vgg19(weights=weights).to(device).eval()  

        # Модель с потерями 
        model = get_model_and_losses(cnn, content_img, style_img)
        model.requires_grad_(False)

        # Начальное изображение на GPU
        input_img = content_img.clone().requires_grad_(True).to(device)

        # Оптимизатор
        optimizer = torch.optim.Adam([input_img], lr=0.01)

        step = [0]
        while step[0] <= num_steps:
            optimizer.zero_grad()
            
            with torch.no_grad():
                input_img.clamp_(0, 1)
            
            model(input_img)
            
            style_score = 0
            content_score = 0
            
            for name, layer in model.named_children():
                if isinstance(layer, ContentLoss):
                    content_score += layer.loss 
                if isinstance(layer, StyleLoss):
                    style_score += layer.loss
            
            style_score *= 10000  
            content_score *= 1
            tv_score = tv_loss(input_img)
            
            loss = style_score + content_score + tv_score
            loss.backward()
            
            optimizer.step()  # ← Без closure!
            
            step[0] += 1
            
            if step[0] % 50 == 0:
                print(f"Step {step[0]}, Loss: {loss.item():.4f}")

        # Финальный результат, перенос обратно на CPU для сохранения
        with torch.no_grad():
            result_tensor = input_img.clamp(0, 1).cpu().squeeze(0)  
            pil_image = transforms.ToPILImage()(result_tensor)
            # Масштабируем до оригинального размера
            pil_image = pil_image.resize(original_size, Image.Resampling.LANCZOS)
            return pil_image

    except RuntimeError as e:
        if "out of memory" in str(e):
            raise Exception("Недостаточно памяти. Попробуйте уменьшить размер изображения.")
        else:
            raise e

def process_image(content_bytes, style_bytes):
    """Функция для интеграции в сервис"""  
    output_pil = run_style_transfer(content_bytes, style_bytes, num_steps=150)  

    # Конвертация обратно в байты
    output_bytes = io.BytesIO()
    output_pil.save(output_bytes, format='JPEG') 
    output_bytes.seek(0)
    return output_bytes.getvalue()