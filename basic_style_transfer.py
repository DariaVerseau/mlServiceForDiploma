import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image 
import io

# Устройство 
device = torch.device("cpu") # пока без GPU

def load_image(image_bytes, size = 256):
    """Загружает изображение и преобразует в тензор"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

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

def run_style_transfer(content_bytes, style_bytes, num_steps=200):
    """Основная функция переноса стиля"""

    # Загрузка изображений 
    content_img = load_image(content_bytes, size=384)
    style_img = load_image(style_bytes, size=384)

    # Загрузка VGG19
    weights = VGG19_Weights.IMAGENET1K_V1
    cnn = models.vgg19(weights=weights).to(device).eval()

    # Модель с потерями 
    model = get_model_and_losses(cnn, content_img, style_img)
    model.requires_grad_(False)

    # Начальное изображение = content 
    input_img = content_img.clone().requires_grad_(True)

    # Оптимизатор
    optimizer = torch.optim.LBFGS([input_img])

    step = [0]
    while step[0] <= num_steps:
        def closure():
            optimizer.zero_grad()

            # Обрезаем значения до [0,1]
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

            # Веса потерь 
            style_score *= 10000  
            content_score *= 1

            loss = style_score + content_score
            loss.backward()

            step[0] += 1
            return loss
        
        optimizer.step(closure)

    # Финальный результат 
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

def process_image(content_bytes, style_bytes):
    """Функция для интеграции в сервис"""  
    output = run_style_transfer(content_bytes, style_bytes, num_steps=200)  

    # Конвертация обратно в байты
    output_image = output.cpu().squeeze(0)
    output_pil = transforms.ToPILImage()(output_image)

    output_bytes = io.BytesIO()
    output_pil.save(output_bytes, format='JPEG') 
    return output_bytes.getvalue()            
