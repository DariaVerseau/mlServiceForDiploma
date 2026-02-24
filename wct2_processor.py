import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import io

# Устройство (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Преобразования для изображений
def image_loader(image_bytes):
    """Загружает изображение и преобразует его для VGG"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Размеры как в оригинальном туториале
    imsize = 256 if device == "cpu" else 512
    
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img):
    """Создаёт модель и функции потерь"""
    
    # Нормализация
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    # Слои для content и style
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    cnn = copy.deepcopy(cnn)
    
    model = nn.Sequential(normalization)
    
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)

    # Обрезаем модель после последнего слоя потерь
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model

# Вспомогательные классы
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

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

import copy

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, num_steps=100,
                       style_weight=1000000, content_weight=1):
    """Запускает перенос стиля"""
    
    model = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                       style_img, content_img)
    
    input_img = content_img.clone()
    
    model.requires_grad_(False)
    input_img.requires_grad_(True)
    
    optimizer = optim.LBFGS([input_img])
    
    step = 0
    while step <= num_steps:
        def closure():
            nonlocal step
            with torch.no_grad():
                input_img.clamp_(0, 1)
                
            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0
            
            for mod in model.modules():
                if isinstance(mod, ContentLoss):
                    content_score += mod.loss
                if isinstance(mod, StyleLoss):
                    style_score += mod.loss
                    
            style_score *= style_weight
            content_score *= content_weight
            
            loss = style_score + content_score
            loss.backward()
            
            step += 1
            return loss
            
        optimizer.step(closure)
        
    with torch.no_grad():
        input_img.clamp_(0, 1)
        
    return input_img

def process_image(content_bytes, style_name="vangogh"):
    """Основная функция обработки"""
    
    # Загружаем VGG19
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    # Параметры нормализации для VGG
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
    
    # Загружаем content изображение
    content_img = image_loader(content_bytes)
    
    # Для style используем пока content (заглушка)
    # Позже добавим разные стили
    style_img = content_img.clone()
    
    # Запускаем перенос стиля
    output = run_style_transfer(
        cnn, normalization_mean, normalization_std,
        content_img, style_img, num_steps=100
    )
    
    # Конвертируем обратно в байты
    output_image = output.cpu().squeeze(0)
    output_pil = transforms.ToPILImage()(output_image)
    
    output_bytes = io.BytesIO()
    output_pil.save(output_bytes, format='JPEG')
    return output_bytes.getvalue()