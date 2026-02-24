import torch 
from PIL import Image 
import torchvision.transforms as transforms

#загрузи content изображение 
content_image = Image.open("test.jpg").convert("RGB")
print("Content загружено")

#загрузи style изображение (пока просто другое изображение)
style_image = Image.open("styles/vangogh1.jpg".convert("RGB"))
print("Style загружено")

#преобразуй в тензоры 
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]) 

content_tensor = transform(content_image).unsqueeze(0) # [1, 3, 256, 256]
style_tensor = transform(style_image).unsqueeze(0)

print("Форма content:", content_tensor.shape)
print("Форма style:", style_tensor.shape)

#пробуем просто смешать их (заглушка)
mixed = 0.5 * content_tensor + 0.5 * style_tensor

#сохрани результат 
from torchvision.utils import save_image 
save_image(mixed, "result.jpg")
print("Результат сохранён как result.jpg")