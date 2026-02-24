import torch
from PIL import Image 
import torchvision.transforms as transforms
from torchvision.utils import save_image 

# загрузка изображения 
image = Image.open("test.jpg").convert("RGB")
print("Изображение загружено:", image.size)

# преобразование в тензор [3, H, W]
transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor()
])
tensor = transform(image)
print("Форма тензора:", tensor.shape)

# создание стиля - просто другое изображение или случайный тензор
# ВариантА - то же изображение(заглушка)
style_tensor = tensor.clone()

# Вариант B - случайный тензор(эксперимент)
# style_tensor = torch.randn_like(tensor)

# просто смешивание (не перенос стиля, а базовая операция)
result_tensor = 0.7 * tensor + 0.3 * style_tensor

# сохранение результата 
save_image(result_tensor, "result_simple.jpg")
print("Результат сохранён: result_simple.jpg")