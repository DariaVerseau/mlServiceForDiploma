import cv2
import numpy as np
from PIL import Image, ImageEnhance

def enhance_photo_pil_cv2(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img = ImageEnhance.Sharpness(pil_img).enhance(2.0)   
    img = ImageEnhance.Contrast(img).enhance(1.5)        
    img = ImageEnhance.Brightness(img).enhance(1.1)
    open_cv_image = np.array(img)[:, :, ::-1].copy()
    denoised = cv2.fastNlMeansDenoisingColored(
        open_cv_image, None, h=5, hColor=5,
        templateWindowSize=7, searchWindowSize=21
    )
    return Image.fromarray(denoised[:, :, ::-1])