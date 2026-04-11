import io
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn
import os
from PIL import Image, ImageEnhance

#from processors.face_enhancement import enhance_faces первая модель, неплохая, но плохо справлятся с обработкой глаз, делает их не естественными  
#from processors.codeformer_enhancement import enhance_faces_codeformer #улучшенная модель 
from basic_style_transfer import process_image #пока использую базовый перенос стиля 
from processors.super_resolution import upscale_image
from processors.postprocess import enhance_photo_pil_cv2


import uuid
import sys
import subprocess
import tempfile
import shutil


app = FastAPI(title="ml-service")

# Создаём папку для результатов при запуске
os.makedirs("results", exist_ok=True)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-service"}

@app.post(
    "/process",
    summary="Применить художественный стиль к изображению",
    description="""
    Принимает изображение и название стиля, возвращает обработанное изображение.
    
    Поддерживаемые стили: vangogh, picasso, monet, monet2, erinHanson
    Максимальный размер файла: 10MB
    """,
    response_description="URL для скачивания результата"
)
async def process_style_transfer(
    image: UploadFile = File(
        ...,
        description="Исходное изображение в формате JPEG или PNG"
    ),
    style: str = Form(
        "vangogh",
        description="Художественный стиль для применения",
        pattern="^(vangogh|picasso|monet|monet2|erinHanson|sketch)$"
    )
):
    # Создаём папку results если не существует
    os.makedirs("results", exist_ok=True)
    
    # Проверка поддерживаемых стилей
    supported_styles = ["vangogh", "picasso", "monet", "monet2", "erinHanson", "sketch"]
    if style not in supported_styles:
        raise HTTPException(
            status_code=400, 
            detail=f"Style '{style}' not supported. Available: {', '.join(supported_styles)}"
        )
    
    # Читаем content изображение
    content_bytes = await image.read()
    
    # Загружаем соответствующее style изображение
    style_path = f"styles/{style}.jpg"
    if not os.path.exists(style_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Style image '{style}.jpg' not found in styles/ folder"
        )
    
    with open(style_path, "rb") as f:
        style_bytes = f.read()
    
    # Применяем перенос стиля
    try:
        result_bytes = process_image(content_bytes, style_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")
    
    # Сохраняем результат
    result_filename = f"result_{uuid.uuid4().hex}.jpg"
    with open(f"results/{result_filename}", "wb") as f:
        f.write(result_bytes)
    
    return {"result_url": f"/results/{result_filename}", "style": style}

# Пути
BASE_DIR = Path(__file__).parent
ADAIN_DIR = BASE_DIR / "pytorch-AdaIN"
ADAIN_SCRIPT = ADAIN_DIR / "test.py"
STYLE_DIR = ADAIN_DIR / "input" / "style"
MODELS_DIR = ADAIN_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
TEMP_DIR = BASE_DIR / "temp_content"

# Создаём папки
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Монтируем папку с результатами
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Пути к весам
decoder_weights = MODELS_DIR / "decoder.pth"
vgg_weights = MODELS_DIR / "vgg_normalised.pth"


@app.post(
    "/style_transfer_adain",
    summary="Применить художественный стиль к изображению",
    description="Использует AdaIN из официального репозитория pytorch-AdaIN",
    response_description="URL для скачивания результата"
)
async def style_transfer_adain(
    image: UploadFile = File(..., description="Изображение для стилизации"),
    style: str = Form("vangogh", description="Название художественного стиля"),
    alpha: float = Form(1.0, ge=0.0, le=1.0, description="Сила стилизации"),
    preserve_color: bool = Form(False, description="Сохранять цвет оригинала")
):
    """Перенос художественного стиля через вызов test.py из репозитория"""
    
    # === 1. Валидация стиля ===
    supported_styles = [
        "vangogh", "picasso", "monet", "monet2", "erinHanson",
        "antimonocromatismo", "asheville", "brushstrokes", "contrast_of_forms",
        "en_campo_gris", "goeritz", "impronte_d_artista", "la_muse",
        "mondrian_cropped", "picasso_seated_nude_hr", "picasso_self_portrait",
        "scene_de_rue", "sketch", "the_resevoir_at_poitiers", "trial",
        "woman_in_peasant_dress_cropped", "woman_in_peasant_dress",
        "woman_with_hat_matisse"
    ]
    
    if style not in supported_styles:
        raise HTTPException(400, f"Style '{style}' not supported")
    
    # === 2. Проверяем наличие файла стиля ===
    style_path = STYLE_DIR / f"{style}.jpg"
    if not style_path.exists():
        style_path = STYLE_DIR / f"{style}.png"
        if not style_path.exists():
            raise HTTPException(404, f"Style image '{style}' not found")
    
    # === 3. Проверяем наличие весов ===
    if not decoder_weights.exists():
        raise HTTPException(500, f"Decoder weights not found at {decoder_weights}")
    if not vgg_weights.exists():
        raise HTTPException(500, f"VGG weights not found at {vgg_weights}")
    
    # === 4. Сохраняем загруженное изображение ===
    content_bytes = await image.read()
    if len(content_bytes) == 0:
        raise HTTPException(400, "Empty image file")
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=TEMP_DIR) as tmp:
        tmp.write(content_bytes)
        content_path = tmp.name
    
    # === 5. Временный файл для результата (с именем, которое создаст test.py) ===
    # test.py создаёт файлы вида: {content_name}_stylized_{style_name}.jpg
    # Мы сохраним во временную папку, а потом переименуем
    temp_output_dir = TEMP_DIR / "adain_output"
    temp_output_dir.mkdir(exist_ok=True)
    
    # === 6. Вызываем test.py ===
    command = [
        "python", str(ADAIN_SCRIPT),
        "--content", content_path,
        "--style", str(style_path),
        "--output", str(temp_output_dir),  # Сохраняем во временную папку
        "--alpha", str(alpha),
        "--decoder", str(decoder_weights),
        "--vgg", str(vgg_weights),
    ]
    
    if preserve_color:
        command.append("--preserve_color")
    
    print(f"Running: {' '.join(command)}")
    
    try:
        # Запускаем процесс
        result = subprocess.run(
            command,
            cwd=str(ADAIN_DIR),
            capture_output=True,
            text=True,
            timeout=120,
            check=False
        )
        
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"AdaIN failed: {result.stderr}")
        
        # === 7. Находим созданный файл ===
        # test.py создаёт файл с именем: {basename}_stylized_{stylename}.jpg
        content_basename = Path(content_path).stem
        style_basename = style_path.stem
        expected_filename = f"{content_basename}_stylized_{style_basename}.jpg"
        temp_result_path = temp_output_dir / expected_filename
        
        # Если файл не найден, ищем любой jpg во временной папке
        if not temp_result_path.exists():
            jpg_files = list(temp_output_dir.glob("*.jpg"))
            if jpg_files:
                temp_result_path = jpg_files[0]
            else:
                raise RuntimeError("No output file found")
        
        print(f"Found temp result: {temp_result_path}")
        
        # === 8. Переименовываем в нужное имя ===
        final_filename = f"adain_{style}_{uuid.uuid4().hex}.jpg"
        final_path = RESULTS_DIR / final_filename
        
        shutil.move(str(temp_result_path), str(final_path))
        print(f"Moved to: {final_path}")
        
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Style transfer timeout (120 seconds)")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, f"Style transfer failed: {str(e)}")
    finally:
        # Удаляем временные файлы
        if os.path.exists(content_path):
            os.unlink(content_path)
        # Очищаем временную папку с результатами
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir, ignore_errors=True)
    
    return {
        "result_url": f"/results/{final_filename}",
        "style": style,
        "alpha": alpha,
        "preserve_color": preserve_color,
        "method": "AdaIN (official repo)",
        "status": "success"
    }

@app.get("/results/{filename}")
async def get_result(filename: str):
    """Скачать результат по имени файла"""
    file_path = f"results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.post(
    "/upscale",
    summary="Повысить качество изображения",
    description="Повышает разрешение изображения в 2x или 4x раза с помощью Real-ESRGAN",
    response_description="URL для скачивания улучшенного изображения"
)
async def upscale_endpoint(
    image: UploadFile = File(...),
    scale: int = Form(4)
):
    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
    
    # Чтение изображения
    image_bytes = await image.read()
    
    # Обработка
    try:
        result_bytes = upscale_image(image_bytes, scale=scale)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upscale failed: {str(e)}")
    
    # Сохранение результата
    os.makedirs("results", exist_ok=True)
    result_filename = f"upscale_{uuid.uuid4().hex}.jpg"
    with open(f"results/{result_filename}", "wb") as f:
        f.write(result_bytes)
    
    return {"result_url": f"/results/{result_filename}", "scale": scale}

@app.post(
    "/enhance",
    summary="Улучшение изображения: восстановление лиц и постобработка",
    description="""
    Обрабатывает изображение с помощью модели CodeFormer:
    
    - Восстанавливает детали лиц (глаза, кожа, текстуры)
    - Применяет постобработку: резкость, контраст, шумоподавление
    - Сохраняет в компактный JPEG
    
    Параметры:
    - fidelity_weight: 0.0–1.0 (0.6 рекомендуется)
    - postprocess: true/false (включить классическую постобработку)
    """,
    response_description="URL для скачивания обработанного изображения"
)
async def enhance_image(
    image: UploadFile = File(...),
    fidelity_weight: float = Form(0.6),
    postprocess: bool = Form(True)
):
    # Валидация
    if not (0.0 <= fidelity_weight <= 1.0):
        raise HTTPException(status_code=400, detail="fidelity_weight must be 0.0–1.0")
    
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG supported")

    input_name = f"upload_{uuid.uuid4().hex}{ext}"
    final_result_path = None

    with open(input_name, "wb") as f:
        f.write(await image.read())

    try:
        print("=== DEBUG START ===")
        print(f"Input file: {input_name}")
        print(f"fidelity_weight: {fidelity_weight}")
        print(f"postprocess: {postprocess}")
        
        # === УМЕНЬШАЕМ ВХОД ДО CODEFORMER ===
        max_input_width = 1000
        with Image.open(input_name) as img_orig:
            if img_orig.width > max_input_width:
                ratio = max_input_width / img_orig.width
                new_size = (max_input_width, int(img_orig.height * ratio))
                img_orig = img_orig.resize(new_size, Image.LANCZOS)
                img_orig.save(input_name)
                print(f"Input resized BEFORE CodeFormer: {img_orig.size}")
        
        # === ЗАПУСК CODEFORMER ===
        cmd = [
            sys.executable,
            "CodeFormer/inference_codeformer.py",
            "--input_path", input_name,
            "-w", str(fidelity_weight),
            "--bg_upsampler", "None"  # ← УБРАЛ --has_aligned
        ]
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=600)
        
        print(f"CodeFormer return code: {result.returncode}")
        if result.returncode != 0:
            print("CodeFormer ERROR:", result.stderr)
        
        # === ПОИСК РЕЗУЛЬТАТА В ПРАВИЛЬНОЙ ПАПКЕ ===
        img_path = input_name  # fallback: исходное изображение

        # Ищем в папке: results/test_img_{fidelity_weight}/final_results/
        expected_dir = f"results/test_img_{fidelity_weight}/final_results"
        print(f"Looking for results in: {expected_dir}")
        print(f"Directory exists: {os.path.exists(expected_dir)}")

        if os.path.exists(expected_dir):
            files_in_final = [
                f for f in os.listdir(expected_dir)
                if os.path.isfile(os.path.join(expected_dir, f))
            ]
            print(f"Files found: {files_in_final}")
            
            if files_in_final:
                # Берём самый свежий файл (самый новый по времени создания)
                latest_file = max(
                    files_in_final,
                    key=lambda f: os.path.getctime(os.path.join(expected_dir, f))
                )
                img_path = os.path.join(expected_dir, latest_file)
                print(f"✓ Using CodeFormer result: {img_path}")
            else:
                print("✗ No files in final_results/ - using input")
        else:
            print("✗ Expected directory not found - using input")
        
        # === ПОСТОБРАБОТКА ===
        print(f"Opening image from: {img_path}")
        
        with Image.open(img_path) as img:
            print(f"Image mode: {img.mode}, size: {img.size}")
            
            # Ограничиваем максимальную ширину до 2000 пикселей
            max_width = 2000
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                print(f"Resized to: {img.size}")
            
            # Конвертируем в RGB если нужно
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Применяем постобработку если включена
            if postprocess:
                print(">>> APPLYING POSTPROCESSING <<<")
                img = enhance_photo_pil_cv2(img)
                print("Postprocessing applied!")
            else:
                print(">>> SKIPPING POSTPROCESSING <<<")
            
            # Сохраняем в компактный JPEG
            postfix = f"_postprocess_w{fidelity_weight}" if postprocess else f"_w{fidelity_weight}"
            final_result_path = f"results/enhanced{postfix}_{uuid.uuid4().hex}.jpg"
            quality = 90 if postprocess else 95
            img.save(final_result_path, "JPEG", quality=quality, optimize=True)
            print(f"Saved to: {final_result_path}")
        
        print("=== DEBUG END ===")
        
        return {
            "result_url": f"/results/{os.path.basename(final_result_path)}",
            "postprocess": postprocess,
            "fidelity_weight": fidelity_weight
        }

    finally:
        # Удаляем ТОЛЬКО входной временный файл
        if input_name and os.path.exists(input_name):
            os.unlink(input_name)
    

@app.post(
    "/postprocess",
    summary="Классическая постобработка изображения",
    description="""
    Применяет классические методы обработки изображений:
    
    - Повышение резкости (Sharpness) 0.0–3.0
    - Повышение контраста (Contrast) 0.0–2.0
    - Коррекция яркости (Brightness) 0.0–2.0
    - Шумоподавление (Denoising) 
    
    Подходит для улучшения качества без использования нейросетей.
    """,
    response_description="URL для скачивания обработанного изображения"
)
async def postprocess_image(
    image: UploadFile = File(...),
    sharpness: float = Form(1.25),
    contrast: float = Form(1.12),
    brightness: float = Form(1.05),
    denoise: bool = Form(True)
):
    # Валидация
    if not (0.0 <= sharpness <= 3.0):
        raise HTTPException(status_code=400, detail="sharpness must be 0.0–3.0")
    if not (0.0 <= contrast <= 2.0):
        raise HTTPException(status_code=400, detail="contrast must be 0.0–2.0")
    if not (0.0 <= brightness <= 2.0):
        raise HTTPException(status_code=400, detail="brightness must be 0.0–2.0")
    
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG supported")

    input_name = f"upload_{uuid.uuid4().hex}{ext}"
    final_result_path = None

    with open(input_name, "wb") as f:
        f.write(await image.read()) 

    try:
        print("=== POSTPROCESS START ===")
        print(f"Input file: {input_name}")
        print(f"sharpness: {sharpness}, contrast: {contrast}, brightness: {brightness}, denoise: {denoise}")
        
        # Открываем изображение
        with Image.open(input_name) as img:
            print(f"Image mode: {img.mode}, size: {img.size}")
            
            # Конвертируем в RGB если нужно
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Применяем улучшения в указанном порядке
            if sharpness != 1.0:
                img = ImageEnhance.Sharpness(img).enhance(sharpness)
                print(f"Sharpness applied: {sharpness}")
            
            if contrast != 1.0:
                img = ImageEnhance.Contrast(img).enhance(contrast)
                print(f"Contrast applied: {contrast}")
            
            if brightness != 1.0:
                img = ImageEnhance.Brightness(img).enhance(brightness)
                print(f"Brightness applied: {brightness}")
            
            # Применяем шумоподавление через OpenCV
            if denoise:
                print("Applying denoising...")
                open_cv_image = np.array(img)[:, :, ::-1].copy()
                denoised = cv2.fastNlMeansDenoisingColored(
                    open_cv_image, None, h=3, hColor=3,
                    templateWindowSize=7, searchWindowSize=21
                )
                img = Image.fromarray(denoised[:, :, ::-1])
                print("Denoising applied!")
            
            
            # Сохраняем результат
            final_result_path = f"results/postprocessed_{uuid.uuid4().hex}.jpg"
            print(f"Attempting to save to: {final_result_path}")

            try:
                img.save(final_result_path, "JPEG", quality=90, optimize=True)
                print(f"✓ File saved successfully!")
                print(f"File exists after save: {os.path.exists(final_result_path)}")
                print(f"File size: {os.path.getsize(final_result_path)} bytes")
            except Exception as e:
                print(f"✗ Error saving file: {e}")
                raise
        
        print("=== POSTPROCESS END ===")
        
        return {
            "result_url": f"/results/{os.path.basename(final_result_path)}",
            "sharpness": sharpness,
            "contrast": contrast,
            "brightness": brightness,
            "denoise": denoise
        }

    finally:
        # Удаляем ВРЕМЕННЫЕ файлы, но НЕ финальный результат
        print(f"Finally block: final_result_path = {final_result_path}")
        
        # Удаляем только входной файл
        if input_name and os.path.exists(input_name):
            try:
                os.unlink(input_name)
                print(f"Deleted input file: {input_name}")
            except:
                pass
        
        # НЕ удаляем final_result_path здесь!

@app.post(
    "/colorize",
    summary="Раскраска старых фотографий",
    description="Преобразует чёрно-белые изображения в цветные с правильной обработкой для старых фото",
    response_description="URL для скачивания раскрашенного изображения"
)
async def colorize_image(image: UploadFile = File(...)):
    # Валидация
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG/BMP supported")

    input_name = f"upload_{uuid.uuid4().hex}{ext}"
    final_result_path = None

    with open(input_name, "wb") as f:
        f.write(await image.read())

    try:
        print("=== COLORIZE START ===")
        
        # Импорты
        import sys
        sys.path.append('./colorization')
        from colorizers import siggraph17
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import numpy as np
        import cv2
        
        # Загружаем модель
        colorizer = siggraph17(pretrained=True).eval()
        
        # Загружаем изображение и конвертируем в чистый грейскейл
        img = Image.open(input_name).convert('L')
        
        # Предобработка для старых фото
        img_array = np.array(img)
        
        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)
        
        # Удаление шума
        img_array = cv2.fastNlMeansDenoising(img_array, h=12)
        
        # Конвертация в RGB для модели
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Конвертация в Lab
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        img_l = img_lab[:, :, 0:1].astype(np.float32)
        
        # Нормализация L канала к [0, 100]
        img_l_norm = img_l / 100.0
        
        # Подготовка для модели
        img_l_rs = torch.from_numpy(img_l_norm).permute(2, 0, 1).float().unsqueeze(0)
        img_l_rs = F.interpolate(img_l_rs, size=(224, 224), mode='bilinear')
        
        # Цветизация
        with torch.no_grad():
            img_ab = colorizer(img_l_rs)
            img_ab = F.interpolate(img_ab, size=(img_lab.shape[0], img_lab.shape[1]), mode='bilinear')
        
        # Реконструкция с ПРАВИЛЬНЫМ масштабом
        img_lab_out = np.zeros_like(img_lab)
        img_lab_out[:, :, 0] = img_l[:, :, 0]
        img_lab_out[:, :, 1:] = img_ab[0].cpu().permute(1, 2, 0).numpy() * 100.0  # КРИТИЧЕСКИ ВАЖНО: *100
        
        # Конвертация в RGB
        img_rgb_out = cv2.cvtColor(np.uint8(img_lab_out), cv2.COLOR_Lab2RGB)
        
        # Смешивание с оригиналом для естественности
        img_original_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        colorized = cv2.addWeighted(img_rgb_out, 0.7, img_original_rgb, 0.3, 0)
        
        # Сохранение
        final_result_path = f"results/colorized_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(final_result_path, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
        
        print(f"✓ Colorized successfully! Saved to: {final_result_path}")
        print("=== COLORIZE END ===")
        
        return {"result_url": f"/results/{os.path.basename(final_result_path)}"}

    except Exception as e:
        print(f"✗ Error colorizing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Colorization failed: {str(e)}")
    
    finally:
        # Очистка временных файлов
        if os.path.exists(input_name):
            try:
                os.unlink(input_name)
            except:
                pass

if __name__ == "__main__":
    print("Сервер запущен! Открой в браузере:")
    print("http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)