import io
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
import uvicorn
import os
from PIL import Image, ImageEnhance

from basic_style_transfer import process_image #пока использую базовый перенос стиля 
from processors.super_resolution import upscale_image
from processors.postprocess import enhance_photo_pil_cv2

import uuid
import sys
import subprocess
import tempfile
import shutil

torch.set_num_threads(6)

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
    style_path = f"style/{style}.jpg"
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
    
    return Response(content=result_bytes, media_type="image/jpeg")

# Пути
BASE_DIR = Path(__file__).parent
ADAIN_DIR = BASE_DIR / "pytorch-AdaIN"
ADAIN_SCRIPT = ADAIN_DIR / "test.py"

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
    response_description="Обработанное изображение в формате JPEG"
)
async def style_transfer_adain(
    image: UploadFile = File(..., description="Изображение для стилизации"),
    style: str = Form("vangogh", description="Название художественного стиля"),
    alpha: float = Form(1.0, ge=0.0, le=1.0, description="Сила стилизации"),
    preserve_color: bool = Form(False, description="Сохранять цвет оригинала")
):
    """Перенос художественного стиля через вызов test.py из репозитория"""
    
    # === ЕДИНЫЙ ПУТЬ К СТИЛЯМ ===
    STYLES_DIR = Path("style")

    if not STYLES_DIR.exists():
        raise HTTPException(500, f"Styles directory not found: {STYLES_DIR}")

    # Автоматическое определение поддерживаемых стилей
    supported_styles = {
        f.stem: f 
        for f in STYLES_DIR.iterdir() 
        if f.suffix.lower() in ('.jpg', '.png') and f.is_file()
    }

    if style not in supported_styles:
        available = ", ".join(supported_styles.keys())
        raise HTTPException(400, f"Style '{style}' not supported. Available: {available}")

    style_path = supported_styles[style]  

    # === Проверяем наличие весов ===
    if not decoder_weights.exists():
        raise HTTPException(500, f"Decoder weights not found at {decoder_weights}")
    if not vgg_weights.exists():
        raise HTTPException(500, f"VGG weights not found at {vgg_weights}")

    # === Обработка изображения ===
    content_bytes = await image.read()
    if len(content_bytes) == 0:
        raise HTTPException(400, "Empty image file")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(content_bytes)
        content_path = tmp.name

    try:
        with tempfile.TemporaryDirectory() as temp_output_dir:
            command = [
                "python", str(ADAIN_SCRIPT),
                "--content", content_path,
                "--style", str(style_path.resolve()),
                "--output", temp_output_dir,
                "--alpha", str(alpha),
                "--decoder", str(decoder_weights),
                "--vgg", str(vgg_weights),
            ]
            if preserve_color:
                command.append("--preserve_color")

            result = subprocess.run(
                command,
                cwd=str(ADAIN_DIR),
                capture_output=True,
                text=True,
                timeout=120,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"AdaIN failed: {result.stderr}")

            output_files = list(Path(temp_output_dir).glob("*.jpg"))
            if not output_files:
                raise RuntimeError("No output file generated")

            with open(output_files[0], "rb") as f:
                result_bytes = f.read()

        return Response(content=result_bytes, media_type="image/jpeg")

    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Style transfer timeout (120 seconds)")
    except Exception as e:
        raise HTTPException(500, f"Style transfer failed: {str(e)}")
    finally:
        if os.path.exists(content_path):
            os.unlink(content_path)

@app.get("/results/{filename}")
async def get_result(filename: str):
    """Скачать результат по имени файла"""
    file_path = f"results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

from fastapi.responses import Response

@app.post(
    "/upscale",
    summary="Повысить качество изображения",
    description="Повышает разрешение изображения в 2x или 4x раза с помощью Real-ESRGAN",
    response_description="Улучшенное изображение в формате JPEG"
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
    
    # Возвращаем изображение напрямую (без сохранения на диск!)
    return Response(content=result_bytes, media_type="image/jpeg")

def detect_grayscale(img_path):
    """Определяет, является ли изображение чёрно-белым"""
    img = cv2.imread(img_path)
    if len(img.shape) == 2:
        return True
    # Проверяем разницу между цветовыми каналами
    b, g, r = cv2.split(img)
    if cv2.norm(b - g) == 0 and cv2.norm(b - r) == 0:
        return True
    return False

def detect_grayscale_from_face(face_img):
    """Определяет, является ли лицо чёрно-белым"""
    if len(face_img.shape) == 2:
        return True
    b, g, r = cv2.split(face_img)
    if cv2.norm(b - g) == 0 and cv2.norm(b - r) == 0:
        return True
    return False

def auto_detect_and_repair_defects(img_path):
    """
    Финальная версия:
    - Только узкие царапины (не шире 10 пикселей)
    - Ограниченная область воздействия
    - Сохранение текстуры через смешивание
    """
    img = cv2.imread(img_path)
    if img is None:
        return
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # === ДЕТЕКЦИЯ ЛИЦА ДЛЯ ЗАЩИТЫ ===
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    protection_mask = np.zeros((h, w), dtype=np.uint8)
    
    for (x, y, wf, hf) in faces:
        # Защита глаз
        roi_gray = gray[y:y+hf, x:x+wf]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cx, cy = x + ex + ew//2, y + ey + eh//2
            cv2.circle(protection_mask, (cx, cy), ew + 3, 255, -1)
        
        # Защита рта
        mouth_y = y + int(hf * 0.7)
        mouth_h = int(hf * 0.2)
        mouth_x = x + int(wf * 0.2)
        mouth_w = int(wf * 0.6)
        cv2.rectangle(protection_mask, (mouth_x, mouth_y), 
                     (mouth_x + mouth_w, mouth_y + mouth_h), 255, -1)
    
    # === ДЕТЕКЦИЯ ТОЛЬКО УЗКИХ ЦАРАПИН ===
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    
    # Оставляем только тонкие линии (царапины)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
    # Детекция светлых царапин
    _, light = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    light = cv2.erode(light, kernel, iterations=1)
    
    # Объединяем
    mask = cv2.bitwise_or(edges, light)
    
    # Убираем широкие области (оставляем только узкие царапины)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        # Если область широкая — это не царапина, пропускаем
        if cw > 15 and ch > 15:
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
    
    # Расширяем маску совсем чуть-чуть
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Убираем защищённые зоны
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(protection_mask))
    
    defect_count = cv2.countNonZero(mask)
    if defect_count < 20:
        print(f"✓ No scratches detected ({defect_count} pixels)")
        return
    
    print(f"🩹 Detected {defect_count} scratch pixels, repairing...")
    
    # Сохраняем маску для отладки
    debug_path = img_path.replace('.jpg', f'_scratch_mask_{uuid.uuid4().hex}.png')
    cv2.imwrite(debug_path, mask)
    print(f"Debug mask saved: {debug_path}")
    
    # === ЛОКАЛЬНЫЙ ИНПЕЙНТИНГ (только для царапин) ===
    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # === ПЛАВНОЕ СМЕШИВАНИЕ ТОЛЬКО ПО МАСКЕ ===
    # Создаём размытую маску для плавного перехода
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
    mask_norm = mask_blur.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm, mask_norm, mask_norm], axis=2)
    
    # Смешиваем только в области маски
    blended = (result.astype(np.float32) * mask_3ch + 
               original.astype(np.float32) * (1 - mask_3ch))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    cv2.imwrite(img_path, blended)
    print("✓ Scratches repaired with texture preservation")


def remove_large_white_spot(img_path):
    """
    Отдельная функция для удаления крупных белых пятен
    (требует ручного указания области или специальной детекции)
    """
    img = cv2.imread(img_path)
    if img is None:
        return
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Детекция крупных светлых областей
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Морфология для связности
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    # Убираем мелкие шумы
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        # Только крупные пятна (больше 1000 пикселей)
        if area < 1000:
            cv2.drawContours(white_mask, [contour], -1, 0, thickness=cv2.FILLED)
    
    if cv2.countNonZero(white_mask) < 100:
        print("✓ No large white spots detected")
        return
    
    print(f"🩹 Detected large white spot ({cv2.countNonZero(white_mask)} pixels)")
    
    # Для крупных пятен используем NS метод
    result = cv2.inpaint(img, white_mask, inpaintRadius=10, flags=cv2.INPAINT_NS)
    
    # Плавное смешивание
    mask_blur = cv2.GaussianBlur(white_mask, (15, 15), 0)
    mask_norm = mask_blur.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm, mask_norm, mask_norm], axis=2)
    
    blended = (result.astype(np.float32) * mask_3ch + 
               original.astype(np.float32) * (1 - mask_3ch))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    cv2.imwrite(img_path, blended)
    print("✓ Large white spot repaired")

def detect_and_crop_face(img_path, target_size=512):
    """Детектирует лицо с помощью OpenCV, вырезает и приводит к target_size"""
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    
    # Используем Haar Cascade для детекции лица (уже в OpenCV)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("⚠️ No face detected")
        return None, None, None
    
    # Берём первое лицо (самое большое)
    x, y, w, h = faces[0]
    
    # Добавляем отступы (20% от размера)
    padding = int(max(w, h) * 0.2)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # Вырезаем лицо
    face_img = img[y:y+h, x:x+w]
    
    # Сохраняем параметры для обратной вставки
    face_info = {
        'x': x, 'y': y, 'w': w, 'h': h,
        'original_img': img.copy()
    }
    
    # Ресайз до target_size
    face_resized = cv2.resize(face_img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    return face_resized, face_info, target_size

def paste_back_face(original_info, processed_face):
    """Вставляет обработанное лицо обратно в изображение"""
    img = original_info['original_img']
    x, y, w, h = original_info['x'], original_info['y'], original_info['w'], original_info['h']
    
    # Ресайз обработанного лица до исходного размера
    processed_resized = cv2.resize(processed_face, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Вставляем
    img[y:y+h, x:x+w] = processed_resized
    
    return img

def apply_colorization(face_img):
    """
    Применяет раскрашивание к лицу 512×512.
    Возвращает ТОЛЬКО цветное лицо 512×512 (без вставки).
    """
    if face_img is None:
        return None
    
    try:
        # Сохраняем лицо во временный файл
        face_path = f"temp_face_{uuid.uuid4().hex}.png"
        cv2.imwrite(face_path, face_img)
        
        temp_dir = f"temp_color_{uuid.uuid4().hex}"
        
        cmd = [
            sys.executable,
            "CodeFormer/inference_colorization.py",
            "--input_path", face_path,
            "-o", temp_dir
        ]
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=120)
        
        print(f"Colorization return code: {result.returncode}")
        
        if result.returncode == 0 and os.path.exists(temp_dir):
            files = os.listdir(temp_dir)
            if files:
                colorized_path = os.path.join(temp_dir, files[0])
                colorized_face = cv2.imread(colorized_path)
                
                # Очистка
                os.unlink(face_path)
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                print("✓ Face colorized successfully (512x512)")
                return colorized_face  # ← ТОЛЬКО ЛИЦО, БЕЗ ВСТАВКИ
        
        # Очистка при ошибке
        if os.path.exists(face_path):
            os.unlink(face_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Colorization error: {e}")
    
    return None

@app.post(
    "/enhance",
    summary="Улучшение изображения: восстановление лиц и постобработка",
    description="""
    Обрабатывает изображение с помощью модели CodeFormer:
    
    - Автоматическое удаление царапин и дефектов (inpainting)
    - Автоматическое раскрашивание чёрно-белых фотографий (опционально)
    - Восстанавливает детали лиц (глаза, кожа, текстуры)
    - Улучшает фон с помощью Real-ESRGAN
    - Увеличивает разрешение восстановленного лица
    - Применяет постобработку: резкость, контраст, шумоподавление
    
    Параметры:
    - fidelity_weight: 0.0–1.0 (0.7 рекомендуется)
    - postprocess: true/false (включить классическую постобработку)
    - colorize: true/false (автоматическое раскрашивание ч/б фото)
    """,
    response_description="Обработанное изображение в формате JPEG"
)
async def enhance_image(
    image: UploadFile = File(...),
    fidelity_weight: float = Form(0.7),
    postprocess: bool = Form(True),
    colorize: bool = Form(False)
):
    # Валидация
    if not (0.0 <= fidelity_weight <= 1.0):
        raise HTTPException(status_code=400, detail="fidelity_weight must be 0.0–1.0")
    
    # Получаем расширение
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"

    # Создаём имя файла
    input_name = f"upload_{uuid.uuid4().hex}{ext}"
    final_result_path = None

    with open(input_name, "wb") as f:
        f.write(await image.read())

    try:
        print("=== ENHANCE START ===")
        print(f"Input file: {input_name}")
        print(f"fidelity_weight={fidelity_weight}, postprocess={postprocess}, colorize={colorize}")
        
        # === 2. ДЕТЕКЦИЯ ЛИЦА ===
        face_img, face_info, target_size = detect_and_crop_face(input_name, 512)
        
        if face_img is not None:
            # === 3. COLORIZATION (только для лица, если включено) ===
            if colorize:
                is_grayscale = detect_grayscale_from_face(face_img)
                if is_grayscale:
                    print("🎨 Applying colorization to face (512x512)...")
                    colorized_face = apply_colorization(face_img)
                    if colorized_face is not None:
                        # Возвращаем только цветное лицо
                        output_path = f"colorized_face_{uuid.uuid4().hex}.jpg"
                        cv2.imwrite(output_path, colorized_face)
                        
                        with open(output_path, "rb") as f:
                            result_bytes = f.read()
                        os.unlink(output_path)
                        
                        return Response(content=result_bytes, media_type="image/jpeg")
                    else:
                        print("⚠️ Colorization failed")
                        raise HTTPException(status_code=500, detail="Colorization failed")
                else:
                    print("✓ Face is already colored, skipping colorization")
        
        # === 4. ПОДГОТОВКА ДЛЯ CODEFORMER ===
        max_input_width = 1000
        with Image.open(input_name) as img_orig:
            if img_orig.width > max_input_width:
                ratio = max_input_width / img_orig.width
                new_size = (max_input_width, int(img_orig.height * ratio))
                img_orig = img_orig.resize(new_size, Image.LANCZOS)
                img_orig.save(input_name)
                print(f"Input resized BEFORE CodeFormer: {img_orig.size}")
        
        # === 5. ЗАПУСК CODEFORMER ===
        cmd = [
            sys.executable,
            "CodeFormer/inference_codeformer.py",
            "--input_path", input_name,
            "-w", str(fidelity_weight),
            "--bg_upsampler", "realesrgan",
            "--face_upsample"
        ]
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=600)
        
        print(f"CodeFormer return code: {result.returncode}")
        if result.returncode != 0:
            print("CodeFormer WARNING:", result.stderr[:300])
        
        # === 6. ПОИСК РЕЗУЛЬТАТА ===
        img_path = input_name
        expected_dir = f"results/test_img_{fidelity_weight}/final_results"
        if os.path.exists(expected_dir):
            files = [f for f in os.listdir(expected_dir) if os.path.isfile(os.path.join(expected_dir, f))]
            if files:
                latest = max(files, key=lambda f: os.path.getctime(os.path.join(expected_dir, f)))
                img_path = os.path.join(expected_dir, latest)
                print(f"✓ Using CodeFormer result: {img_path}")
                
        # === 7. АВТОМАТИЧЕСКОЕ УДАЛЕНИЕ ДЕФЕКТОВ (ЦАРАПИНЫ, ВМЯТИНЫ) ===
        auto_detect_and_repair_defects(input_name)
        
        # === 8. УДАЛЕНИЕ КРУПНЫХ БЕЛЫХ ПЯТЕН ===v 
        remove_large_white_spot(input_name)        
        
        # === 9. ПОСТОБРАБОТКА ===
        with Image.open(img_path) as img:
            if img.width > 2000:
                ratio = 2000 / img.width
                img = img.resize((2000, int(img.height * ratio)), Image.LANCZOS)
            
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_cv = cv2.medianBlur(img_cv, 3)  # убирает белые точки
            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
            if postprocess:
                print(">>> APPLYING POSTPROCESSING <<<")
                img = enhance_photo_pil_cv2(img)
                print("Postprocessing applied!")
            
            output_path = f"results/enhanced_{uuid.uuid4().hex}.jpg"
            img.save(output_path, "JPEG", quality=90, optimize=True)
            
            with open(output_path, "rb") as f:
                result_bytes = f.read()
            os.unlink(output_path)
            
            return Response(content=result_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(input_name):
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
    
    """ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG supported")"""

    input_name = f"upload_{uuid.uuid4().hex}.jpg"
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
        
        with open(final_result_path, "rb") as f:
            result_bytes = f.read()
        os.unlink(final_result_path)
        return Response(content=result_bytes, media_type="image/jpeg")

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
        

from fastapi.responses import Response

if __name__ == "__main__":
    print("Сервер запущен! Открой в браузере:")
    print("http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)