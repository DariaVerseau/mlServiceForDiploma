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
#STYLE_DIR = BASE_DIR.parent / "style"
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

    style_path = supported_styles[style]  # ← файл точно существует

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

def apply_colorization(img_path):
    """Применяет раскрашивание к лицу на изображении"""
    try:
        # Детектируем лицо
        import cv2
        img = cv2.imread(img_path)
        
        # Используем Haar Cascade для детекции лица
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("⚠️ No face detected, skipping colorization")
            return None
        
        # Берём первое лицо
        x, y, w, h = faces[0]
        
        # Вырезаем лицо и увеличиваем до 512x512
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (512, 512), interpolation=cv2.INTER_LANCZOS)
        
        # Сохраняем вырезанное лицо
        face_path = f"temp_face_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(face_path, face_resized)
        
        # Запускаем colorization на вырезанном лице
        temp_dir = f"temp_color_{uuid.uuid4().hex}"
        cmd = [
            sys.executable,
            "CodeFormer/inference_colorization.py",
            "--input_path", face_path,
            "-o", temp_dir
        ]
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(temp_dir):
            # Находим результат
            files = os.listdir(temp_dir)
            if files:
                colorized_face_path = os.path.join(temp_dir, files[0])
                colorized_face = cv2.imread(colorized_face_path)
                
                # Уменьшаем обратно до исходного размера лица
                colorized_face_resized = cv2.resize(colorized_face, (w, h))
                
                # Вставляем обратно в изображение
                img[y:y+h, x:x+w] = colorized_face_resized
                cv2.imwrite(img_path, img)
                
                # Очистка
                os.unlink(face_path)
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                print("✓ Colorization applied successfully")
                return True
        
        # Очистка
        if os.path.exists(face_path):
            os.unlink(face_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"Colorization error: {e}")
    
    return None

@app.post(
    "/enhance",
    summary="Улучшение изображения: восстановление лиц и постобработка",
    description="""
    Обрабатывает изображение с помощью модели CodeFormer:
    
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

    # Если расширение не поддерживается или пустое, используем .jpg
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"

    # Создаём имя файла
    input_name = f"upload_{uuid.uuid4().hex}{ext}"
    final_result_path = None
    colorized_path = None

    with open(input_name, "wb") as f:
        f.write(await image.read())

    try:
        print("=== DEBUG START ===")
        print(f"Input file: {input_name}")
        print(f"fidelity_weight: {fidelity_weight}")
        print(f"postprocess: {postprocess}")
        print(f"colorize: {colorize}")
        
        # === АВТОМАТИЧЕСКОЕ РАСКРАШИВАНИЕ (по желанию) ===
        if colorize:
            is_grayscale = detect_grayscale(input_name)
            if is_grayscale:
                print("Grayscale image detected, applying colorization...")
                temp_color_dir = f"temp_color_{uuid.uuid4().hex}"
                colorized_result = apply_colorization(input_name, temp_color_dir)
                if colorized_result and os.path.exists(colorized_result):
                    # Заменяем входное изображение раскрашенным
                    shutil.copy(colorized_result, input_name)
                    print("✓ Colorization applied")
                    # Очищаем временную папку
                    shutil.rmtree(temp_color_dir, ignore_errors=True)
                else:
                    print("Colorization failed, proceeding with original image")
            else:
                print("✓ Image is already colored, skipping colorization")
        
        # === УМЕНЬШАЕМ ВХОД ДО CODEFORMER ===
        max_input_width = 1000
        with Image.open(input_name) as img_orig:
            if img_orig.width > max_input_width:
                ratio = max_input_width / img_orig.width
                new_size = (max_input_width, int(img_orig.height * ratio))
                img_orig = img_orig.resize(new_size, Image.LANCZOS)
                img_orig.save(input_name)
                print(f"Input resized BEFORE CodeFormer: {img_orig.size}")
        
        # === ЗАПУСК CODEFORMER С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ ===
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
            print("CodeFormer ERROR:", result.stderr)
            print("CodeFormer STDOUT:", result.stdout)
        
        # === ПОИСК РЕЗУЛЬТАТА ===
        img_path = input_name

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
            
            max_width = 2000
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                print(f"Resized to: {img.size}")
            
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            if postprocess:
                print(">>> APPLYING POSTPROCESSING <<<")
                img = enhance_photo_pil_cv2(img)
                print("Postprocessing applied!")
            else:
                print(">>> SKIPPING POSTPROCESSING <<<")
            
            postfix = f"_postprocess_w{fidelity_weight}" if postprocess else f"_w{fidelity_weight}"
            final_result_path = f"results/enhanced{postfix}_{uuid.uuid4().hex}.jpg"
            quality = 90 if postprocess else 95
            img.save(final_result_path, "JPEG", quality=quality, optimize=True)
            print(f"Saved to: {final_result_path}")
        
        print("=== DEBUG END ===")
        
        with open(final_result_path, "rb") as f:
            result_bytes = f.read()
        os.unlink(final_result_path)  
        return Response(content=result_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"Exception in enhance_image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if input_name and os.path.exists(input_name):
            try:
                os.unlink(input_name)
                print(f"Deleted input file: {input_name}")
            except Exception as e:
                print(f"Failed to delete input file: {e}")

"""@app.post(
    "/enhance"
)
async def enhance_image(
    image: UploadFile = File(...),
    fidelity_weight: float = Form(0.7),  # изменено с 0.6 на 0.7
    postprocess: bool = Form(True)
):
    # Валидация
    if not (0.0 <= fidelity_weight <= 1.0):
        raise HTTPException(status_code=400, detail="fidelity_weight must be 0.0–1.0")
    
    # Получаем расширение
    ext = os.path.splitext(image.filename)[1].lower()

    # Если расширение не поддерживается или пустое, используем .jpg
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"

    # Создаём имя файла
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
        
        # === ЗАПУСК CODEFORMER С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ ===
        cmd = [
            sys.executable,
            "CodeFormer/inference_codeformer.py",
            "--input_path", input_name,
            "-w", str(fidelity_weight),
            "--bg_upsampler", "realesrgan",  # ← ИЗМЕНЕНО с "None" на "realesrgan"
            "--face_upsample"                 # ← ДОБАВЛЕНО
        ]
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=600)
        
        print(f"CodeFormer return code: {result.returncode}")
        if result.returncode != 0:
            print("CodeFormer ERROR:", result.stderr)
            print("CodeFormer STDOUT:", result.stdout)
        
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
                # Берём самый свежий файл
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
        
        with open(final_result_path, "rb") as f:
            result_bytes = f.read()
        os.unlink(final_result_path)  
        return Response(content=result_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"Exception in enhance_image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Удаляем ТОЛЬКО входной временный файл
        if input_name and os.path.exists(input_name):
            try:
                os.unlink(input_name)
                print(f"Deleted input file: {input_name}")
            except Exception as e:
                print(f"Failed to delete input file: {e}")
  """  

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
        
        # НЕ удаляем final_result_path здесь!

from fastapi.responses import Response

"""@app.post(
    "/colorize",
    summary="Раскраска старых фотографий",
    description="Преобразует чёрно-белые изображения в цветные",
    response_description="Цветное изображение в формате JPEG"
)
async def colorize_image(image: UploadFile = File(...)):
    # Валидация
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG/BMP supported")

    input_name = f"upload_{uuid.uuid4().hex}{ext}"
    
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
        
        # Загружаем изображение и конвертируем в грейскейл
        img = Image.open(input_name).convert('L')
        img_array = np.array(img)
        
        # Предобработка для старых фото
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)
        img_array = cv2.fastNlMeansDenoising(img_array, h=12)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Конвертация в Lab
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        img_l = img_lab[:, :, 0:1].astype(np.float32)
        img_l_norm = img_l / 100.0
        img_l_rs = torch.from_numpy(img_l_norm).permute(2, 0, 1).float().unsqueeze(0)
        img_l_rs = F.interpolate(img_l_rs, size=(224, 224), mode='bilinear')
        
        # Цветизация
        with torch.no_grad():
            img_ab = colorizer(img_l_rs)
            img_ab = F.interpolate(img_ab, size=(img_lab.shape[0], img_lab.shape[1]), mode='bilinear')
        
        # Реконструкция
        img_lab_out = np.zeros_like(img_lab)
        img_lab_out[:, :, 0] = img_l[:, :, 0]
        img_lab_out[:, :, 1:] = img_ab[0].cpu().permute(1, 2, 0).numpy() * 100.0
        img_rgb_out = cv2.cvtColor(np.uint8(img_lab_out), cv2.COLOR_Lab2RGB)
        
        # Смешивание с оригиналом
        img_original_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        colorized = cv2.addWeighted(img_rgb_out, 0.7, img_original_rgb, 0.3, 0)
        
        # Конвертация в байты
        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
        if not is_success:
            raise RuntimeError("Failed to encode image")
        
        print("=== COLORIZE SUCCESS ===")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        print(f"✗ Error colorizing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Colorization failed: {str(e)}")
    
    finally:
        # Удаляем временный файл
        if os.path.exists(input_name):
            os.unlink(input_name)
 """

if __name__ == "__main__":
    print("Сервер запущен! Открой в браузере:")
    print("http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)