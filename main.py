from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
#from wct2_processor import process_image --- более продвинутая модель, возможно заменю на нее позже
from basic_style_transfer import process_image #пока использую базовый перенос стиля 
import uuid
import os
from processors.super_resolution import upscale_image

app = FastAPI(title="WCT2 Style Transfer Service")

# Создаём папку для результатов при запуске
os.makedirs("results", exist_ok=True)

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
        pattern="^(vangogh|picasso|monet|monet2|erinHanson)$"
    )
):
    # Создаём папку results если не существует
    os.makedirs("results", exist_ok=True)
    
    # Проверка поддерживаемых стилей
    supported_styles = ["vangogh", "picasso", "monet", "monet2", "erinHanson"]
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

if __name__ == "__main__":
    print("Сервер запущен! Открой в браузере:")
    print("http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)