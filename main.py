from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
from wct2_processor import process_image

app = FastAPI(title="WCT2 Style Transfer Service")

# Создаём папку для результатов при запуске
os.makedirs("results", exist_ok=True)

@app.post(
    "/process",
    summary="Применить художественный стиль к изображению",
    description="""
    Принимает изображение и название стиля, возвращает обработанное изображение.
    
    Поддерживаемые стили: vangogh, monet, picasso
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
        description="Художественный стиль для применения"
    )
):
    try:
        # Читаем файл
        content = await image.read()
        
        # Обрабатываем изображение
        result_bytes = process_image(content, style)
        
        # Сохраняем результат
        result_filename = f"result_{image.filename}"
        result_path = f"results/{result_filename}"
        with open(result_path, "wb") as f:
            f.write(result_bytes)
            
        return JSONResponse({
            "result_url": f"/results/{result_filename}",
            "style": style
        })
        
    except Exception as e:
        return JSONResponse(
            {"error": str(e)}, 
            status_code=500
        )

@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = f"results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    print("Сервер запущен! Открой в браузере:")
    print("http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)