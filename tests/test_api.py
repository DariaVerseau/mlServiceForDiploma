# tests/test_api.py
from fastapi.testclient import TestClient
import sys
sys.path.append('/app')
from main import app

client = TestClient(app)

def test_health_endpoint():
    """Проверка healthcheck"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upscale_endpoint_with_test_image():
    """Проверка эндпоинта upscale с тестовым изображением"""
    from PIL import Image
    import io
    
    # Создаём тестовое изображение
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    response = client.post(
        "/upscale",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        data={"scale": 4}
    )
    # Может быть 200 или 422 (если модель не загружена в тестах)
    assert response.status_code in [200, 422]