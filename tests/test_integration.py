# tests/test_integration.py
import subprocess
import requests
import time

def test_container_health():
    """Проверка работающего контейнера"""
    # Запускаем контейнер
    subprocess.Popen([
        "docker", "run", "-d", "--name", "test-ml", "-p", "8888:8000",
        "real-esrgan-service:latest"
    ])
    time.sleep(10)  # Ждём загрузки
    
    # Проверяем health
    response = requests.get("http://localhost:8888/health")
    assert response.status_code == 200
    
    # Останавливаем контейнер
    subprocess.run(["docker", "stop", "test-ml"])
    subprocess.run(["docker", "rm", "test-ml"])