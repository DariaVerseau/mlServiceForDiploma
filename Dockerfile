# ============================================================
# ЭТАП 1: Сборщик (Builder)
# ============================================================
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ===== 1. Установка PyTorch и NumPy 1.x (ФИКСИРОВАННЫЕ ВЕРСИИ) =====
RUN pip install --no-cache-dir \
    torch==1.13.1 \
    torchvision==0.14.1 \
    'numpy==1.24.4' \
    --index-url https://download.pytorch.org/whl/cpu

# ===== 2. Установка basicsar ИЗ РЕПОЗИТОРИЯ (не через pip!) =====
# Это критически важно: basicsr.data.degradations есть только в репозитории
RUN git clone https://github.com/xinntao/BasicSR.git && \
    cd BasicSR && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py develop

# ===== 3. Real-ESRGAN (использует basicsr из репозитория) =====
RUN git clone https://github.com/xinntao/Real-ESRGAN.git && \
    cd Real-ESRGAN && \
    pip install --no-cache-dir facexlib gfpgan && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py develop

# ===== 4. CodeFormer =====
RUN git clone https://github.com/sczhou/CodeFormer.git && \
    cd CodeFormer && \
    pip install --no-cache-dir -r requirements.txt

# ===== 5. pytorch-AdaIN =====
RUN git clone https://github.com/naoto0804/pytorch-AdaIN.git

# ===== 6. Установка остальных пакетов =====
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pillow \
    opencv-python \
    httpx \
    aiofiles

# ===== 7. Скачивание весов =====
RUN mkdir -p /build/weights/realesrgan && \
    curl -L -o /build/weights/realesrgan/RealESRGAN_x4plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" && \
    curl -L -o /build/weights/realesrgan/RealESRGAN_x2plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

RUN mkdir -p /build/weights/codeformer && \
    curl -L -o /build/weights/codeformer/codeformer.pth \
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

RUN mkdir -p /build/weights/adain && \
    curl -L -o /build/weights/adain/vgg_normalised.pth \
    "https://raw.githubusercontent.com/naoto0804/pytorch-AdaIN/master/models/vgg_normalised.pth" && \
    curl -L -o /build/weights/adain/decoder.pth \
    "https://raw.githubusercontent.com/naoto0804/pytorch-AdaIN/master/models/decoder.pth"

# ===== В КОНЦЕ BUILDER ЭТАПА: принудительная фиксация NumPy 1.x =====
RUN pip install --force-reinstall 'numpy==1.24.4'

# ============================================================
# ЭТАП 2: Финальный образ (Runtime)
# ============================================================
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/BasicSR:/app/Real-ESRGAN:/app/CodeFormer:/app/pytorch-AdaIN"

# Установка runtime-зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libx11-6 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование установленных Python-пакетов
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копирование репозиториев (ВКЛЮЧАЯ BasicSR!)
COPY --from=builder /build/BasicSR /app/BasicSR
COPY --from=builder /build/Real-ESRGAN /app/Real-ESRGAN
COPY --from=builder /build/CodeFormer /app/CodeFormer
COPY --from=builder /build/pytorch-AdaIN /app/pytorch-AdaIN

# Копирование весов
COPY --from=builder /build/weights /app/weights

# ===== В ФИНАЛЬНОМ ЭТАПЕ: переустановка NumPy =====
RUN pip install --force-reinstall --no-cache-dir 'numpy==1.24.4'

# Создание символических ссылок
RUN mkdir -p /root/.cache/realesrgan && \
    ln -sf /app/weights/realesrgan/RealESRGAN_x4plus.pth /root/.cache/realesrgan/RealESRGAN_x4plus.pth && \
    ln -sf /app/weights/realesrgan/RealESRGAN_x2plus.pth /root/.cache/realesrgan/RealESRGAN_x2plus.pth && \
    mkdir -p /app/CodeFormer/weights/CodeFormer && \
    ln -sf /app/weights/codeformer/codeformer.pth /app/CodeFormer/weights/CodeFormer/codeformer.pth && \
    mkdir -p /app/pytorch-AdaIN/models && \
    ln -sf /app/weights/adain/vgg_normalised.pth /app/pytorch-AdaIN/models/vgg_normalised.pth && \
    ln -sf /app/weights/adain/decoder.pth /app/pytorch-AdaIN/models/decoder.pth

# Создание пользователя
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/results /app/styles /app/temp_content && \
    chown -R appuser:appuser /app

USER appuser

# Копирование кода проекта
COPY --chown=appuser:appuser . .

COPY --chown=appuser:appuser styles/ ./styles/

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]