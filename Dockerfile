FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -e .

COPY . /app/

ENV \
    # Model settings
    MODEL_DIR="models" \
    MODEL_NAME="bge-m3-onnx-o4" \
    ONNX_FILE="model_optimized.onnx" \
    DEVICE="cuda" \
    # Processing settings
    BATCH_SIZE=4 \
    MAX_LENGTH=5000 \
    # Queue and timeout settings
    MAX_QUEUE_SIZE=100 \
    MAX_REQUEST=10 \
    REQUEST_FLUSH_TIMEOUT=0.05 \
    REQUEST_TIMEOUT=30 \
    GPU_TIMEOUT=60 \
    # Server settings
    HOST="0.0.0.0" \
    PORT=3000 \
    WORKERS=1 \
    ENABLE_CORS="False" \
    # Worker threads for the ThreadPoolExecutor
    WORKER_THREADS=4

EXPOSE 3000

CMD ["python3", "m3_server.py"]
