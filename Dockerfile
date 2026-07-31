FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TEXT_ENCODER_MODE=local

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-gpu.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-gpu.txt

COPY patches /app/patches
RUN git clone --filter=blob:none https://github.com/nv-tlabs/kimodo.git /tmp/kimodo \
    && git -C /tmp/kimodo checkout --detach 1aece8c124d73d255ceff5086d983b844c9f4e94 \
    && git -C /tmp/kimodo apply /app/patches/kimodo-python3-cmake.patch \
    && python -m pip install --no-deps /tmp/kimodo \
    && rm -rf /tmp/kimodo

COPY . /app
CMD ["python", "-u", "runpod_handler.py"]
