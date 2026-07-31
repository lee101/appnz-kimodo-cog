FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TEXT_ENCODER_MODE=local

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-gpu.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-gpu.txt \
    && python -m pip install --no-deps git+https://github.com/nv-tlabs/kimodo.git

COPY . /app
CMD ["python", "-u", "runpod_handler.py"]

