FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    ninja-build \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install packaging setuptools wheel

# PyTorch 및 관련 패키지 설치
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install causal_conv1d==1.1.1

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
