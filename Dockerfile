# Dockerfile for Wan 2.1 Video Generation Homework
# Base image with CUDA support and PyTorch
# Using CUDA 12.1 base image (compatible with host CUDA 12.8 runtime)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libx264-dev \
    libxvidcore-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip, setuptools, and wheel to latest versions
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel 

# Copy requirements first for better caching
COPY /requirements.txt /app/requirements.txt

# Install Python dependencies
# Install huggingface_hub first, then other requirements (excluding torch and flash_attn)
RUN pip3 install --no-cache-dir huggingface_hub[cli]
# Copy the entire project
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
# CUDA_VISIBLE_DEVICES will be set automatically by docker-entrypoint.sh
# to select GPU with highest available memory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN pip install git+https://github.com/huggingface/diffusers
RUN pip install accelerate wand

RUN pip install -r /app/requirements.txt

# accelerate configuration saved at $HOME/.cache/huggingface/accelerate/default_config.yaml
CMD ["bash"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
