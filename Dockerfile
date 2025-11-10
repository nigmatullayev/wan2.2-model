FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Python o'rnatish
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Wan2.2 repository clone
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

WORKDIR /app/Wan2.2

# Dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

# Model yuklab olish (80GB+ kerak!)
RUN pip install "huggingface_hub[cli]" && \
    huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B

# Handler
COPY handler_video.py /app/handler.py

WORKDIR /app

CMD ["python", "-u", "handler.py"]