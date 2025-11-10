FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Python o'rnatish
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python3 ni python deb belgilash
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]