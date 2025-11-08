FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System packages
RUN apt-get update && \
    apt-get install -y git python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .

# Upgrade pip & setuptools (important for diffusers/torch metadata build)
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy main handler code
COPY handler.py .

# Entrypoint for RunPod Serverless
CMD ["python3", "handler.py"]
