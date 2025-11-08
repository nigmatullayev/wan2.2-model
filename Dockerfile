FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y git python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY handler.py .

# RunPod handler entrypoint
CMD ["python3", "handler.py"]
