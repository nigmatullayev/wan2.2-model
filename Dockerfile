FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# --- System packages ---
RUN apt-get update && \
    apt-get install -y git python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Copy and prepare requirements ---
COPY requirements.txt .

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# --- Install PyTorch (GPU) manually first ---
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Install remaining dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy app files ---
COPY handler.py .

# --- Entrypoint ---
CMD ["python3", "handler.py"]
