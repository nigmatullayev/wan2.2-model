# Wan2.2 Text-to-Video Model - RunPod Serverless
# Professional MLOps Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python aliases
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip and install base packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# NumPy 1.x (NumPy 2.x compatibility issue fix)
RUN pip install --no-cache-dir "numpy<2.0"

WORKDIR /app

# Clone Wan2.2 repository
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2 && \
    cd /app/Wan2.2 && \
    git checkout main || true

WORKDIR /app/Wan2.2

# NumPy constraint file
RUN echo "numpy<2.0" > /tmp/numpy_constraint.txt

# Install Wan2.2 dependencies with NumPy constraint
RUN if [ -f requirements.txt ]; then \
        echo "📦 Installing dependencies from requirements.txt..." && \
        pip install --no-cache-dir -r requirements.txt --constraint /tmp/numpy_constraint.txt 2>&1 | tee /tmp/install.log || \
        (echo "⚠️  Constraint install failed, trying without constraint..." && \
         pip install --no-cache-dir -r requirements.txt 2>&1 | tee /tmp/install.log || \
         (echo "❌ Dependency installation failed!" && \
          echo "📋 Error log:" && tail -100 /tmp/install.log && \
          echo "📋 First 50 lines of requirements.txt:" && head -50 requirements.txt && \
          exit 1)); \
    else \
        echo "⚠️  requirements.txt not found, skipping..." && \
        ls -la; \
    fi

# Install RunPod and HuggingFace Hub
RUN pip install --no-cache-dir runpod "huggingface_hub[cli]"

# Create model and output directories
# NOTE: Model will be downloaded at runtime (first request) to avoid build timeout
RUN mkdir -p ./Wan2.2-T2V-A14B /app/Wan2.2/output && \
    echo "✅ Directories created (model will be downloaded at runtime)"

# Copy handler file
COPY handler.py /app/handler.py

# Verify handler file
RUN if [ ! -f /app/handler.py ]; then \
        echo "❌ handler.py not found!" && exit 1; \
    else \
        echo "✅ handler.py copied successfully"; \
    fi

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Start handler
CMD ["python", "-u", "handler.py"]
