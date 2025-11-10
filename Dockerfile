FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python va dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Pip yangilash
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# NumPy 1.x o'rnatish (NumPy 2.x bilan muammo bo'lmasligi uchun)
RUN pip install --no-cache-dir "numpy<2.0"

WORKDIR /app

# Wan2.2 repository clone
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

WORKDIR /app/Wan2.2

# NumPy constraint faylini yaratish
RUN echo "numpy<2.0" > /tmp/numpy_constraint.txt

# Wan2.2 dependencies o'rnatish
# NumPy versiyasini constraint qilib o'rnatamiz
RUN if [ -f requirements.txt ]; then \
        echo "📦 Requirements.txt topildi, dependency'lar o'rnatilmoqda..." && \
        echo "📋 NumPy 1.x bilan constraint qilib o'rnatilmoqda..." && \
        pip install --no-cache-dir -r requirements.txt --constraint /tmp/numpy_constraint.txt 2>&1 | tee /tmp/install.log || \
        (echo "⚠️  Constraint bilan o'rnatishda xatolik, oddiy usul bilan urinib ko'ramiz..." && \
         pip install --no-cache-dir -r requirements.txt 2>&1 | tee /tmp/install.log || \
         (echo "❌ Requirements.txt o'rnatishda xatolik!" && \
          echo "📋 Xatolik log'i:" && cat /tmp/install.log && \
          echo "📋 Fayl tarkibi (birinchi 50 qator):" && head -50 requirements.txt && \
          exit 1)); \
    else \
        echo "⚠️  Requirements.txt topilmadi, skip qilindi" && \
        echo "📋 Papka tarkibi:" && ls -la; \
    fi

# RunPod qo'shish
RUN pip install --no-cache-dir runpod

# HuggingFace Hub o'rnatish (runtime'da model yuklab olish uchun)
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# Model papkasini yaratish (runtime'da yuklab olinadi)
RUN mkdir -p ./Wan2.2-T2V-A14B

# Output papkasi
RUN mkdir -p /app/Wan2.2/output

# Handler faylini ko'chirish (build context'da mavjudligini tekshirish)
COPY handler.py /app/handler.py

# Handler faylini tekshirish
RUN if [ ! -f /app/handler.py ]; then \
        echo "❌ Handler.py topilmadi!" && exit 1; \
    else \
        echo "✅ Handler.py muvaffaqiyatli ko'chirildi"; \
    fi

WORKDIR /app

# Health check (ixtiyoriy)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Handler ishga tushirish
CMD ["python", "-u", "handler.py"]