FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Ish katalogi
WORKDIR /app

# Tizim paketlarini yangilash
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python kutubxonalarini o'rnatish
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Handler faylini ko'chirish
COPY handler.py .

# Modelni oldindan yuklash (build vaqtida)
RUN python -c "from transformers import AutoTokenizer; \
    AutoTokenizer.from_pretrained('wan-ai/wan-2.2-preview')" || true

# Handler ishga tushirish
CMD ["python", "-u", "handler.py"]