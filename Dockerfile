FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Ish katalogini sozlash
WORKDIR /app

# Requirements faylini ko'chirish va o'rnatish
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# WAN 2.2 modelini oldindan yuklab olish (ixtiyoriy, lekin tavsiya etiladi)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('wan-ai/wan-2.2-preview'); \
    AutoModelForCausalLM.from_pretrained('wan-ai/wan-2.2-preview', device_map='cpu')"

# Handler faylini ko'chirish
COPY handler.py .

# Portni ochish (ixtiyoriy)
EXPOSE 8000

# Handler ishga tushirish
CMD ["python", "-u", "handler.py"]
