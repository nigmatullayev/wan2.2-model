import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Logging sozlash
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global o'zgaruvchilar (bir marta yuklanadi)
MODEL_NAME = "wan-ai/wan-2.2-preview"
tokenizer = None
model = None


def load_model():
    """Modelni yuklash funksiyasi"""
    global tokenizer, model

    if model is None:
        logger.info("Model yuklanmoqda...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        logger.info("Model muvaffaqiyatli yuklandi!")

    return tokenizer, model


def handler(event):
    """RunPod serverless handler"""
    try:
        # Modelni yuklash
        tok, mdl = load_model()

        # Input parametrlarini olish
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "")
        max_length = input_data.get("max_length", 512)
        temperature = input_data.get("temperature", 0.7)
        top_p = input_data.get("top_p", 0.9)

        if not prompt:
            return {"error": "Prompt bo'sh bo'lmasligi kerak"}

        logger.info(f"Prompt: {prompt[:50]}...")

        # Tokenizatsiya
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)

        # Generation
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tok.eos_token_id
            )

        # Natijani dekodlash
        result = tok.decode(outputs[0], skip_special_tokens=True)

        logger.info("Generation muvaffaqiyatli yakunlandi")

        return {
            "output": result,
            "tokens_generated": len(outputs[0])
        }

    except Exception as e:
        logger.error(f"Xatolik: {str(e)}")
        return {"error": str(e)}


# RunPod serverless start
if __name__ == "__main__":
    logger.info("RunPod serverless handler ishga tushmoqda...")
    runpod.serverless.start({"handler": handler})