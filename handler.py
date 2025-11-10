import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model nomi
MODEL_NAME = "wan-ai/wan-2.2-preview"
tokenizer = None
model = None


def load_model():
    """Model yuklanish funksiyasi"""
    global tokenizer, model

    try:
        if model is None:
            logger.info(f"Model yuklanmoqda: {MODEL_NAME}")

            # Cache papkasini sozlash
            cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")

            # Tokenizer yuklash
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            # Model yuklash
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            logger.info("Model muvaffaqiyatli yuklandi!")
            logger.info(f"Model device: {model.device}")

        return tokenizer, model

    except Exception as e:
        logger.error(f"Model yuklashda xatolik: {str(e)}")
        raise


def handler(event):
    """RunPod handler"""
    try:
        # Input tekshirish
        if not event or "input" not in event:
            return {"error": "Input topilmadi"}

        # Model yuklash
        tok, mdl = load_model()

        # Parametrlarni olish
        input_data = event["input"]
        prompt = input_data.get("prompt", "")
        max_new_tokens = input_data.get("max_new_tokens", 256)
        temperature = input_data.get("temperature", 0.7)
        top_p = input_data.get("top_p", 0.9)

        if not prompt:
            return {"error": "Prompt kiritilmagan"}

        logger.info(f"Prompt qabul qilindi: {prompt[:100]}...")

        # Tokenizatsiya
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        # Text generation
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )

        # Faqat yangi tokenlarni dekodlash
        generated_text = tok.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        logger.info("Generation tugallandi")

        return {
            "output": generated_text,
            "input_tokens": input_length,
            "output_tokens": len(outputs[0]) - input_length,
            "total_tokens": len(outputs[0])
        }

    except Exception as e:
        error_msg = f"Handler xatolik: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


# Serverless start
if __name__ == "__main__":
    logger.info("RunPod serverless handler ishga tushmoqda...")
    runpod.serverless.start({"handler": handler})