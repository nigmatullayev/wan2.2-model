import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import sys

# Logging sozlash
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Global
MODEL_NAME = "wan-ai/wan-2.2-preview"
tokenizer = None
model = None


def load_model():
    """Model yuklash"""
    global tokenizer, model

    if model is not None:
        return tokenizer, model

    try:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        logger.info(f"Model yuklanmoqda: {MODEL_NAME}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Pad token sozlash
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model
        logger.info("Model yuklanmoqda...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info(f"Model muvaffaqiyatli yuklandi! Device: {model.device}")
        return tokenizer, model

    except Exception as e:
        logger.error(f"Model yuklashda xatolik: {e}", exc_info=True)
        raise


def handler(event):
    """Handler funksiyasi"""
    try:
        logger.info(f"Request qabul qilindi: {event}")

        # Model yuklash
        tok, mdl = load_model()

        # Input
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "")
        max_new_tokens = input_data.get("max_new_tokens", 256)
        temperature = input_data.get("temperature", 0.7)
        top_p = input_data.get("top_p", 0.9)

        if not prompt:
            return {"error": "Prompt bo'sh"}

        logger.info(f"Generating for prompt: {prompt[:100]}...")

        # Tokenize
        inputs = tok(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )

        # Decode
        generated = tok.decode(outputs[0][input_len:], skip_special_tokens=True)

        logger.info("Generation tugadi")

        return {
            "output": generated,
            "tokens": len(outputs[0]) - input_len
        }

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})