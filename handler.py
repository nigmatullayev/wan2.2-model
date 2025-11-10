import runpod
import subprocess
import os
import json
import logging
import base64
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Wan2.2 papkasi
WAN_DIR = "/app/Wan2.2"
MODEL_DIR = "/app/Wan2.2/Wan2.2-T2V-A14B"
OUTPUT_DIR = "/app/Wan2.2/output"

# Model mavjudligini tekshirish va yuklab olish
def ensure_model_downloaded():
    """Model mavjudligini tekshirish va agar yo'q bo'lsa yuklab olish"""
    # Model papkasini tekshirish
    if os.path.exists(MODEL_DIR):
        files = os.listdir(MODEL_DIR)
        if files and len(files) > 0:
            logger.info(f"✅ Model allaqachon mavjud: {MODEL_DIR} ({len(files)} fayl)")
            return True
        else:
            logger.warning(f"⚠️  Model papkasi bo'sh: {MODEL_DIR}")
    
    logger.info("📥 Model topilmadi, yuklab olinmoqda...")
    logger.info(f"📂 Model papkasi: {MODEL_DIR}")
    
    try:
        # Disk space tekshirish
        import shutil
        stat = shutil.disk_usage(MODEL_DIR)
        free_gb = stat.free / (1024**3)
        logger.info(f"💾 Bo'sh disk joyi: {free_gb:.2f} GB")
        
        if free_gb < 50:
            logger.warning(f"⚠️  Disk joyi yetarli emas (kerak: ~50GB, mavjud: {free_gb:.2f}GB)")
        
        from huggingface_hub import snapshot_download
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        logger.info("🔄 HuggingFace Hub orqali model yuklab olinmoqda...")
        logger.info("⏳ Bu uzoq vaqt olishi mumkin (model ~27GB)...")
        
        snapshot_download(
            repo_id='Wan-AI/Wan2.2-T2V-A14B',
            local_dir=MODEL_DIR,
            resume_download=True,
            local_files_only=False
        )
        
        # Model yuklab olinganini tekshirish
        if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
            files_count = len(os.listdir(MODEL_DIR))
            logger.info(f"✅ Model muvaffaqiyatli yuklab olindi! ({files_count} fayl)")
            return True
        else:
            logger.error("❌ Model yuklab olindi, lekin papka bo'sh!")
            return False
            
    except ImportError as e:
        logger.error(f"❌ HuggingFace Hub import xatolik: {e}")
        logger.error("💡 Iltimos, 'huggingface_hub' paketini o'rnatishni tekshiring")
        return False
    except Exception as e:
        logger.error(f"❌ Model yuklab olishda xatolik: {e}", exc_info=True)
        logger.error(f"📋 Xatolik turi: {type(e).__name__}")
        return False


def handler(event):
    """Video generation handler"""
    try:
        logger.info(f"📥 Request qabul qilindi: {event}")

        # Model mavjudligini tekshirish va yuklab olish
        model_status = ensure_model_downloaded()
        if not model_status:
            return {
                "error": "Model yuklab olinmadi. Iltimos, qayta urinib ko'ring.",
                "details": "Model yuklab olishda xatolik yuz berdi. Log'larni tekshiring."
            }

        # Input parametrlar
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "")
        size = input_data.get("size", "1280*720")  # 1280*720 yoki 848*480
        offload = input_data.get("offload_model", True)

        if not prompt:
            return {"error": "Prompt kiritilmagan"}

        logger.info(f"🎬 Video yaratilmoqda: {prompt}")
        logger.info(f"📐 Razmer: {size}")

        # Output papkasini tozalash
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Wan2.2 generate.py ni chaqirish
        cmd = [
            "python", f"{WAN_DIR}/generate.py",
            "--task", "t2v-A14B",
            "--size", size,
            "--ckpt_dir", MODEL_DIR,
            "--prompt", prompt
        ]

        # Memory optimization parametrlari
        if offload:
            cmd.extend(["--offload_model", "True"])
            cmd.append("--convert_model_dtype")
            cmd.append("--t5_cpu")

        logger.info(f"🚀 Buyruq: {' '.join(cmd)}")

        # Video generation (uzoq vaqt ketishi mumkin!)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 daqiqa timeout
            cwd=WAN_DIR
        )

        logger.info(f"📋 STDOUT: {result.stdout}")

        if result.stderr:
            logger.warning(f"⚠️  STDERR: {result.stderr}")

        if result.returncode != 0:
            return {
                "error": f"Video generation failed: {result.stderr}",
                "stdout": result.stdout
            }

        # Yaratilgan video faylini topish
        video_files = list(Path(OUTPUT_DIR).glob("*.mp4"))

        if not video_files:
            return {
                "error": "Video fayl topilmadi",
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        # Eng oxirgi yaratilgan video
        latest_video = max(video_files, key=lambda p: p.stat().st_ctime)
        logger.info(f"✅ Video topildi: {latest_video}")

        # Video hajmini tekshirish
        video_size_mb = latest_video.stat().st_size / (1024 * 1024)
        logger.info(f"📦 Video hajmi: {video_size_mb:.2f} MB")

        # Video ni base64 ga o'girish
        with open(latest_video, 'rb') as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        logger.info("✅ Video muvaffaqiyatli yaratildi!")

        return {
            "success": True,
            "video_base64": video_base64,
            "video_size_mb": round(video_size_mb, 2),
            "video_filename": latest_video.name,
            "message": "Video muvaffaqiyatli yaratildi"
        }

    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout: 15 daqiqadan ko'p vaqt oldi")
        return {"error": "Video generation timeout (15 min)"}

    except Exception as e:
        logger.error(f"❌ Xatolik: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    logger.info("🎥 Wan2.2 Video Generation Handler ishga tushdi!")
    runpod.serverless.start({"handler": handler})