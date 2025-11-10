import runpod
import subprocess
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handler(event):
    """Video generation handler"""
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "")
        size = input_data.get("size", "1280*720")

        if not prompt:
            return {"error": "Prompt bo'sh"}

        logger.info(f"Generating video for: {prompt}")

        # Video generation buyrug'i
        cmd = [
            "python", "Wan2.2/generate.py",
            "--task", "t2v-A14B",
            "--size", size,
            "--ckpt_dir", "./Wan2.2/Wan2.2-T2V-A14B",
            "--offload_model", "True",
            "--convert_model_dtype",
            "--prompt", prompt
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 daqiqa
        )

        if result.returncode != 0:
            return {"error": result.stderr}

        # Video faylini topish
        output_dir = "./Wan2.2/output"
        video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]

        if not video_files:
            return {"error": "Video yaratilmadi"}

        latest_video = max(
            [os.path.join(output_dir, f) for f in video_files],
            key=os.path.getctime
        )

        # Video faylini base64 ga o'girish (yoki S3 ga yuklash)
        import base64
        with open(latest_video, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode()

        return {
            "video_base64": video_data,
            "message": "Video muvaffaqiyatli yaratildi"
        }

    except subprocess.TimeoutExpired:
        return {"error": "Timeout: Video generation 10 daqiqadan ko'p vaqt oldi"}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    logger.info("Starting Wan2.2 video generation handler...")
    runpod.serverless.start({"handler": handler})