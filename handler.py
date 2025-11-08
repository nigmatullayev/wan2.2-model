import runpod
import torch
from diffusers import DiffusionPipeline

# Modelni bir marta yuklab olish
print("🚀 Loading Wan2.2 model...")
pipe = DiffusionPipeline.from_pretrained("Wan-2.2", torch_dtype=torch.float16)
pipe.to("cuda")

def generate_video(job):
    """RunPod handler function"""
    prompt = job["input"].get("prompt", "A cinematic mountain landscape at sunset")
    num_frames = job["input"].get("num_frames", 16)
    output_file = "output.mp4"

    print(f"🎬 Generating video: {prompt}")
    video = pipe(prompt=prompt, num_frames=num_frames).videos[0]
    video.save(output_file)

    return {"message": "Video generated successfully", "file": output_file}

# RunPod serverless handler
runpod.serverless.start({"handler": generate_video})
