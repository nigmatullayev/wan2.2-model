# Wan2.2 Text-to-Video Model - RunPod Serverless

Professional MLOps implementation of Wan2.2-T2V-A14B text-to-video model for RunPod serverless platform.

## 📋 Overview

This project provides a production-ready serverless endpoint for generating videos from text prompts using the [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) model. The implementation includes:

- ✅ **Model Caching**: Automatic model download and caching on first request
- ✅ **Error Handling**: Comprehensive error handling with retry mechanisms
- ✅ **Thread Safety**: Thread-safe model initialization
- ✅ **Logging**: Detailed logging for debugging and monitoring
- ✅ **Resource Management**: Efficient memory and disk space management
- ✅ **Production Ready**: MLOps best practices implementation

## 🚀 Features

- **Text-to-Video Generation**: Generate 5-second videos at 480P or 720P resolution
- **Automatic Model Management**: Model is downloaded automatically on first request
- **Retry Mechanism**: Automatic retry on model download failures
- **Memory Optimization**: Configurable model offloading for GPU memory efficiency
- **Base64 Response**: Videos returned as base64-encoded strings
- **Health Checks**: Built-in health check endpoints

## 📁 Project Structure

```
wan2.2_model/
├── Dockerfile          # Docker container configuration
├── handler.py          # RunPod serverless handler with MLOps features
├── requirements.txt    # Python dependencies
├── .dockerignore       # Docker ignore file
└── README.md           # This file
```

## 🔧 Installation & Deployment

### Prerequisites

- RunPod account with serverless access
- GPU-enabled pod (recommended: 80GB+ VRAM for full model, or use offload options)
- **Minimum 50GB disk space** (required for model storage)

### Deployment Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nigmatullayev/wan2.2-model.git
   cd wan2.2-model
   ```

2. **Deploy to RunPod**:
   - Go to RunPod Dashboard → Serverless → Create Template
   - Upload the Dockerfile and handler.py
   - Configure GPU requirements (recommended: A100 80GB or similar)
   - **IMPORTANT: Set Volume Size to at least 50GB** (required for model storage)
   - Deploy the template

### ⚠️ Disk Space Configuration

**CRITICAL**: Model requires ~27GB disk space. You must configure disk size in RunPod template:

1. **Go to RunPod Dashboard** → Serverless → Templates
2. **Create or Edit Template**:
   - Find **"Volume Size"** or **"Disk Size"** setting
   - Set to **minimum 50GB** (recommended: 60-100GB for safety)
3. **Save Template** and redeploy

**Why 50GB?**
- Model size: ~27GB
- System files: ~5GB
- Temporary files: ~5GB
- Safety margin: ~13GB
- **Total recommended: 50GB+**

If you see "Disk joyi yetarli emas" error, increase the volume size in your template.

4. **Automatic Deployment** (GitHub Actions):
   - Push to main branch triggers automatic deployment
   - Configure RunPod API key in GitHub Secrets

## 📝 API Usage

### Request Format

```json
{
  "input": {
    "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "size": "1280*720",
    "offload_model": true
  }
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description for video generation |
| `size` | string | No | `"1280*720"` | Video resolution: `"1280*720"` or `"848*480"` |
| `offload_model` | boolean | No | `true` | Enable model offloading for memory optimization |

### Response Format

**Success Response**:
```json
{
  "success": true,
  "video_base64": "base64_encoded_video_data...",
  "video_size_mb": 12.5,
  "video_filename": "generated_video.mp4",
  "execution_time_seconds": 245.3,
  "message": "Video muvaffaqiyatli yaratildi"
}
```

**Error Response**:
```json
{
  "error": "Error message",
  "details": "Detailed error information",
  "model_status": {
    "status": "downloading|completed|error",
    "error": "Error message if any",
    "progress": 0
  }
}
```

## 🛠️ Technical Details

### Model Information

- **Model**: Wan-AI/Wan2.2-T2V-A14B
- **Type**: Text-to-Video MoE (Mixture of Experts)
- **Size**: ~27GB
- **Resolutions**: 480P (848×480) and 720P (1280×720)
- **Video Length**: 5 seconds
- **FPS**: 24

### System Requirements

- **Base Image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Python**: 3.10
- **CUDA**: 11.8+
- **GPU**: Recommended 80GB+ VRAM (or use offload options)
- **Disk Space**: Minimum 50GB free space

### Architecture

1. **Initialization**: Model is downloaded in background thread on first startup
2. **Caching**: Model is cached after first download, subsequent requests use cached model
3. **Thread Safety**: Thread-safe model download and access
4. **Error Handling**: Comprehensive error handling with retry mechanisms
5. **Resource Management**: Automatic disk space checking and memory optimization

## 🔍 Monitoring & Debugging

### 📋 Log'larni Ko'rish

RunPod serverless log'larini ko'rish uchun:

#### 1. RunPod Dashboard orqali:
1. **RunPod Dashboard** ga kiring: https://www.runpod.io/
2. **Serverless** → **Endpoints** bo'limiga o'ting
3. Endpoint'ni toping va **"View Logs"** yoki **"Logs"** tugmasini bosing
4. Real-time log'larni ko'rasiz

#### 2. Endpoint Details orqali:
1. Endpoint'ni oching
2. **"Logs"** tab'ini tanlang
3. Yoki **"View Logs"** tugmasini bosing

#### 3. Request Log'lari:
- Har bir request uchun alohida log'lar ko'rsatiladi
- Request ID orqali log'larni filtrlash mumkin
- Real-time log streaming mavjud

### 📊 Log Format

Log'larda quyidagilar ko'rinadi:
- ✅ Model download progress (har 5 soniyada)
- ✅ Video generation status
- ✅ Error messages with stack traces
- ✅ Resource usage information
- ✅ Disk space information
- ✅ Model status updates

### Log Misollari:

```
📥 Model yuklab olinmoqda: Wan-AI/Wan2.2-T2V-A14B
📂 Manzil: /app/Wan2.2/Wan2.2-T2V-A14B
🚀 Model yuklab olish boshlandi...
📥 Repository ma'lumotlari olinmoqda...
📋 Jami fayllar: 45
📥 Model fayllari yuklab olinmoqda...
📊 Progress: 15.3% | 4.13 GB / ~27.00 GB | Tezlik: 28.45 MB/s | ETA: 13.2 min
📊 Progress: 32.7% | 8.83 GB / ~27.00 GB | Tezlik: 31.20 MB/s | ETA: 9.8 min
✅ Model yuklab olish yakunlandi!
📦 Jami hajm: 27.45 GB
🎬 Video yaratilmoqda...
✅ Video muvaffaqiyatli yaratildi!
```

### Health Checks

Health check endpoint verifies:
- CUDA availability
- Model directory existence
- System resources

## ⚙️ Configuration

### Environment Variables

- `PYTHONUNBUFFERED=1`: Unbuffered Python output
- `DEBIAN_FRONTEND=noninteractive`: Non-interactive package installation

### Model Download

Model is automatically downloaded on first request. The download process:
- Checks disk space (requires ~30GB)
- Downloads model with resume support
- Verifies model integrity
- Caches model for subsequent requests

## 🐛 Troubleshooting

### Model Download Fails

1. **Check Disk Space**: Ensure at least 50GB free space
2. **Check Network**: Verify internet connectivity
3. **Check Logs**: Review detailed error logs
4. **Retry**: The handler automatically retries up to 3 times

### Out of Memory Errors

1. **Enable Offload**: Set `offload_model: true` in request
2. **Use Smaller Resolution**: Use `"848*480"` instead of `"1280*720"`
3. **Upgrade GPU**: Use GPU with more VRAM

### Timeout Errors

1. **Increase Timeout**: Default timeout is 15 minutes
2. **Check Model Status**: Verify model is downloaded and ready
3. **Check GPU**: Ensure GPU is available and functioning

## 📚 References

- [Wan2.2 GitHub Repository](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2-T2V-A14B Model](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
- [RunPod Documentation](https://docs.runpod.io/)
- [HuggingFace Hub](https://huggingface.co/docs/hub)

## 📄 License

This project is licensed under the Apache 2.0 License.

## 👤 Author

GitHub: [@nigmatullayev](https://github.com/nigmatullayev)

## 🙏 Acknowledgments

- [Wan Video Team](https://github.com/Wan-Video) - For the Wan2.2 model
- [RunPod](https://www.runpod.io/) - For the serverless platform
- [HuggingFace](https://huggingface.co/) - For model hosting

## 🔄 Version History

- **v1.0.0**: Initial release with MLOps best practices
  - Model caching and automatic download
  - Thread-safe initialization
  - Comprehensive error handling
  - Production-ready implementation
