"""
Wan2.2 Text-to-Video Model Handler for RunPod Serverless
Professional MLOps implementation with model caching and error handling
"""
import runpod
import subprocess
import os
import json
import logging
import base64
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Logging konfiguratsiyasi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Konstanta'lar
WAN_DIR = "/app/Wan2.2"
MODEL_DIR = "/app/Wan2.2/Wan2.2-T2V-A14B"
OUTPUT_DIR = "/app/Wan2.2/output"
MODEL_REPO_ID = "Wan-AI/Wan2.2-T2V-A14B"

# Global state - model yuklab olish holati
_model_download_lock = threading.Lock()
_model_download_status = {"status": "not_started", "error": None, "progress": 0}
_model_ready = False


def check_model_exists() -> bool:
    """Model mavjudligini tekshirish"""
    if not os.path.exists(MODEL_DIR):
        return False
    
    try:
        files = os.listdir(MODEL_DIR)
        # Model fayllari mavjud bo'lishi kerak (kamida 5 ta fayl)
        if files and len(files) >= 5:
            # Asosiy fayllarni tekshirish
            required_files = ['config.json', 'model_index.json']
            has_required = any(f in files for f in required_files)
            if has_required:
                logger.info(f"✅ Model mavjud: {MODEL_DIR} ({len(files)} fayl)")
                return True
        logger.warning(f"⚠️  Model papkasi to'liq emas: {len(files) if files else 0} fayl")
        return False
    except Exception as e:
        logger.error(f"❌ Model tekshirishda xatolik: {e}")
        return False


def download_model_with_retry(max_retries: int = 3) -> bool:
    """
    Model yuklab olish retry mexanizmi bilan
    """
    global _model_download_status, _model_ready
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 Model yuklab olish urinishi {attempt}/{max_retries}")
            _model_download_status["status"] = "downloading"
            _model_download_status["progress"] = 0
            _model_download_status["error"] = None
            
            # Disk space tekshirish
            import shutil
            stat = shutil.disk_usage(MODEL_DIR)
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_gb = stat.used / (1024**3)
            logger.info(f"💾 Disk ma'lumotlari:")
            logger.info(f"   - Jami: {total_gb:.2f} GB")
            logger.info(f"   - Ishlatilgan: {used_gb:.2f} GB")
            logger.info(f"   - Bo'sh: {free_gb:.2f} GB")
            
            # Minimum disk space requirement (model ~27GB + overhead)
            min_required_gb = 35
            if free_gb < min_required_gb:
                error_msg = (
                    f"❌ Disk joyi yetarli emas!\n"
                    f"   - Mavjud: {free_gb:.2f} GB\n"
                    f"   - Kerak: {min_required_gb} GB\n"
                    f"   - Qo'shimcha kerak: {min_required_gb - free_gb:.2f} GB\n\n"
                    f"💡 Yechim: RunPod template'da disk hajmini oshiring:\n"
                    f"   1. RunPod Dashboard → Serverless → Templates\n"
                    f"   2. Template'ni tahrirlash\n"
                    f"   3. 'Volume Size' ni kamida 50GB ga oshiring\n"
                    f"   4. Template'ni saqlang va qayta deploy qiling"
                )
                logger.error(error_msg)
                _model_download_status["error"] = error_msg
                return False
            
            # HuggingFace Hub import
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                error_msg = f"HuggingFace Hub import xatolik: {e}"
                logger.error(f"❌ {error_msg}")
                _model_download_status["error"] = error_msg
                return False
            
            # Model papkasini yaratish
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            logger.info(f"📥 Model yuklab olinmoqda: {MODEL_REPO_ID}")
            logger.info(f"📂 Manzil: {MODEL_DIR}")
            logger.info("⏳ Bu uzoq vaqt olishi mumkin (model ~27GB)...")
            
            # Model yuklab olish progress bilan
            logger.info("🚀 Model yuklab olish boshlandi...")
            
            try:
                from huggingface_hub import snapshot_download, HfApi
                import threading
                
                # Progress tracking uchun thread
                download_start_time = time.time()
                last_size = 0
                last_log_time = download_start_time
                
                # Download tugaguncha progress monitoring
                download_active = threading.Event()
                download_active.set()  # Download faol
                
                def monitor_progress():
                    """Model yuklab olish progress'ini kuzatish"""
                    nonlocal last_size, last_log_time
                    logger.info("🔄 Progress monitoring boshlandi...")
                    check_count = 0
                    
                    while download_active.is_set():
                        try:
                            check_count += 1
                            # Model papkasidagi fayllar hajmini hisoblash
                            if os.path.exists(MODEL_DIR):
                                try:
                                    files = [f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f))]
                                    current_size = sum(
                                        os.path.getsize(os.path.join(MODEL_DIR, f))
                                        for f in files
                                    )
                                    
                                    current_time = time.time()
                                    elapsed = current_time - last_log_time
                                    
                                    # Har 3 soniyada yoki hajm o'zgarganda progress ko'rsatish
                                    if elapsed >= 3 or current_size > last_size:
                                        current_size_gb = current_size / (1024**3)
                                        downloaded_since_last = current_size - last_size
                                        speed_mbps = (downloaded_since_last / elapsed / (1024**2)) if elapsed > 0 and downloaded_since_last > 0 else 0
                                        
                                        # Taxminiy jami hajm (~27GB)
                                        estimated_total_gb = 27.0
                                        progress_percent = (current_size_gb / estimated_total_gb) * 100 if estimated_total_gb > 0 else 0
                                        
                                        total_elapsed = current_time - download_start_time
                                        avg_speed = (current_size / total_elapsed / (1024**2)) if total_elapsed > 0 and current_size > 0 else 0
                                        remaining_gb = estimated_total_gb - current_size_gb
                                        eta_minutes = (remaining_gb * 1024 / avg_speed / 60) if avg_speed > 0 else 0
                                        
                                        logger.info(
                                            f"📊 Progress: {progress_percent:.1f}% | "
                                            f"{current_size_gb:.2f} GB / ~{estimated_total_gb:.2f} GB | "
                                            f"Fayllar: {len(files)} | "
                                            f"Tezlik: {speed_mbps:.2f} MB/s | "
                                            f"ETA: {eta_minutes:.1f} min"
                                        )
                                        
                                        _model_download_status["progress"] = int(progress_percent)
                                        last_size = current_size
                                        last_log_time = current_time
                                    
                                    # Birinchi tekshiruvda ham log qilish
                                    if check_count == 1:
                                        logger.info(f"📁 Model papkasi mavjud, fayllar soni: {len(files)}")
                                        
                                except Exception as e:
                                    logger.warning(f"⚠️  Fayllarni hisoblashda xatolik: {e}")
                            else:
                                if check_count == 1:
                                    logger.info("📁 Model papkasi hali yaratilmagan...")
                                    
                        except Exception as e:
                            logger.warning(f"⚠️  Progress monitoring xatolik: {e}")
                        
                        time.sleep(3)  # 3 soniyada bir marta tekshirish
                    
                    logger.info("✅ Progress monitoring yakunlandi")
                
                # Repository ma'lumotlarini olish
                try:
                    logger.info("📥 Repository ma'lumotlari olinmoqda...")
                    api = HfApi()
                    repo_info = api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
                    total_files = len(repo_info.siblings)
                    # Jami hajmini hisoblash
                    total_size_bytes = 0
                    for sibling in repo_info.siblings:
                        if hasattr(sibling, 'rfilename') and sibling.rfilename:
                            try:
                                file_info = api.get_path_info(sibling.rfilename, repo_id=MODEL_REPO_ID)
                                if hasattr(file_info, 'size'):
                                    total_size_bytes += file_info.size
                            except:
                                pass
                    total_size_gb = total_size_bytes / (1024**3)
                    logger.info(f"📋 Repository ma'lumotlari:")
                    logger.info(f"   - Jami fayllar: {total_files}")
                    logger.info(f"   - Taxminiy jami hajm: {total_size_gb:.2f} GB")
                except Exception as e:
                    logger.warning(f"⚠️  Repository ma'lumotlarini olishda xatolik: {e}")
                    total_files = 32  # Default qiymat
                    total_size_gb = 27.0
                
                # Progress monitoring thread'ni boshlash
                logger.info("🔄 Progress monitoring thread boshlandi...")
                progress_thread = threading.Thread(target=monitor_progress, daemon=True)
                progress_thread.start()
                time.sleep(1)  # Thread boshlanishini kutish
                
                # Model yuklab olish
                logger.info("📥 Model fayllari yuklab olinmoqda...")
                logger.info(f"📊 {total_files} ta fayl yuklab olinmoqda (taxminiy hajm: ~{total_size_gb:.2f} GB)...")
                
                try:
                    snapshot_download(
                        repo_id=MODEL_REPO_ID,
                        local_dir=MODEL_DIR,
                        resume_download=True,
                        local_files_only=False,
                        token=None
                    )
                finally:
                    # Progress monitoring thread'ni to'xtatish
                    download_active.clear()
                    time.sleep(3)  # Oxirgi progress log'ini kutish
                    _model_download_status["status"] = "completed"
                
                # Yakuniy progress
                final_size = sum(
                    os.path.getsize(os.path.join(MODEL_DIR, f))
                    for f in os.listdir(MODEL_DIR)
                    if os.path.isfile(os.path.join(MODEL_DIR, f))
                )
                final_size_gb = final_size / (1024**3)
                total_time = time.time() - download_start_time
                avg_speed = (final_size / total_time / (1024**2)) if total_time > 0 else 0
                
                logger.info(f"✅ Model yuklab olish yakunlandi!")
                logger.info(f"📦 Jami hajm: {final_size_gb:.2f} GB")
                logger.info(f"⏱️  Vaqt: {total_time/60:.1f} daqiqa")
                logger.info(f"🚀 O'rtacha tezlik: {avg_speed:.2f} MB/s")
                
            except Exception as e:
                logger.error(f"❌ Model yuklab olishda xatolik: {e}", exc_info=True)
                raise
            
            # Model yuklab olinganidan keyin hajmini ko'rsatish
            try:
                total_size = sum(
                    os.path.getsize(os.path.join(MODEL_DIR, f))
                    for f in os.listdir(MODEL_DIR)
                    if os.path.isfile(os.path.join(MODEL_DIR, f))
                )
                total_size_gb = total_size / (1024**3)
                logger.info(f"📦 Model hajmi: {total_size_gb:.2f} GB")
            except Exception as e:
                logger.warning(f"⚠️  Model hajmini hisoblashda xatolik: {e}")
            
            # Model yuklab olinganini tekshirish
            if check_model_exists():
                files_count = len(os.listdir(MODEL_DIR))
                logger.info(f"✅ Model muvaffaqiyatli yuklab olindi! ({files_count} fayl)")
                _model_download_status["status"] = "completed"
                _model_download_status["progress"] = 100
                _model_ready = True
                return True
            else:
                error_msg = "Model yuklab olindi, lekin tekshirishdan o'tmadi"
                logger.error(f"❌ {error_msg}")
                _model_download_status["error"] = error_msg
                if attempt < max_retries:
                    logger.info(f"🔄 Qayta urinib ko'ramiz...")
                    time.sleep(5)  # 5 soniya kutish
                    continue
                return False
                
        except Exception as e:
            error_msg = f"Model yuklab olishda xatolik (urinish {attempt}/{max_retries}): {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            _model_download_status["error"] = error_msg
            
            if attempt < max_retries:
                wait_time = attempt * 10  # Exponential backoff
                logger.info(f"⏳ {wait_time} soniya kutib, qayta urinib ko'ramiz...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Barcha urinishlar muvaffaqiyatsiz")
                return False
    
    return False


def initialize_model():
    """
    Model yuklab olishni initialization'da qilish (background thread'da)
    """
    global _model_ready, _model_download_status
    
    logger.info("🚀 Model initialization boshlandi...")
    
    # Model allaqachon mavjud bo'lsa
    if check_model_exists():
        logger.info("✅ Model allaqachon mavjud, initialization o'tkazildi")
        _model_ready = True
        _model_download_status["status"] = "ready"
        return True
    
    # Model yuklab olish
    logger.info("📥 Model topilmadi, yuklab olinmoqda...")
    success = download_model_with_retry(max_retries=3)
    
    if success:
        logger.info("✅ Model initialization muvaffaqiyatli yakunlandi")
        _model_ready = True
    else:
        logger.error("❌ Model initialization muvaffaqiyatsiz")
        _model_ready = False
    
    return success


def ensure_model_ready() -> Tuple[bool, Optional[str]]:
    """
    Model tayyorligini tekshirish va agar kerak bo'lsa yuklab olish
    """
    global _model_ready, _model_download_status
    
    # Model allaqachon tayyor bo'lsa
    if _model_ready and check_model_exists():
        return True, None
    
    # Model yuklab olish jarayonida bo'lsa
    if _model_download_status["status"] == "downloading":
        return False, "Model hali yuklab olinmoqda. Iltimos, biroz kutib turing."
    
    # Model yuklab olishni boshlash
    with _model_download_lock:
        if _model_download_status["status"] == "not_started":
            logger.info("🔄 Model yuklab olish boshlandi...")
            success = download_model_with_retry(max_retries=3)
            if success:
                return True, None
            else:
                error = _model_download_status.get("error", "Noma'lum xatolik")
                return False, f"Model yuklab olinmadi: {error}"
        elif _model_download_status["status"] == "completed":
            if check_model_exists():
                _model_ready = True
                return True, None
            else:
                return False, "Model yuklab olindi, lekin tekshirishdan o'tmadi"
        else:
            error = _model_download_status.get("error", "Noma'lum xatolik")
            return False, f"Model yuklab olishda xatolik: {error}"


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Video generation handler - RunPod serverless endpoint
    """
    try:
        logger.info(f"📥 Request qabul qilindi: {json.dumps(event, indent=2)}")
        
        # Model tayyorligini tekshirish
        model_ready, error_msg = ensure_model_ready()
        if not model_ready:
            logger.error(f"❌ Model tayyor emas: {error_msg}")
            return {
                "error": "Model tayyor emas",
                "details": error_msg,
                "model_status": _model_download_status
            }
        
        # Input parametrlar
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "")
        size = input_data.get("size", "1280*720")  # 1280*720 yoki 848*480
        offload = input_data.get("offload_model", True)
        
        # Validation
        if not prompt or not prompt.strip():
            return {"error": "Prompt kiritilmagan yoki bo'sh"}
        
        logger.info(f"🎬 Video yaratilmoqda...")
        logger.info(f"📝 Prompt: {prompt[:100]}...")
        logger.info(f"📐 Razmer: {size}")
        logger.info(f"💾 Offload: {offload}")
        
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
        
        # Video generation
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 daqiqa timeout
            cwd=WAN_DIR
        )
        execution_time = time.time() - start_time
        
        logger.info(f"⏱️  Video generation vaqti: {execution_time:.2f} soniya")
        logger.info(f"📋 Return code: {result.returncode}")
        
        if result.stdout:
            logger.info(f"📋 STDOUT (oxirgi 500 belgi): {result.stdout[-500:]}")
        
        if result.stderr:
            logger.warning(f"⚠️  STDERR (oxirgi 500 belgi): {result.stderr[-500:]}")
        
        if result.returncode != 0:
            return {
                "error": "Video generation failed",
                "return_code": result.returncode,
                "stderr": result.stderr[-1000:] if result.stderr else None,
                "stdout": result.stdout[-1000:] if result.stdout else None
            }
        
        # Yaratilgan video faylini topish
        video_files = list(Path(OUTPUT_DIR).glob("*.mp4"))
        
        if not video_files:
            return {
                "error": "Video fayl topilmadi",
                "output_dir": OUTPUT_DIR,
                "files_in_output": os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else [],
                "stdout": result.stdout[-1000:] if result.stdout else None,
                "stderr": result.stderr[-1000:] if result.stderr else None
            }
        
        # Eng oxirgi yaratilgan video
        latest_video = max(video_files, key=lambda p: p.stat().st_ctime)
        logger.info(f"✅ Video topildi: {latest_video}")
        
        # Video hajmini tekshirish
        video_size_mb = latest_video.stat().st_size / (1024 * 1024)
        logger.info(f"📦 Video hajmi: {video_size_mb:.2f} MB")
        
        # Video ni base64 ga o'girish
        try:
            with open(latest_video, 'rb') as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            logger.info("✅ Video muvaffaqiyatli yaratildi va encode qilindi!")
            
            return {
                "success": True,
                "video_base64": video_base64,
                "video_size_mb": round(video_size_mb, 2),
                "video_filename": latest_video.name,
                "execution_time_seconds": round(execution_time, 2),
                "message": "Video muvaffaqiyatli yaratildi"
            }
        except Exception as e:
            logger.error(f"❌ Video encode qilishda xatolik: {e}", exc_info=True)
            return {
                "error": f"Video encode qilishda xatolik: {str(e)}",
                "video_path": str(latest_video),
                "video_size_mb": round(video_size_mb, 2)
            }
    
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout: 15 daqiqadan ko'p vaqt oldi")
        return {"error": "Video generation timeout (15 min)"}
    
    except Exception as e:
        logger.error(f"❌ Xatolik: {e}", exc_info=True)
        return {"error": str(e), "error_type": type(e).__name__}


if __name__ == "__main__":
    logger.info("🎥 Wan2.2 Video Generation Handler ishga tushdi!")
    logger.info(f"📂 Model papkasi: {MODEL_DIR}")
    logger.info(f"📂 Output papkasi: {OUTPUT_DIR}")
    
    # Model initialization (background thread'da)
    init_thread = threading.Thread(target=initialize_model, daemon=True)
    init_thread.start()
    logger.info("🔄 Model initialization background thread'da boshlandi")
    
    # RunPod serverless start
    runpod.serverless.start({"handler": handler})
