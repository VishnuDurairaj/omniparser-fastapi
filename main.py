import io
import os
import base64
import gc
from typing import Optional, Tuple
from contextlib import asynccontextmanager

# ----------------------------
# CRITICAL: Set threading limits BEFORE any imports
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch

# Set PyTorch settings immediately
torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass

# GPU optimization settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image

# Your original demo imports
from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# ----------------------------
# Configuration
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/icon_detect/model.pt")
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "florence2")
CAPTION_MODEL_PATH = os.getenv("CAPTION_MODEL_PATH", "weights/icon_caption_florence")

# Global model storage - will be loaded in lifespan
yolo_model = None
caption_model_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle properly"""
    global yolo_model, caption_model_processor
    
    print(f"🚀 Starting OmniParser on device: {DEVICE}")
    
    # Load models
    print("📦 Loading YOLO model...")
    yolo_model = get_yolo_model(model_path=YOLO_MODEL_PATH)
    
    print("📦 Loading caption model...")
    caption_model_processor = get_caption_model_processor(
        model_name=CAPTION_MODEL_NAME,
        model_name_or_path=CAPTION_MODEL_PATH,
        device=DEVICE
    )
    
    # Move models to GPU and set to eval mode
    if hasattr(yolo_model, 'to'):
        yolo_model = yolo_model.to(DEVICE)
    if hasattr(yolo_model, 'eval'):
        yolo_model.eval()
    
    if caption_model_processor and 'model' in caption_model_processor:
        caption_model_processor['model'].eval()
    
    # Clear any accumulated memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("✅ Models loaded successfully")
    print(f"💾 GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if torch.cuda.is_available() else "")
    
    yield
    
    # Cleanup on shutdown
    print("🧹 Cleaning up models...")
    yolo_model = None
    caption_model_processor = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("✅ Shutdown complete")


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="OmniParser FastAPI",
    version="2.0.0",
    description="GPU-Optimized FastAPI wrapper for OmniParser",
    lifespan=lifespan,
)


class ParseResponse(BaseModel):
    overlay_png_b64: Optional[str]
    parsed_content_list: dict
    processing_time_ms: Optional[float] = None


@app.get("/healthz")
def healthz():
    """Health check endpoint with GPU stats"""
    health = {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": yolo_model is not None and caption_model_processor is not None,
    }
    
    if torch.cuda.is_available():
        health.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}",
            "gpu_memory_reserved_gb": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}",
            "gpu_memory_free_gb": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f}",
        })
    
    return health


def _draw_bbox_config(img_w: int):
    """Generate bbox drawing configuration"""
    box_overlay_ratio = img_w / 3200
    return {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }


def _run_pipeline(
    image_pil: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int,
) -> Tuple[str, dict]:
    """
    Run the OmniParser pipeline with GPU optimization
    """
    import time
    
    start_time = time.time()
    
    # Ensure we're in inference mode
    with torch.inference_mode():
        draw_bbox_config = _draw_bbox_config(image_pil.size[0])

        # OCR step
        ocr_start = time.time()
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_pil,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": 0.9},
            use_paddleocr=use_paddleocr,
        )
        text, ocr_bbox = ocr_bbox_rslt
        print(f"⏱️  OCR time: {(time.time() - ocr_start)*1000:.2f}ms")

        # Main labeling
        label_start = time.time()
        dino_labeled_img_b64, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_pil,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )
        print(f"⏱️  Labeling time: {(time.time() - label_start)*1000:.2f}ms")

    # Convert labeled image to base64 PNG string
    if isinstance(dino_labeled_img_b64, bytes):
        overlay_png_b64 = dino_labeled_img_b64.decode("utf-8")
    elif isinstance(dino_labeled_img_b64, str):
        overlay_png_b64 = dino_labeled_img_b64
    else:
        if isinstance(dino_labeled_img_b64, Image.Image):
            buf = io.BytesIO()
            dino_labeled_img_b64.save(buf, format="PNG")
            overlay_png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            overlay_png_b64 = base64.b64encode(dino_labeled_img_b64).decode("utf-8")

    # Format parsed content
    parsed_content_dict = {}
    for i, item in enumerate(parsed_content_list):
        if isinstance(item, dict):
            parsed_content_dict[f"element_{i}"] = item
        else:
            parsed_content_dict[f"element_{i}"] = str(item)
    
    total_time = (time.time() - start_time) * 1000
    print(f"⏱️  Total pipeline time: {total_time:.2f}ms")

    return overlay_png_b64, parsed_content_dict

from starlette.concurrency import run_in_threadpool

@app.post("/v1/parse", response_model=ParseResponse)
async def parse_image(
    image_file: UploadFile = File(..., description="Image to parse"),
    box_threshold: float = Query(0.05, ge=0.01, le=1.0, description="Detection confidence threshold"),
    iou_threshold: float = Query(0.1, ge=0.01, le=1.0, description="IoU threshold for NMS"),
    use_paddleocr: bool = Query(True, description="Use PaddleOCR (GPU) instead of EasyOCR"),
    imgsz: int = Query(640, ge=320, le=4096, description="YOLO detection size"),
):
    """
    Parse an image to detect and label UI elements.
    
    Returns:
    - overlay_png_b64: Base64 encoded annotated image
    - parsed_content_list: Detected elements with bounding boxes and labels
    - processing_time_ms: Total processing time in milliseconds
    """
    import time
    
    start_time = time.time()
    
    # Validate models are loaded
    if yolo_model is None or caption_model_processor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Server may still be starting up."
        )
    
    # Read and decode image
    try:
        raw = await image_file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        print(f"📷 Image size: {image.size}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run pipeline
    try:
        overlay_png_b64, parsed_content_dict = await run_in_threadpool(
            _run_pipeline,
            image,
            box_threshold,
            iou_threshold,
            use_paddleocr,
            imgsz,
        )
    except Exception as e:
        # Clear GPU cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import traceback
        error_detail = f"Pipeline error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ Error: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
    
    # Optional: Clear GPU cache after each request to prevent memory buildup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    processing_time = (time.time() - start_time) * 1000

    return ParseResponse(
        overlay_png_b64=overlay_png_b64,
        parsed_content_list=parsed_content_dict,
        processing_time_ms=processing_time,
    )


@app.get("/")
def root():
    """API documentation redirect"""
    return {
        "message": "OmniParser FastAPI Server",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/healthz",
        "parse_endpoint": "/v1/parse",
    }


# Run directly: `python main.py`
if __name__ == "__main__":
    import uvicorn
    
    # Optimized uvicorn settings
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=22070,
        workers=1,  # Single worker to avoid loading models multiple times
        reload=False,
        log_level="info",
        access_log=True,
        # limit_concurrency=5,  # Limit concurrent requests to prevent memory spikes
        timeout_keep_alive=5,
    )
