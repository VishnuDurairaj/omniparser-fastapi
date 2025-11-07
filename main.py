import io
import os
import base64
from typing import Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
import torch
from PIL import Image

# Your original demo imports
from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# ----------------------------
# Model init (once at startup)
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep the same paths/args used by your Gradio demo
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/icon_detect/model.pt")
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "florence2")
CAPTION_MODEL_PATH = os.getenv("CAPTION_MODEL_PATH", "weights/icon_caption_florence")

# Load models once
yolo_model = get_yolo_model(model_path=YOLO_MODEL_PATH)
caption_model_processor = get_caption_model_processor(
    model_name=CAPTION_MODEL_NAME,
    model_name_or_path=CAPTION_MODEL_PATH
)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="OmniParser FastAPI",
    version="1.0.0",
    description="FastAPI wrapper of the Gradio demo pipeline",
)

class ParseResponse(BaseModel):
    overlay_png_b64: Optional[str]
    parsed_content_list: dict

@app.get("/healthz")
def healthz():
    return {"ok": True, "device": str(DEVICE)}

def _draw_bbox_config(img_w: int):
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
) -> Tuple[str, str]:
    """
    Returns (overlay_png_b64, parsed_content_list_str)
    to match your Gradio demo behavior.
    """
    draw_bbox_config = _draw_bbox_config(image_pil.size[0])

    # OCR step (same as demo)
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_pil,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
    )
    text, ocr_bbox = ocr_bbox_rslt

    # Main labeling (same as demo)
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

    # Convert labeled image (base64) -> normalized base64 PNG string
    # Your demo returns base64 already (named dino_labled_img). We'll normalize it.
    # If it's bytes, keep; if it's a PIL, re-encode.
    # Here we assume it's base64-encoded bytes/str representing PNG.
    if isinstance(dino_labeled_img_b64, bytes):
        overlay_png_b64 = dino_labeled_img_b64.decode("utf-8")
    elif isinstance(dino_labeled_img_b64, str):
        overlay_png_b64 = dino_labeled_img_b64
    else:
        # Fallback: if it somehow contains raw image bytes, encode to b64
        if isinstance(dino_labeled_img_b64, Image.Image):
            buf = io.BytesIO()
            dino_labeled_img_b64.save(buf, format="PNG")
            overlay_png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            # Last resort: try to interpret as bytes of PNG
            overlay_png_b64 = base64.b64encode(dino_labeled_img_b64).decode("utf-8")

    # Join the parsed_content_list exactly like your demo
    parsed_content_list_str = {f"icon {i}": v for i, v in enumerate(parsed_content_list)}

    return overlay_png_b64, parsed_content_list_str

@app.post("/v1/parse", response_model=ParseResponse)
async def parse_image(
    image_file: UploadFile = File(..., description="Image to parse"),
    box_threshold: float = Query(0.05, ge=0.01, le=1.0),
    iou_threshold: float = Query(0.1, ge=0.01, le=1.0),
    use_paddleocr: bool = Query(True),
    imgsz: int = Query(640, ge=320, le=4096, description="YOLO detection size"),
):
    # Read and decode image
    try:
        raw = await image_file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        overlay_png_b64, parsed_content_list_str = _run_pipeline(
            image,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return ParseResponse(
        overlay_png_b64=overlay_png_b64,
        parsed_content_list=parsed_content_list_str,
    )

# Run directly: `python app/main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=22070)

