# runpod_handler.py
import os
import io
import base64
from typing import Dict, Any, Tuple

import runpod
from PIL import Image
import torch

# Your original demo utils
from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# ----------------------------
# Model init (global, once per pod)
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/icon_detect/model.pt")
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "florence2")
CAPTION_MODEL_PATH = os.getenv("CAPTION_MODEL_PATH", "weights/icon_caption_florence")

yolo_model = get_yolo_model(model_path=YOLO_MODEL_PATH)
caption_model_processor = get_caption_model_processor(
    model_name=CAPTION_MODEL_NAME,
    model_name_or_path=CAPTION_MODEL_PATH,
)

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
    draw_bbox_config = _draw_bbox_config(image_pil.size[0])

    (text, ocr_bbox), _ = check_ocr_box(
        image_pil,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
    )

    labeled_img_b64, label_coordinates, parsed_content_list = get_som_labeled_img(
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

    # normalize to str base64
    if isinstance(labeled_img_b64, bytes):
        overlay_png_b64 = labeled_img_b64.decode("utf-8")
    elif isinstance(labeled_img_b64, str):
        overlay_png_b64 = labeled_img_b64
    else:
        # if bytes of an actual PNG:
        overlay_png_b64 = base64.b64encode(labeled_img_b64).decode("utf-8")

    parsed_content_list_str = "\n".join(
        [f"icon {i}: {v}" for i, v in enumerate(parsed_content_list)]
    )
    return overlay_png_b64, parsed_content_list_str

def _load_image_from_event(inp: Dict[str, Any]) -> Image.Image:
    """
    Input options:
      - "image_b64": base64-encoded image string (PNG/JPG)
      - "image_url": public URL (avoid if not necessary on Serverless; prefer b64)
    """
    if "image_b64" in inp and inp["image_b64"]:
        raw = base64.b64decode(inp["image_b64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # Optional: simple URL fetch (disabled by default for tight sandboxing)
    if "image_url" in inp and inp["image_url"]:
        import requests
        resp = requests.get(inp["image_url"], timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    raise ValueError("Provide 'image_b64' (preferred) or 'image_url' in input.")

def handler(event):
    """
    Runpod Serverless handler.
    event['input'] should contain:
      {
        "image_b64": "<base64 string>",  # preferred
        // "image_url": "https://..."    # optional
        "box_threshold": 0.05,
        "iou_threshold": 0.1,
        "use_paddleocr": true,
        "imgsz": 640
      }
    """
    try:
        inp = event.get("input", {}) or {}
        image = _load_image_from_event(inp)

        box_threshold = float(inp.get("box_threshold", 0.05))
        iou_threshold = float(inp.get("iou_threshold", 0.1))
        use_paddleocr = bool(inp.get("use_paddleocr", True))
        imgsz = int(inp.get("imgsz", 640))

        overlay_png_b64, parsed_content_list_str = _run_pipeline(
            image, box_threshold, iou_threshold, use_paddleocr, imgsz
        )
        return {
            "overlay_png_b64": overlay_png_b64,
            "parsed_content_list": parsed_content_list_str,
            "device": str(DEVICE),
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Start the Runpod event loop
    runpod.serverless.start({"handler": handler})
