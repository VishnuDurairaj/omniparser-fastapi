#!/usr/bin/env python
"""
Download OmniParser V2 model weights into ./weights/

- Downloads the YOLOv8 detector and Florence-2 caption weights from:
  repo_id = "microsoft/OmniParser-v2.0" (Hugging Face, model repo)
- Places files under:
    weights/icon_detect/{model.pt, model.yaml, train_args.yaml}
    weights/icon_caption_florence/{config.json, generation_config.json, model.safetensors}
- Verifies presence and prints a summary with file sizes.

Run:
    python download_weights.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

REPO_ID = "microsoft/OmniParser-v2.0"
ALLOW_PATTERNS = [
    "icon_detect/train_args.yaml",
    "icon_detect/model.pt",
    "icon_detect/model.yaml",
    "icon_caption/config.json",
    "icon_caption/generation_config.json",
    "icon_caption/model.safetensors",
]

EXPECTED_FILES = [
    "weights/icon_detect/model.pt",
    "weights/icon_detect/model.yaml",
    "weights/icon_detect/train_args.yaml",
    "weights/icon_caption_florence/config.json",
    "weights/icon_caption_florence/generation_config.json",
    "weights/icon_caption_florence/model.safetensors",
]

def ensure_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[info] Installing huggingface_hub...", flush=True)
        # Try current interpreter
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]"])
    # Import after (possibly) installing
    from huggingface_hub import snapshot_download  # noqa: F401
    return True

def human_size(bytes_count: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} PB"

def main():
    print("=== OmniParser V2 Weights Downloader ===")
    root = Path(__file__).resolve().parent
    weights_dir = root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    ensure_hf_hub()
    from huggingface_hub import snapshot_download

    print("[step] Downloading required files from Hugging Face…")
    # Download only needed files to ./weights (no symlinks for maximum Windows compatibility)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(weights_dir),
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERNS,
        token=os.environ.get("HF_TOKEN")
        # You can set token via env HF_TOKEN if needed for private repos (this one is public).
    )

    # Rename icon_caption -> icon_caption_florence (expected by the repo)
    src = weights_dir / "icon_caption"
    dst = weights_dir / "icon_caption_florence"
    if src.exists():
        # If dst exists, remove it (or merge). We’ll replace to be clean.
        if dst.exists():
            print("[info] Removing existing 'weights/icon_caption_florence' to replace it.")
            shutil.rmtree(dst)
        print("[step] Renaming 'weights/icon_caption' -> 'weights/icon_caption_florence'")
        src.rename(dst)

    # Verify everything
    print("[step] Verifying downloaded files…")
    missing = []
    for rel in EXPECTED_FILES:
        p = root / rel
        if not p.exists():
            missing.append(rel)

    if missing:
        print("\n❌ Missing files:")
        for m in missing:
            print("  -", m)
        print("\nIf your network is flaky, just run this script again. "
              "You can also delete the 'weights' folder before retrying.")
        sys.exit(2)

    # Print a small summary with sizes
    print("\n✅ All OmniParser V2 weights are present.\n")
    for rel in EXPECTED_FILES:
        p = root / rel
        try:
            size = p.stat().st_size
            print(f"  {rel}  —  {human_size(size)}")
        except OSError:
            print(f"  {rel}  —  (size unavailable)")

    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
