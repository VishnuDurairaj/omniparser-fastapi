# GPU base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git git-lfs curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# ---- app code ----
WORKDIR /app

# If you keep OmniParser code in this repo, copy it in:
# (Assumes your project structure brings in `util/` and related files)
COPY . /app

# If you prefer to clone the official repo instead, uncomment:
# RUN git clone https://github.com/microsoft/OmniParser.git /app

# Python deps
# If OmniParser includes its own requirements.txt, install that first
# and then any additional runtime deps (runpod, requests).
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip install runpod requests pillow

# ---- weights ----
# Pull the v2 weights into ./weights and rename icon_caption -> icon_caption_florence
RUN mkdir -p weights && \
    bash -lc '\
      python3 -m pip install "huggingface_hub>=0.23"; \
      huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/train_args.yaml --local-dir weights; \
      huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.pt --local-dir weights; \
      huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.yaml --local-dir weights; \
      huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/config.json --local-dir weights; \
      huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/generation_config.json --local-dir weights; \
      huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/model.safetensors --local-dir weights; \
      if [ -d weights/icon_caption ]; then mv weights/icon_caption weights/icon_caption_florence; fi \
    '

# Useful envs (used by handler optionally)
ENV YOLO_MODEL_PATH="weights/icon_detect/model.pt" \
    CAPTION_MODEL_NAME="florence2" \
    CAPTION_MODEL_PATH="weights/icon_caption_florence"

# No ports; serverless handler pulls jobs from Runpod
CMD ["python3", "-u", "runpod_handler.py"]
