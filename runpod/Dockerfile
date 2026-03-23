# Extends the RunPod ComfyUI base image with Chatterbox TTS server.
# ComfyUI runs on port 8188, Chatterbox TTS on port 3200.
#
# Build:
#   docker build -t maltepetersen/comfyui-chatterbox:latest -f runpod/Dockerfile .
#
# For RTX 5090 (CUDA 13.0):
#   docker build --build-arg BASE_IMAGE=runpod/comfyui:cuda13.0 \
#     -t maltepetersen/comfyui-chatterbox:cuda13.0 -f runpod/Dockerfile .

ARG BASE_IMAGE=runpod/comfyui:latest
FROM ${BASE_IMAGE}

# Install Chatterbox TTS and FastAPI server dependencies.
# The base image has PyTorch, torchaudio, and numpy already installed.
# chatterbox-tts pins numpy<1.26 but no binary wheel exists for that on Python 3.12,
# so we install chatterbox-tts with --no-deps and add its other dependencies separately.
# Downgrade numpy to 1.26.4 (last 1.x with cp312 wheels) — numpy 2.x causes
# dtype mismatches (float64 vs float32) in chatterbox's s3tokenizer.
RUN pip install --no-cache-dir --no-deps chatterbox-tts && \
    pip install --no-cache-dir "numpy==1.26.4" \
    librosa==0.11.0 \
    s3tokenizer \
    transformers==4.46.3 \
    diffusers==0.29.0 \
    resemble-perth==1.0.1 \
    conformer==0.3.2 \
    safetensors==0.5.3 \
    spacy-pkuseg \
    pykakasi==2.3.0 \
    pyloudnorm \
    omegaconf \
    fastapi \
    python-multipart \
    torchcodec \
    uvicorn[standard]

# Install ComfyUI custom nodes for image quality
# Impact Pack: FaceDetailer, SAM-based face detection/refinement
# Note: install.py needs PYTHONPATH to find ComfyUI modules at build time.
# We also install the subpack's deps separately (no || true hiding errors).
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && \
    pip install --no-cache-dir -r requirements.txt && \
    PYTHONPATH=/comfyui python install.py || echo "install.py warning (non-fatal)" && \
    cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git && \
    cd ComfyUI-Impact-Subpack && \
    pip install --no-cache-dir -r requirements.txt

# Download face detection + SAM models for FaceDetailer
RUN mkdir -p /comfyui/models/ultralytics/bbox && \
    mkdir -p /comfyui/models/ultralytics/segm && \
    mkdir -p /comfyui/models/sams && \
    curl -sL -o /comfyui/models/ultralytics/bbox/face_yolov8m.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" && \
    curl -sL -o /comfyui/models/ultralytics/segm/face_yolov8m-seg_2.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m-seg_2.pt" && \
    curl -sL -o /comfyui/models/sams/sam_vit_b_01ec64.pth \
      "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# Copy TTS server and voice presets
COPY server.py /opt/chatterbox-tts/server.py
COPY voices/ /opt/chatterbox-tts/voices/

# HuggingFace model cache: symlink to workspace so models persist across pod restarts.
# The actual symlink is created at runtime in start-all.sh (workspace volume isn't
# available at build time).

# Startup wrapper: launches TTS server, then hands off to the original ComfyUI start.sh
COPY runpod/start-all.sh /start-all.sh
RUN chmod +x /start-all.sh

EXPOSE 3200

ENTRYPOINT ["/start-all.sh"]
