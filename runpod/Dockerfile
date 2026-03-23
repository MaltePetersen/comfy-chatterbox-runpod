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
# The base image already has PyTorch + torchaudio installed, so chatterbox-tts
# will reuse those. We install into the system Python (3.12) which the ComfyUI
# venv inherits via --system-site-packages.
RUN pip install --no-cache-dir \
    chatterbox-tts \
    fastapi \
    uvicorn[standard]

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
