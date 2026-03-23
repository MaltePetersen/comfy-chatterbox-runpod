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
# The base image's numpy 2.x works fine with chatterbox at runtime.
RUN pip install --no-cache-dir --no-deps chatterbox-tts && \
    pip install --no-cache-dir \
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
