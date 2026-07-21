# Extends the RunPod ComfyUI base image with Chatterbox TTS server.
# ComfyUI runs on port 8188, Chatterbox TTS on port 3200.
#
# Build (this is what CI pushes, and what the RunPod template pulls):
#   docker build -t maltepetersen/comfyui-chatterbox:latest -f runpod/Dockerfile .
#
# The CUDA version comes from the base image — runpod/comfyui:latest currently
# ships cu130 (Blackwell/RTX 5090). Nothing below hardcodes it.

ARG BASE_IMAGE=runpod/comfyui:latest
FROM ${BASE_IMAGE}

# ==========================================
# Chatterbox TTS — fully isolated venv (NO --system-site-packages).
#
# Chatterbox needs numpy 1.x, ComfyUI needs numpy>=2.0. A venv with
# --system-site-packages only *shadows* the system packages it installs
# itself — everything else falls through to /usr/local/lib/.../dist-packages.
# That is how a system scipy built against numpy 2 got pulled into a venv
# pinned to numpy 1.26.4 (transformers -> modeling_llama -> loss_utils ->
# scipy -> np.long -> AttributeError at startup).
#
# So: the venv gets its own torch, scipy and numpy. Any `pip install` in the
# system Python (ComfyUI updates, custom nodes) is now invisible to Chatterbox.
#
# Torch is not pinned to a literal version — we install whatever the base image
# already runs, from the matching CUDA wheel index. That keeps the two Torch
# builds in lockstep across base image bumps (cu128 -> cu130 -> ...) without
# editing this file.
#
# The constraints file is the safety net: without it a transitive requirement
# could quietly swap our CUDA torch for a CPU wheel off PyPI, or drag numpy 2
# back in. With it, such a resolution fails the build instead. chatterbox-tts
# goes in with --no-deps for the same reason.
#
# We deliberately do NOT install torchcodec. On torch>=2.9 torchaudio.save()
# delegates to torchcodec, whose prebuilt libs never matched this base image's
# bleeding-edge torch (undefined symbol torch_dtype_float4_e2m1fn_x2) or its
# FFmpeg 6 — every core lib failed to load, surfacing as a 500 on /tts. server.py
# writes WAVs via soundfile (libsndfile) instead, so torchcodec is unnecessary.
# ==========================================
RUN set -eux; \
    TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"; \
    TA_VER="$(python3 -c 'import torchaudio; print(torchaudio.__version__)' 2>/dev/null || true)"; \
    CU_TAG="${TORCH_VER##*+}"; \
    if [ "$CU_TAG" = "$TORCH_VER" ]; then \
        TORCH_INDEX="https://pypi.org/simple"; \
    else \
        TORCH_INDEX="https://download.pytorch.org/whl/${CU_TAG}"; \
    fi; \
    echo "base image: torch=${TORCH_VER} torchaudio=${TA_VER:-<absent>} index=${TORCH_INDEX}"; \
    python3 -m venv /opt/chatterbox-venv; \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir --upgrade pip; \
    if [ -n "$TA_VER" ]; then TA_SPEC="torchaudio==${TA_VER}"; else TA_SPEC="torchaudio"; fi; \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir \
        --index-url "$TORCH_INDEX" "torch==${TORCH_VER}" "$TA_SPEC"; \
    printf 'numpy==1.26.4\nscipy==1.12.0\ntorch==%s\n' "$TORCH_VER" > /tmp/constraints.txt; \
    if [ -n "$TA_VER" ]; then printf 'torchaudio==%s\n' "$TA_VER" >> /tmp/constraints.txt; fi; \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir -c /tmp/constraints.txt \
        numpy==1.26.4 \
        scipy==1.12.0 \
        librosa==0.11.0 \
        soundfile \
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
        uvicorn[standard]; \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir --no-deps chatterbox-tts; \
    rm -f /tmp/constraints.txt

# Fail the build, not the pod, when the dependency set is broken.
COPY runpod/verify_venv.py /tmp/verify_venv.py
RUN /opt/chatterbox-venv/bin/python /tmp/verify_venv.py && rm /tmp/verify_venv.py

# ==========================================
# ComfyUI custom nodes — uses system Python (numpy 2.x is fine here)
# ==========================================
RUN cd /opt/comfyui-baked/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && \
    pip install --no-cache-dir -r requirements.txt && \
    PYTHONPATH=/opt/comfyui-baked python install.py || echo "install.py warning (non-fatal)" && \
    cd /opt/comfyui-baked/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git && \
    cd ComfyUI-Impact-Subpack && \
    pip install --no-cache-dir -r requirements.txt

# Download face detection + SAM models for FaceDetailer
RUN mkdir -p /opt/comfyui-baked/models/ultralytics/bbox && \
    mkdir -p /opt/comfyui-baked/models/ultralytics/segm && \
    mkdir -p /opt/comfyui-baked/models/sams && \
    curl -sL -o /opt/comfyui-baked/models/ultralytics/bbox/face_yolov8m.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" && \
    curl -sL -o /opt/comfyui-baked/models/ultralytics/segm/face_yolov8m-seg_2.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m-seg_2.pt" && \
    curl -sL -o /opt/comfyui-baked/models/sams/sam_vit_b_01ec64.pth \
      "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# Copy TTS server and voice presets
COPY server.py /opt/chatterbox-tts/server.py
COPY voices/ /opt/chatterbox-tts/voices/

# HuggingFace model cache: symlink to workspace so models persist across pod restarts.
# The actual symlink is created at runtime in start-all.sh (workspace volume isn't
# available at build time).

# Startup wrapper: launches TTS server, then hands off to the original ComfyUI start.sh
COPY runpod/start-all.sh /start-all.sh
COPY runpod/download-models.sh /download-models.sh
RUN chmod +x /start-all.sh /download-models.sh

EXPOSE 3200

ENTRYPOINT ["/start-all.sh"]
