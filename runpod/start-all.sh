#!/bin/bash
# Wrapper entrypoint: starts Chatterbox TTS, then hands off to ComfyUI's start.sh.

# --- HuggingFace cache persistence ---
# Symlink HF cache to workspace volume so model downloads survive pod restarts.
HF_CACHE="/workspace/runpod-slim/.cache/huggingface"
mkdir -p "$HF_CACHE"
if [ ! -L /root/.cache/huggingface ]; then
    rm -rf /root/.cache/huggingface
    mkdir -p /root/.cache
    ln -s "$HF_CACHE" /root/.cache/huggingface
fi

# --- Chatterbox TTS voice persistence ---
# Copy bundled voices to workspace on first boot, then use workspace copy.
TTS_DIR="/workspace/runpod-slim/chatterbox-tts"
if [ ! -d "$TTS_DIR" ]; then
    echo "[chatterbox] First boot: setting up TTS workspace..."
    mkdir -p "$TTS_DIR"
    cp -r /opt/chatterbox-tts/voices "$TTS_DIR/"
    cp /opt/chatterbox-tts/server.py "$TTS_DIR/"
else
    # Always update server.py from image (picks up code changes on image rebuild)
    cp /opt/chatterbox-tts/server.py "$TTS_DIR/"
fi

# Ensure my_voices directory exists for uploads
mkdir -p "$TTS_DIR/voices/my_voices"

# --- Start Chatterbox TTS server (using isolated venv) ---
echo "[chatterbox] Starting TTS server on port 3200 (venv: /opt/chatterbox-venv)..."
cd "$TTS_DIR"
CHATTERBOX_PRELOAD=turbo \
    /opt/chatterbox-venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 3200 \
    >> /workspace/runpod-slim/chatterbox-tts.log 2>&1 &
echo "[chatterbox] TTS server PID: $!"

# --- Hand off to original ComfyUI start script ---
exec /start.sh
