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

# --- Sync custom nodes from baked image to workspace ---
# /start.sh copies /opt/comfyui-baked to /workspace on first boot,
# but if the workspace already exists (from an older image), new
# custom nodes won't be copied. Force-sync them here.
COMFY_NODES="/workspace/runpod-slim/ComfyUI/custom_nodes"
if [ -d "$COMFY_NODES" ]; then
    for node_dir in /opt/comfyui-baked/custom_nodes/ComfyUI-Impact-Pack \
                     /opt/comfyui-baked/custom_nodes/ComfyUI-Impact-Subpack; do
        node_name=$(basename "$node_dir")
        if [ -d "$node_dir" ] && [ ! -d "$COMFY_NODES/$node_name" ]; then
            echo "[startup] Syncing custom node: $node_name"
            cp -r "$node_dir" "$COMFY_NODES/"
        fi
    done
fi

# Also sync face detection + SAM models
COMFY_MODELS="/workspace/runpod-slim/ComfyUI/models"
if [ -d "$COMFY_MODELS" ]; then
    for model_dir in ultralytics/bbox ultralytics/segm sams; do
        src="/opt/comfyui-baked/models/$model_dir"
        dst="$COMFY_MODELS/$model_dir"
        if [ -d "$src" ] && [ ! -d "$dst" ]; then
            echo "[startup] Syncing models: $model_dir"
            mkdir -p "$dst"
            cp -r "$src/"* "$dst/"
        fi
    done
fi

# Kill any ComfyUI that RunPod may have started before our nodes were ready
echo "[startup] Ensuring clean ComfyUI startup with custom nodes..."
pkill -f "main.py" 2>/dev/null || true
sleep 2

# --- Start Chatterbox TTS server (using isolated venv) ---
echo "[chatterbox] Starting TTS server on port 3200 (venv: /opt/chatterbox-venv)..."
cd "$TTS_DIR"
CHATTERBOX_PRELOAD=turbo \
    /opt/chatterbox-venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 3200 \
    >> /workspace/runpod-slim/chatterbox-tts.log 2>&1 &
echo "[chatterbox] TTS server PID: $!"

# --- Hand off to original ComfyUI start script ---
# This starts ComfyUI fresh — it will now discover Impact Pack + Subpack nodes.
exec /start.sh
