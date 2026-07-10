#!/bin/bash
# Wrapper entrypoint: starts Chatterbox TTS, then hands off to ComfyUI's start.sh.
#
# Key insight: /start.sh copies /opt/comfyui-baked → /workspace/runpod-slim/ComfyUI
# on FIRST BOOT only. On subsequent boots the workspace already exists and nothing
# is copied. So we must sync our custom nodes + models AFTER /start.sh has
# potentially created the workspace, but BEFORE ComfyUI loads its nodes.
#
# Solution: We run /start.sh in the background, wait for the workspace to exist,
# kill the ComfyUI it started (which loaded without our nodes), sync everything,
# then restart ComfyUI properly.

set -e

# --- HuggingFace cache persistence ---
HF_CACHE="/workspace/runpod-slim/.cache/huggingface"
mkdir -p "$HF_CACHE"
if [ ! -L /root/.cache/huggingface ]; then
    rm -rf /root/.cache/huggingface
    mkdir -p /root/.cache
    ln -s "$HF_CACHE" /root/.cache/huggingface
fi

# --- Chatterbox TTS voice persistence ---
TTS_DIR="/workspace/runpod-slim/chatterbox-tts"
if [ ! -d "$TTS_DIR" ]; then
    echo "[chatterbox] First boot: setting up TTS workspace..."
    mkdir -p "$TTS_DIR"
    cp -r /opt/chatterbox-tts/voices "$TTS_DIR/"
    cp /opt/chatterbox-tts/server.py "$TTS_DIR/"
else
    cp /opt/chatterbox-tts/server.py "$TTS_DIR/"
fi
mkdir -p "$TTS_DIR/voices/my_voices"

# --- Start Chatterbox TTS server (isolated venv) ---
echo "[chatterbox] Starting TTS server on port 3200..."
cd "$TTS_DIR"
TTS_LOG="/workspace/runpod-slim/chatterbox-tts.log"
CHATTERBOX_PRELOAD=turbo \
    /opt/chatterbox-venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 3200 \
    >> "$TTS_LOG" 2>&1 &
TTS_PID=$!
echo "[chatterbox] TTS server PID: $TTS_PID"

# The TTS server logs to a file, so a crash-on-startup used to look like
# "autostart is broken" instead of an error. Surface it on stdout.
(
    sleep 120
    if kill -0 "$TTS_PID" 2>/dev/null; then
        echo "[chatterbox] TTS server still alive after 120s."
    else
        echo "[chatterbox] ERROR: TTS server died during startup. Last 30 log lines:"
        tail -n 30 "$TTS_LOG" 2>/dev/null || true
    fi
) &

# --- Let /start.sh create workspace + start ComfyUI ---
# We run it in the background, wait for workspace, then fix things up.
/start.sh &
START_PID=$!

# Wait for workspace to be created (max 60s)
COMFY_DIR="/workspace/runpod-slim/ComfyUI"
echo "[startup] Waiting for ComfyUI workspace..."
for i in $(seq 1 60); do
    if [ -d "$COMFY_DIR/custom_nodes" ]; then
        break
    fi
    sleep 1
done

if [ ! -d "$COMFY_DIR/custom_nodes" ]; then
    echo "[startup] ERROR: Workspace not created after 60s, continuing anyway"
    wait $START_PID
    exit 1
fi

echo "[startup] Workspace exists. Syncing custom nodes + models..."

# --- Sync custom nodes (always overwrite to pick up updates) ---
for node_name in ComfyUI-Impact-Pack ComfyUI-Impact-Subpack; do
    src="/opt/comfyui-baked/custom_nodes/$node_name"
    dst="$COMFY_DIR/custom_nodes/$node_name"
    if [ -d "$src" ]; then
        if [ ! -d "$dst" ]; then
            echo "[startup] Installing custom node: $node_name"
            cp -r "$src" "$dst"
        else
            echo "[startup] Custom node exists: $node_name"
        fi
    fi
done

# --- Sync models (create dirs + copy files if missing) ---
for model_path in ultralytics/bbox/face_yolov8m.pt ultralytics/segm/face_yolov8m-seg_2.pt sams/sam_vit_b_01ec64.pth; do
    src="/opt/comfyui-baked/models/$model_path"
    dst="$COMFY_DIR/models/$model_path"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        mkdir -p "$(dirname "$dst")"
        echo "[startup] Copying model: $model_path"
        cp "$src" "$dst"
    fi
done

# --- Pick the interpreter ComfyUI actually runs on ---
# Used for both the dependency sync and the restart, so the two can never diverge.
VENV_DIR="$COMFY_DIR/.venv-cu128"
if [ -x "$VENV_DIR/bin/python" ]; then
    COMFY_PY="$VENV_DIR/bin/python"
else
    COMFY_PY="$(command -v python3)"
fi
echo "[startup] ComfyUI interpreter: $COMFY_PY"

# --- Sync ComfyUI's Python deps to the code in the workspace ---
# ComfyUI lives on the persistent volume, its packages do not. A `git pull` in
# the pod moves the code ahead of the image's packages — that is how a workspace
# ComfyUI ended up importing a comfy_kitchen that was never installed, and
# crashed with `'NoneType' object has no attribute 'Params'` on quantized models.
#
# The stamp lives in the container, NOT in /workspace: a recreated pod gets a
# fresh system Python, and a stamp on the volume would falsely claim the
# packages are already there.
REQS="$COMFY_DIR/requirements.txt"
STAMP="/var/lib/comfyui-reqs.sha256"
if [ -f "$REQS" ]; then
    want="$(sha256sum "$REQS" | cut -d' ' -f1)"
    have=""
    [ -f "$STAMP" ] && have="$(cat "$STAMP")"
    if [ "$want" != "$have" ]; then
        echo "[startup] ComfyUI deps out of sync — installing $REQS ..."
        if "$COMFY_PY" -m pip install --no-cache-dir -r "$REQS"; then
            echo "$want" > "$STAMP"
            echo "[startup] ComfyUI deps installed."
        else
            echo "[startup] ERROR: pip install -r $REQS failed."
            echo "[startup] ComfyUI will still start, but quantized (fp8/fp4) models may fail to load."
        fi
    else
        echo "[startup] ComfyUI deps already in sync."
    fi
else
    echo "[startup] WARNING: $REQS not found, skipping dependency sync."
fi

# --- Restart ComfyUI so it picks up the new nodes + deps ---
# /start.sh already started ComfyUI, but it loaded before any of the above ran.
echo "[startup] Restarting ComfyUI to load custom nodes..."
sleep 5  # Give ComfyUI time to finish initial startup
pkill -f "main.py.*comfyui" 2>/dev/null || pkill -f "main.py.*8188" 2>/dev/null || true
sleep 2

cd "$COMFY_DIR"

ARGS_FILE="/workspace/runpod-slim/comfyui_args.txt"
CUSTOM_ARGS=""
if [ -s "$ARGS_FILE" ]; then
    CUSTOM_ARGS=$(grep -v '^#' "$ARGS_FILE" | tr '\n' ' ')
fi

echo "[startup] Starting ComfyUI with custom nodes: --listen 0.0.0.0 --port 8188 $CUSTOM_ARGS"
"$COMFY_PY" main.py --listen 0.0.0.0 --port 8188 $CUSTOM_ARGS &
COMFY_PID=$!

# Keep container alive
trap "kill $COMFY_PID 2>/dev/null" SIGTERM SIGINT
wait $COMFY_PID || true

echo "[startup] ComfyUI exited. SSH/Jupyter still available."
sleep infinity
