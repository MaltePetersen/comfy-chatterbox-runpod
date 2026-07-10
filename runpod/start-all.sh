#!/bin/bash
# Wrapper entrypoint: starts Chatterbox TTS, then hands off to ComfyUI's start.sh.
#
# /start.sh does two things: it copies /opt/comfyui-baked -> the workspace on the
# FIRST boot only, and it starts ComfyUI on EVERY boot. We need our custom nodes,
# models and Python deps in place, and ComfyUI running with our args + venv.
#
# On an existing workspace (the common case) everything is already on the volume,
# so we sync BEFORE launching /start.sh — ComfyUI then comes up correct the first
# time and never serves a request against stale packages. On a true first boot the
# workspace doesn't exist yet, so we can only sync after /start.sh created it, then
# restart ComfyUI to pick it all up. That slow path runs once per fresh volume.

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

COMFY_DIR="/workspace/runpod-slim/ComfyUI"

# Resolve ComfyUI's interpreter and the matching dependency stamp.
# The venv lives on the persistent volume, so the stamp lives WITH it: after a
# pod recreation (venv survives, packages intact) an unchanged requirements.txt
# skips pip entirely instead of reinstalling for nothing. The glob tolerates a
# base-image CUDA bump renaming .venv-cu128 -> .venv-cu130. Falling back to the
# system interpreter means the stamp belongs in the container, which resets per
# container exactly like /usr/local packages do.
pick_interpreter() {
    local venv
    # sort -V | tail: on a CUDA bump that leaves .venv-cu128 beside a new
    # .venv-cu130, prefer the highest version — a plain head would pick the stale one.
    venv="$(ls -d "$COMFY_DIR"/.venv-* 2>/dev/null | sort -V | tail -n1)"
    if [ -n "$venv" ] && [ -x "$venv/bin/python" ]; then
        COMFY_PY="$venv/bin/python"
        STAMP="$venv/.comfyui-reqs.sha256"
    else
        COMFY_PY="$(command -v python3)"
        STAMP="/var/lib/comfyui-reqs.sha256"
    fi
    echo "[startup] ComfyUI interpreter: $COMFY_PY"
}

sync_custom_nodes() {
    local node_name src dst
    for node_name in ComfyUI-Impact-Pack ComfyUI-Impact-Subpack; do
        src="/opt/comfyui-baked/custom_nodes/$node_name"
        dst="$COMFY_DIR/custom_nodes/$node_name"
        if [ -d "$src" ] && [ ! -d "$dst" ]; then
            echo "[startup] Installing custom node: $node_name"
            cp -r "$src" "$dst"
        fi
    done
}

sync_models() {
    local model_path src dst
    for model_path in ultralytics/bbox/face_yolov8m.pt ultralytics/segm/face_yolov8m-seg_2.pt sams/sam_vit_b_01ec64.pth; do
        src="/opt/comfyui-baked/models/$model_path"
        dst="$COMFY_DIR/models/$model_path"
        if [ -f "$src" ] && [ ! -f "$dst" ]; then
            mkdir -p "$(dirname "$dst")"
            echo "[startup] Copying model: $model_path"
            cp "$src" "$dst"
        fi
    done
}

# Install ComfyUI's requirements.txt into its own interpreter, but only when the
# file's hash differs from the stamp. A git pull in the pod moves the code ahead
# of the installed packages — that is how the workspace ComfyUI ended up importing
# a comfy_kitchen too old to expose the layout classes, crashing quantized (fp8)
# models with `'NoneType' object has no attribute 'Params'`.
sync_deps() {
    local reqs want have=""
    reqs="$COMFY_DIR/requirements.txt"
    if [ ! -f "$reqs" ]; then
        echo "[startup] WARNING: $reqs not found, skipping dependency sync."
        return
    fi
    want="$(sha256sum "$reqs" | cut -d' ' -f1)"
    [ -f "$STAMP" ] && have="$(cat "$STAMP")"
    if [ "$want" = "$have" ]; then
        echo "[startup] ComfyUI deps already in sync."
        return
    fi
    echo "[startup] ComfyUI deps out of sync — installing $reqs ..."
    if "$COMFY_PY" -m pip install --no-cache-dir -r "$reqs"; then
        mkdir -p "$(dirname "$STAMP")"
        echo "$want" > "$STAMP"
        echo "[startup] ComfyUI deps installed."
    else
        echo "[startup] ERROR: pip install -r $reqs failed."
        echo "[startup] ComfyUI will still start, but quantized (fp8/fp4) models may fail to load."
    fi
}

# Distinguish an existing workspace from a true first boot.
if [ -d "$COMFY_DIR/custom_nodes" ]; then
    WORKSPACE_READY=1
else
    WORKSPACE_READY=0
fi

if [ "$WORKSPACE_READY" = "1" ]; then
    # Common case: sync before ComfyUI ever starts, so /start.sh's ComfyUI is
    # already correct and no wrong-deps window exists on port 8188.
    echo "[startup] Existing workspace — syncing before ComfyUI start."
    pick_interpreter
    sync_custom_nodes
    sync_models
    sync_deps
fi

# --- Let /start.sh create the workspace (first boot) and start ComfyUI ---
/start.sh &
START_PID=$!

echo "[startup] Waiting for ComfyUI workspace..."
for i in $(seq 1 60); do
    if [ -d "$COMFY_DIR/custom_nodes" ]; then
        break
    fi
    sleep 1
done

if [ ! -d "$COMFY_DIR/custom_nodes" ]; then
    echo "[startup] ERROR: Workspace not created after 60s, continuing anyway"
    wait "$START_PID"
    exit 1
fi

# --- Model download (volume-less / first-time provisioning) ---
# Runs in the BACKGROUND: ComfyUI comes up immediately and the models trickle in,
# appearing after a UI refresh — no boot blocked on ~40 GB. Idempotent and
# hash-checked, so on a persistent volume it only fetches what's missing. Gated on
# the token, so the image stays usable without it; set SKIP_MODEL_DOWNLOAD=1 to opt out.
MODEL_DL_LOG="/workspace/runpod-slim/model-download.log"
if [ -x /download-models.sh ] && [ -n "${CIVITAI_TOKEN:-}" ] && [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    echo "[startup] Downloading models in background -> $MODEL_DL_LOG"
    ( /download-models.sh "$COMFY_DIR" >> "$MODEL_DL_LOG" 2>&1 ) &
elif [ -z "${CIVITAI_TOKEN:-}" ]; then
    echo "[startup] CIVITAI_TOKEN not set — skipping model download."
fi

if [ "$WORKSPACE_READY" = "0" ]; then
    # True first boot: the workspace was just created, nothing is synced yet, and
    # the venv only exists now. This sync is slow and unavoidably races the interim
    # ComfyUI — but it happens once per fresh volume, not on every boot.
    echo "[startup] First boot — syncing after workspace creation."
    pick_interpreter
    sync_custom_nodes
    sync_models
    sync_deps
fi

# --- Restart ComfyUI under our control: our interpreter, our args, our nodes ---
# /start.sh's ComfyUI used neither our args nor (on first boot) our custom nodes,
# so we take ownership here to trap/wait on it cleanly. On an existing boot the
# deps are already synced, so this is just a fast node-pickup restart.
echo "[startup] Restarting ComfyUI..."
sleep 5  # let the interim ComfyUI finish its own startup before we kill it
pkill -f "main.py.*comfyui" 2>/dev/null || pkill -f "main.py.*8188" 2>/dev/null || true
sleep 2

cd "$COMFY_DIR"

ARGS_FILE="/workspace/runpod-slim/comfyui_args.txt"
CUSTOM_ARGS=""
if [ -s "$ARGS_FILE" ]; then
    CUSTOM_ARGS=$(grep -v '^#' "$ARGS_FILE" | tr '\n' ' ')
fi

echo "[startup] Starting ComfyUI: --listen 0.0.0.0 --port 8188 $CUSTOM_ARGS"
"$COMFY_PY" main.py --listen 0.0.0.0 --port 8188 $CUSTOM_ARGS &
COMFY_PID=$!

# Keep container alive
trap "kill $COMFY_PID 2>/dev/null" SIGTERM SIGINT
wait "$COMFY_PID" || true

echo "[startup] ComfyUI exited. SSH/Jupyter still available."
sleep infinity
