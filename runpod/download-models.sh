#!/bin/bash
# Downloads the model set into a ComfyUI models tree, idempotently.
#
# Each file is skipped when it already exists with the right SHA256, so re-running
# is cheap and a half-finished download resumes instead of restarting. Safe to run
# by hand in the pod, or to wire into start-all.sh for a volume-less setup.
#
#   CIVITAI_TOKEN=xxxx bash runpod/download-models.sh [COMFY_DIR]
#
# COMFY_DIR defaults to the workspace ComfyUI. Civitai needs the token; the
# HuggingFace files are public. Get a token at civitai.com/user/account.
#
# Folder logic — why some checkpoints land in checkpoints/ and some in
# diffusion_models/:
#   * Illustrious/SDXL checkpoints are all-in-one (UNet+CLIP+VAE)  -> checkpoints/
#   * Krea 2 and Anima ship the diffusion model ONLY               -> diffusion_models/
#     and need a separate text encoder + VAE (both included below) to run.
#
# The two model families use different text-encoder + VAE stacks:
#   * Krea 2 (Moody): qwen_3_4b text encoder + z-image VAE (ae.safetensors)
#   * Anima (Nova, MiaoMiao): anima_baseV10_txt + Qwen Image VAE
# The abliterated Qwen3-VL is the uncensored alternative to the Anima text encoder.

set -u

COMFY_DIR="${1:-/workspace/runpod-slim/ComfyUI}"
MODELS="$COMFY_DIR/models"

# manifest rows: subdir | filename | sha256 | url
# sha256 "-" means unverified (HuggingFace CDN files without a published hash).
MANIFEST=$(cat <<'EOF'
checkpoints|hassakuXL_illustrious_v34.safetensors|1618EDB443D9C641FB01C4961F5875D5F01B1A851FD9F2EA64623CEAC82257E0|https://civitai.red/api/download/models/2615702?fileId=2503211
checkpoints|waiIllustrious_v170.safetensors|F116B0C78FF441467B0CDC8F1936E1ED18EA31E9997C7B132B1B8DB533F0BD04|https://civitai.red/api/download/models/2883731?fileId=2763986
diffusion_models|moodyKrea2Mix_v30_fp8.safetensors|49591A46EF95D21635156408C91F5281BEFEB090284FC243D44BC4B74D783A9E|https://civitai.red/api/download/models/3100032?fileId=2979791
diffusion_models|novaAnimeAM_v30.safetensors|34C30FA880B631C701B256A9B3A0F2479F7F3BC84E30A55C67E9B9AE05AC18B1|https://civitai.red/api/download/models/3086321?fileId=2965742
diffusion_models|miaomiaoHarem_anima14.safetensors|9542FDD6DB4F579B276A3FA6E26955E7E377A42BA65EFD237DCDA0B14E044D1B|https://civitai.red/api/download/models/3107122?fileId=2987069
text_encoders|anima_baseV10_txt.safetensors|CD2A512003E2F9F3CD3C32A9C3573F820BB28C940F73C57B1DDAA983D9223EBA|https://civitai.red/api/download/models/3107122?fileId=2987064
text_encoders|Huihui-Qwen3-VL-4B-abliterated-fp8_scaled.safetensors|-|https://huggingface.co/ahmed22xa/Huihui-Qwen3-VL-4B-Instruct-abliterated-comfy/resolve/main/Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors
vae|qwen_image_vae.safetensors|A70580F0213E67967EE9C95F05BB400E8FB08307E017A924BF3441223E023D1F|https://civitai.red/api/download/models/2110009?fileId=2004692
vae|ae.safetensors|-|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors
text_encoders|qwen_3_4b.safetensors|-|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
EOF
)

have_aria2c() { command -v aria2c >/dev/null 2>&1; }

# Download url -> dest, resuming, with the Civitai token as a header (curl and
# aria2c both drop it on the cross-host redirect to the CDN, so it never leaks
# into storage URLs). aria2c opens many connections per file, which is what makes
# Civitai's throttled single connections fast; curl is the fallback.
fetch() {
    local url="$1" dest="$2" auth=()
    case "$url" in
        *civitai*)
            if [ -z "${CIVITAI_TOKEN:-}" ]; then
                echo "  ERROR: CIVITAI_TOKEN not set — needed for $url" >&2
                return 1
            fi
            auth=(Authorization "Bearer $CIVITAI_TOKEN")
            ;;
        *huggingface*)
            # Optional: public repos need no token, gated ones do.
            [ -n "${HF_TOKEN:-}" ] && auth=(Authorization "Bearer $HF_TOKEN")
            ;;
    esac
    if have_aria2c; then
        local args=(--continue=true --max-connection-per-server=16 --split=16
                    --min-split-size=8M --max-tries=5 --retry-wait=5
                    --dir="$(dirname "$dest")" --out="$(basename "$dest")"
                    --allow-overwrite=true --auto-file-renaming=false --summary-interval=10)
        [ "${#auth[@]}" -gt 0 ] && args+=(--header="${auth[0]}: ${auth[1]}")
        aria2c "${args[@]}" "$url"
    else
        local args=(-L --fail --retry 5 --retry-delay 5 -C - -o "$dest")
        [ "${#auth[@]}" -gt 0 ] && args+=(-H "${auth[0]}: ${auth[1]}")
        curl "${args[@]}" "$url"
    fi
}

verify() {
    local dest="$1" want="$2"
    [ "$want" = "-" ] && return 0
    local got
    got="$(sha256sum "$dest" | cut -d' ' -f1)"
    [ "${got,,}" = "${want,,}" ]
}

echo "[models] target: $MODELS"
echo "[models] downloader: $(have_aria2c && echo aria2c || echo curl)"

ok=0; skip=0; fail=0; failed_names=""
while IFS='|' read -r subdir name sha url; do
    [ -z "$subdir" ] && continue
    dest="$MODELS/$subdir/$name"
    mkdir -p "$MODELS/$subdir"

    if [ -f "$dest" ] && verify "$dest" "$sha"; then
        echo "[models] OK (cached): $subdir/$name"
        skip=$((skip+1))
        continue
    fi

    echo "[models] downloading: $subdir/$name"
    if ! fetch "$url" "$dest"; then
        echo "[models] FAILED download: $subdir/$name" >&2
        fail=$((fail+1)); failed_names="$failed_names $name"
        continue
    fi
    if ! verify "$dest" "$sha"; then
        echo "[models] FAILED checksum: $subdir/$name (deleting corrupt file)" >&2
        rm -f "$dest"
        fail=$((fail+1)); failed_names="$failed_names $name"
        continue
    fi
    echo "[models] done: $subdir/$name"
    ok=$((ok+1))
done <<< "$MANIFEST"

echo "[models] summary: $ok downloaded, $skip cached, $fail failed"
if [ "$fail" -gt 0 ]; then
    echo "[models] failed:$failed_names" >&2
    exit 1
fi
