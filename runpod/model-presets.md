# MODEL_LIST-Presets

Fertige Werte für das Env-Var **`MODEL_LIST`** (siehe `download-models.sh`). Setzt
du eines davon an der RunPod-Pod-Env und rebootest, lädt der Pod genau diese
Dateien statt des eingebauten Manifests — kein Image-Rebuild nötig.

- **Eine** Liste genügt: Civitai und HuggingFace gemischt, die Auth wird am Host
  automatisch gewählt.
- Alle drei Presets brauchen weiterhin `CIVITAI_TOKEN` in der Pod-Env (jedes
  enthält Civitai-Dateien). Die HF-Dateien laufen tokenlos.
- Format pro Eintrag: `subdir|filename|url`, Einträge mit `;` getrennt (einzeilig,
  wie das RunPod-Env-Feld es will).
- Ist `MODEL_LIST` nicht gesetzt, greift das eingebaute Manifest (beide
  Illustrious).

## 1) Alle Modelle (Illustrious + Krea + Anima) — 11 Dateien

```
checkpoints|hassakuXL_illustrious_v34.safetensors|https://civitai.red/api/download/models/2615702?fileId=2503211;checkpoints|waiIllustrious_v170.safetensors|https://civitai.red/api/download/models/2883731?fileId=2763986;checkpoints|oneObsession_v23.safetensors|https://civitai.red/api/download/models/3118448?fileId=2998810;diffusion_models|moodyKrea2Mix_v30_fp8.safetensors|https://civitai.red/api/download/models/3100032?fileId=2979791;diffusion_models|novaAnimeAM_v30.safetensors|https://civitai.red/api/download/models/3086321?fileId=2965742;diffusion_models|miaomiaoHarem_anima14.safetensors|https://civitai.red/api/download/models/3107122?fileId=2987069;text_encoders|anima_baseV10_txt.safetensors|https://civitai.red/api/download/models/3107122?fileId=2987064;text_encoders|Huihui-Qwen3-VL-4B-abliterated-fp8_scaled.safetensors|https://huggingface.co/ahmed22xa/Huihui-Qwen3-VL-4B-Instruct-abliterated-comfy/resolve/main/Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors;vae|qwen_image_vae.safetensors|https://civitai.red/api/download/models/2110009?fileId=2004692;vae|ae.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors;text_encoders|qwen_3_4b.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
```

## 2) Nur Illustrious — 3 Dateien

All-in-one-Checkpoints (UNet+CLIP+VAE), keine separaten Encoder/VAE nötig.

```
checkpoints|hassakuXL_illustrious_v34.safetensors|https://civitai.red/api/download/models/2615702?fileId=2503211;checkpoints|waiIllustrious_v170.safetensors|https://civitai.red/api/download/models/2883731?fileId=2763986;checkpoints|oneObsession_v23.safetensors|https://civitai.red/api/download/models/3118448?fileId=2998810
```

## 3) Nur Anima — 5 Dateien

Zwei Diffusion-Modelle + beide Text-Encoder + VAE. Der HF-Encoder
(`Huihui-Qwen3-VL-abliterated`) ist die unzensierte Alternative zu
`anima_baseV10_txt` — brauchst du ihn nicht, lösch den `text_encoders|Huihui...`-
Block raus.

```
diffusion_models|novaAnimeAM_v30.safetensors|https://civitai.red/api/download/models/3086321?fileId=2965742;diffusion_models|miaomiaoHarem_anima14.safetensors|https://civitai.red/api/download/models/3107122?fileId=2987069;text_encoders|anima_baseV10_txt.safetensors|https://civitai.red/api/download/models/3107122?fileId=2987064;text_encoders|Huihui-Qwen3-VL-4B-abliterated-fp8_scaled.safetensors|https://huggingface.co/ahmed22xa/Huihui-Qwen3-VL-4B-Instruct-abliterated-comfy/resolve/main/Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors;vae|qwen_image_vae.safetensors|https://civitai.red/api/download/models/2110009?fileId=2004692
```
