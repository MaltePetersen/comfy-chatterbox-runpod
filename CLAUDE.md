# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-file FastAPI server (`server.py`) that wraps [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) as a drop-in replacement for the old local-TTS (Coqui) server. The API contract matches the old server so existing frontends work without changes.

## Commands

### Install & Run (RunPod / Linux GPU)
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 3200
```

### Install & Run (Mac Apple Silicon)
```bash
conda create -n chatterbox python=3.11 -y
conda activate chatterbox
bash install_mac.sh
uvicorn server:app --host 0.0.0.0 --port 3200
```

### Environment Variables
- `CHATTERBOX_DEVICE` — override device detection (`cuda`, `cpu`, `mps`). MPS is disabled by default due to Chatterbox tensor allocation issues.

## Architecture

The entire server is in `server.py` (~194 lines). Key design decisions:

- **Three Chatterbox models**, lazy-loaded and cached in a `models` dict:
  - `turbo` (ChatterboxTurboTTS) — default, fastest, supports `[laugh]` syntax
  - `standard` (ChatterboxTTS) — supports `exaggeration` and `cfg_weight` params
  - `multilingual` (ChatterboxMultilingualTTS) — auto-selected for non-English `lang`
- **Voice resolution** maps the old XTTS speaker names to `.wav` files in `voices/presets/`. Unrecognized names fall back to `nicole.wav`.
- **Apple Silicon patch** at startup: replaces `PerthImplicitWatermarker` with `DummyWatermarker` when the ARM binary is missing (no audio quality impact).

### Voice directories
- `voices/presets/` — 14 bundled voice reference WAVs (mapped to XTTS speaker names)
- `voices/my_voices/` — user-uploaded voice files via `/upload` endpoint

### API Endpoints
- `POST /tts` — generate speech (returns WAV). The `tts_voice` param selects behavior: `cpu1` (no cloning), `gpu1` (with voice ref), `cloning` (uploaded voice), `standard` (fine control).
- `POST /upload` — upload a voice reference file to `voices/my_voices/`
- `GET /voices` — list available preset and uploaded voices

## No Test Suite

There are no automated tests. The `test_*.wav` files in the root are manually generated sample outputs.
