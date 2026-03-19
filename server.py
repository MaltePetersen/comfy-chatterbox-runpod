import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uuid
import torch
import torchaudio
import shutil

# Fix for Apple Silicon: resemble-perth watermarker has no ARM binary,
# so PerthImplicitWatermarker is None. Patch it with DummyWatermarker.
# This only affects the inaudible watermark — audio quality is identical.
import perth
if perth.PerthImplicitWatermarker is None:
    print("Patching perth: using DummyWatermarker (no ARM binary available)")
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_custom_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

# Device detection
# Note: MPS (Apple Silicon GPU) is disabled by default — Chatterbox has known
# tensor allocation issues with MPS. Set CHATTERBOX_DEVICE=mps to force it.
override = os.environ.get("CHATTERBOX_DEVICE")
if override:
    device = override
elif torch.cuda.is_available():
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("CUDA Version:", torch.version.cuda)
else:
    device = "cpu"
    print("Using CPU (set CHATTERBOX_DEVICE=mps to try Apple Silicon GPU)")

print(f"Device: {device}")

# Lazy-loaded model cache
models = {}


def get_model(model_type="turbo"):
    """Lazy-load and cache Chatterbox models."""
    if model_type not in models:
        if model_type == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            models[model_type] = ChatterboxTurboTTS.from_pretrained(device=device)
        elif model_type == "standard":
            from chatterbox.tts import ChatterboxTTS
            models[model_type] = ChatterboxTTS.from_pretrained(device=device)
        elif model_type == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            # Multilingual checkpoint was saved with CUDA tensors — patch
            # torch.load to force map_location so it works on CPU/MPS.
            _orig_load = torch.load
            torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": device})
            try:
                models[model_type] = ChatterboxMultilingualTTS.from_pretrained(device=device)
            finally:
                torch.load = _orig_load
        print(f"Loaded Chatterbox {model_type} model on {device}")
    return models[model_type]


VOICES_DIR = "./voices"
PRESETS_DIR = "./voices/presets"
MY_VOICES_DIR = "./voices/my_voices"


def resolve_audio_prompt(tts_voice, xtts_speaker):
    """Resolve the voice reference audio path from the request parameters."""
    if tts_voice == "gpu1":
        if xtts_speaker == "nicole" or xtts_speaker is None:
            return f"{VOICES_DIR}/nicole.wav"
        # Check presets first, then my_voices
        for directory in [PRESETS_DIR, MY_VOICES_DIR]:
            # Try exact match, then with .wav extension
            for name in [xtts_speaker, f"{xtts_speaker}.wav"]:
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    return path
        return f"{VOICES_DIR}/nicole.wav"

    elif tts_voice == "cloning":
        if xtts_speaker:
            # Check my_voices first, then presets
            for directory in [MY_VOICES_DIR, PRESETS_DIR]:
                for name in [xtts_speaker, f"{xtts_speaker}.wav"]:
                    path = os.path.join(directory, name)
                    if os.path.exists(path):
                        return path
            return f"{MY_VOICES_DIR}/{xtts_speaker}"

    return None


@app.post("/tts")
async def generate_tts(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")
    xtts_speaker = data.get("xtts_speaker")
    tts_voice = data.get("tts_voice", "gpu1")

    # Chatterbox-specific optional params
    exaggeration = data.get("exaggeration", 0.5)
    cfg_weight = data.get("cfg_weight", 0.5)

    if not text:
        return JSONResponse(content={"error": "No text provided"}, status_code=400)

    audio_prompt_path = resolve_audio_prompt(tts_voice, xtts_speaker)

    # Validate audio prompt exists
    if audio_prompt_path and not os.path.exists(audio_prompt_path):
        print(f"Warning: audio prompt not found: {audio_prompt_path}")
        audio_prompt_path = None

    # Select model: multilingual for non-English, otherwise turbo or standard
    if lang and lang != "en":
        model = get_model("multilingual")
        generate_kwargs = {
            "text": text,
            "language_id": lang,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path

    elif tts_voice == "standard":
        # Explicit standard model request — supports exaggeration/cfg_weight
        model = get_model("standard")
        generate_kwargs = {
            "text": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path

    else:
        # cpu1, gpu1, cloning, turbo → use Turbo (fastest, supports [laugh] etc.)
        model = get_model("turbo")
        generate_kwargs = {"text": text}
        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path

    print(f"Generating: tts_voice={tts_voice}, lang={lang}, speaker={xtts_speaker}")
    print(f"Model kwargs: { {k: v for k, v in generate_kwargs.items() if k != 'text'} }")

    wav = model.generate(**generate_kwargs)

    tmp_wav = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
    torchaudio.save(tmp_wav, wav, model.sr)

    return FileResponse(tmp_wav, media_type="audio/wav", filename="speech.wav")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(file.filename)
    file_location = f"./voices/my_voices/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={"filename": file.filename, "success": True})


@app.get("/voices")
async def list_voices():
    """List all available voice presets and uploaded voices."""
    presets = []
    if os.path.exists(PRESETS_DIR):
        presets = sorted([
            f for f in os.listdir(PRESETS_DIR)
            if f.endswith((".wav", ".mp3")) and not f.startswith(".")
        ])

    my_voices = []
    if os.path.exists(MY_VOICES_DIR):
        my_voices = sorted([
            f for f in os.listdir(MY_VOICES_DIR)
            if f.endswith((".wav", ".mp3")) and not f.startswith(".")
        ])

    return JSONResponse(content={
        "presets": presets,
        "my_voices": my_voices,
        "default": "nicole.wav",
    })
