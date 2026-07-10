"""Build-time smoke test for the isolated Chatterbox venv.

Runs inside /opt/chatterbox-venv during the Docker build. Fails the build if a
package resolves outside the venv or if the numpy/scipy pins drifted — the
failure mode that otherwise only surfaces at pod startup, as a `np.long`
AttributeError raised by a system scipy built against numpy 2.
"""

import sys

VENV_PREFIX = "/opt/chatterbox-venv/"

import numpy
import scipy
import torch
import torchaudio
import perth  # noqa: F401  — server.py patches this on import

for module in (numpy, scipy, torch, torchaudio):
    if VENV_PREFIX not in module.__file__:
        sys.exit(f"FAIL: {module.__name__} resolves outside the venv: {module.__file__}")

if not numpy.__version__.startswith("1.26"):
    sys.exit(f"FAIL: numpy is {numpy.__version__}, expected 1.26.x")

if not scipy.__version__.startswith("1.12"):
    sys.exit(f"FAIL: scipy is {scipy.__version__}, expected 1.12.x")

# The exact import chain that used to pull the system scipy into the venv:
# transformers -> modeling_llama -> loss_utils -> loss_deformable_detr -> scipy
from transformers.models.llama import modeling_llama  # noqa: E402,F401
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402,F401

print(
    f"venv OK: numpy={numpy.__version__} scipy={scipy.__version__} "
    f"torch={torch.__version__} torchaudio={torchaudio.__version__}"
)
