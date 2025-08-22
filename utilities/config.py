import os
from pathlib import Path

RUN_HOST = os.getenv("MMDFLED_RUN_HOST", "0.0.0.0")
RUN_PORT = int(os.getenv("MMDFLED_RUN_PORT", "5000"))
DEBUG = os.getenv("MMDFLED_DEBUG", "false").lower() in ("1", "true")

BACKENDS = ("auto", "smirnoff", "reaxff", "psi4")
DEFAULT_BACKEND = os.getenv("MMDFLED_DEFAULT_BACKEND", "auto")
if DEFAULT_BACKEND not in BACKENDS:
    raise RuntimeError(f"invalid default backend: {DEFAULT_BACKEND}")

DEFAULT_SMIRNOFF_XML = os.getenv("MMDFLED_SMIRNOFF_XML", "openff-2.0.0.offxml")
FORCEFIELD_DIR = Path(os.getenv("MMDFLED_FORCEFIELD_DIR", "potentials")).resolve()
if not FORCEFIELD_DIR.exists():
    FORCEFIELD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMESTEP_PS = float(os.getenv("MMDFLED_TIMESTEP_PS", "0.00025"))
DEFAULT_TEMPERATURE_K = float(os.getenv("MMDFLED_TEMPERATURE_K", "298.0"))

FS_PER_PS = 1000.0
KCAL2KJ = 4.184
ANG2NM = 0.1

MAX_ATOMS = int(os.getenv("MMDFLED_MAX_ATOMS", "5000"))
MAX_STEPS = int(os.getenv("MMDFLED_MAX_STEPS", "20000"))
MAX_REPORT_FRAMES = int(os.getenv("MMDFLED_MAX_REPORT_FRAMES", "5000"))
MAX_REQUEST_BYTES = int(os.getenv("MMDFLED_MAX_REQUEST_BYTES", str(2 * 1024 * 1024)))

REAXFF_MAX_DT_PS = float(os.getenv("MMDFLED_REAXFF_MAX_DT_PS", "0.001"))
DEFAULT_PSI4_THRESHOLD = int(os.getenv("MMDFLED_PSI4_THRESHOLD", "10"))
