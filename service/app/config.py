"""Runtime settings, all overridable through environment variables (see deploy/forte.env.example)."""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def fenv(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


ROOT = Path(_env("FORTE_HOME", str(Path(__file__).resolve().parents[2])))
VAR_DIR = Path(_env("FORTE_VAR", str(ROOT / "var")))
JOBS_DIR = VAR_DIR / "jobs"
DB_PATH = VAR_DIR / "jobs.sqlite3"
CATALOG_PATH = Path(_env("FORTE_CATALOG", str(Path(__file__).resolve().parent / "catalog.yaml")))
OUTPUT_LANG = _env("FORTE_OUTPUT_LANG", "ru")   # language of `type` and `description`: ru | en
WEB_DIR = Path(__file__).resolve().parents[1] / "web"

API_KEY = _env("FORTE_API_KEY", "")
MAX_UPLOAD_MB = int(_env("FORTE_MAX_UPLOAD_MB", "512"))
MAX_BATCH_FILES = int(_env("FORTE_MAX_BATCH_FILES", "20"))
KEEP_SOURCE = _env("FORTE_KEEP_SOURCE", "0") == "1"   # keep uploaded originals after analysis (off: test videos
                                                      # must not be kept or reused, case section 10)

# Worker
MAX_PARALLEL_JOBS = int(_env("FORTE_MAX_PARALLEL_JOBS", "4"))
JOB_DEADLINE_SEC = float(_env("FORTE_JOB_DEADLINE_SEC", "270"))  # jury limit is 300 s from POST

# Perception
DETECTOR = _env("FORTE_DETECTOR", "small")  # rfdetr size: nano|small|medium
DETECTOR_WEIGHTS = _env("FORTE_DETECTOR_WEIGHTS", "")
DETECTOR_DEVICE = _env("FORTE_DETECTOR_DEVICE", "cuda")
PERCEPTION_FPS = float(_env("FORTE_PERCEPTION_FPS", "5"))
MAX_LONG_SIDE = int(_env("FORTE_MAX_LONG_SIDE", "960"))

# VLM (any OpenAI-compatible server: vLLM, SGLang, TensorRT-LLM, Ollama)
VLM_BASE_URL = _env("FORTE_VLM_BASE_URL", "http://127.0.0.1:8100/v1")
VLM_MODEL = _env("FORTE_VLM_MODEL", "vlm")
VLM_API_KEY = _env("FORTE_VLM_API_KEY", "none")
VLM_TIMEOUT_SEC = float(_env("FORTE_VLM_TIMEOUT_SEC", "90"))
VLM_MAX_CONCURRENCY = int(_env("FORTE_VLM_MAX_CONCURRENCY", "12"))
VLM_FRAME_LONG_SIDE = int(_env("FORTE_VLM_FRAME_LONG_SIDE", "640"))
VLM_FRAME_PIXELS = int(_env("FORTE_VLM_FRAME_PIXELS", "230000"))   # ~640x360 worth of pixels per full frame

# Decision
VIOLATION_THRESHOLD = float(_env("FORTE_VIOLATION_THRESHOLD", "0.5"))
# misses are worse than false alarms: keep review findings (not "other") that verification did not confirm,
# at a lower threshold, and say so in the description
RECALL_MODE = _env("FORTE_RECALL_MODE", "0") == "1"
# optional larger VLM (OpenAI-compatible) that looks at the whole clip; noisy finding types need its agreement
SECOND_VLM_URL = _env("FORTE_SECOND_VLM_URL", "")
SECOND_VLM_KEY = _env("FORTE_SECOND_VLM_KEY", "")
SECOND_VLM_MODEL = _env("FORTE_SECOND_VLM_MODEL", "vlm")
SECOND_VLM_TIMEOUT_SEC = float(_env("FORTE_SECOND_VLM_TIMEOUT_SEC", "150"))
SECOND_VLM_NOISY = set(_env("FORTE_SECOND_VLM_NOISY",
                            "customer_identification_not_observed,cash_counted_without_customer_present,"
                            "PERSONAL_ITEM_IN_CASH_NODE,CASH_PROCESSING_WITHOUT_CONTROLLER,"
                            "PHONE_WITH_CAMERA_IN_CASH_NODE").split(","))
RECALL_THRESHOLD = float(_env("FORTE_RECALL_THRESHOLD", "0.3"))
MAX_VIOLATIONS = int(_env("FORTE_MAX_VIOLATIONS", "3"))
CLIP_PAD_SEC = float(_env("FORTE_CLIP_PAD_SEC", "4"))
