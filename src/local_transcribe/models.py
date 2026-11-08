"""GGUF model resolution, download and a per-process model cache for transcribe.cpp."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import transcribe_cpp

_HF = "https://huggingface.co/handy-computer"

MODEL_URLS: dict[str, str] = {
    "silero-vad-v6.2-F32.gguf": f"{_HF}/silero-vad-v6.2-gguf/resolve/main/silero-vad-v6.2-F32.gguf",
    "lang-id-voxlingua107-ecapa-Q8_0.gguf": f"{_HF}/voxlingua107-ecapa-gguf/resolve/main/lang-id-voxlingua107-ecapa-Q8_0.gguf",
    "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf": f"{_HF}/diar-streaming-sortformer-4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf",
    "parakeet-unified-en-0.6b-Q8_0.gguf": f"{_HF}/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q8_0.gguf",
    "parakeet-tdt-0.6b-v3-Q8_0.gguf": f"{_HF}/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf",
    "qwen3-asr-0.6b-Q8_0.gguf": f"{_HF}/qwen3-asr-0.6b-gguf/resolve/main/qwen3-asr-0.6b-Q8_0.gguf",
    "whisper-large-v3-turbo-Q8_0.gguf": f"{_HF}/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q8_0.gguf",
}


def primary_models_dir() -> Path:
    """Writable model directory: `LOCAL_TRANSCRIBE_MODELS_DIR` or `~/.local/share/local-transcribe/models`."""
    return Path(
        os.getenv(
            "LOCAL_TRANSCRIBE_MODELS_DIR",
            Path.home() / ".local" / "share" / "local-transcribe" / "models",
        )
    ).expanduser()


def model_search_dirs() -> tuple[Path, ...]:
    """Directories searched for GGUF files, highest priority first."""
    return (
        primary_models_dir(),
        Path.home() / "Library" / "Application Support" / "Hex" / "models",
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        size_hint = f" ({total / (1024 * 1024):.0f} MB)" if total >= 1024 * 1024 else ""
        print(
            f"Downloading {dest.name}{size_hint} to {dest.parent} ...",
            file=sys.stderr,
        )
        with open(partial, "wb") as out:
            out.writelines(resp.iter_bytes())
            out.flush()
            os.fsync(out.fileno())
    os.replace(partial, dest)


def resolve_model_path(filename: str, auto_download: bool = True) -> Path:
    """Return the path of a non-empty local copy of `filename`, downloading known models if allowed."""
    dirs = model_search_dirs()
    for directory in dirs:
        candidate = directory / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    if auto_download and filename in MODEL_URLS:
        dest = dirs[0] / filename
        _download(MODEL_URLS[filename], dest)
        return dest

    searched = ", ".join(str(d) for d in dirs)
    hint = (
        "auto-download disabled" if filename in MODEL_URLS else "no known download URL"
    )
    raise FileNotFoundError(f"Model {filename} not found in {searched} ({hint})")


def _silent_log_callback(level: int, message: str) -> None:
    pass


class ModelCache:
    """Load each transcribe.cpp model once, on Metal when available."""

    def __init__(self) -> None:
        # A no-op callable silences native ggml/Metal logging (None resets to stderr).
        transcribe_cpp.set_log_callback(_silent_log_callback)
        self.backend: transcribe_cpp.Backend = (
            "metal" if transcribe_cpp.backend_available("metal") else "cpu"
        )
        self._models: dict[str, transcribe_cpp.Model] = {}

    def get(self, filename: str, auto_download: bool = True) -> transcribe_cpp.Model:
        model = self._models.get(filename)
        if model is None:
            path = resolve_model_path(filename, auto_download=auto_download)
            model = transcribe_cpp.Model(str(path), backend=self.backend)
            self._models[filename] = model
        return model
