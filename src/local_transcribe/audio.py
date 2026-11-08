"""Decode any ffmpeg-readable audio to 16 kHz mono float32 PCM."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".mp4",
    ".opus",
    ".webm",
}
SAMPLE_RATE = 16000
AUDIO_PREP_DESC = "ffmpeg -ar 16000 -ac 1 -f f32le"


def decode_audio_pcm(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "f32le",
                "-",
            ],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with: brew install ffmpeg"
        ) from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed decoding {path}: {stderr}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def pcm_duration_ms(pcm: np.ndarray) -> int:
    return round(len(pcm) * 1000.0 / SAMPLE_RATE)
