import os
import shutil
import subprocess
from pathlib import Path

import pytest

SAY = Path("/usr/bin/say")
SPOKEN_SAMPLE_TEXT = (
    "Today we are testing local speech transcription and speaker diarization "
    "on Apple Silicon Metal. The recording should be detected as English, "
    "transcribed with the Parakeet model, and split into clear sentences."
)


def _to_wav_15s(source: Path, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
         "-af", "apad", "-t", "15", "-ar", "16000", "-ac", "1", str(out)],
        check=True,
    )  # fmt: skip


@pytest.fixture(scope="session")
def sample15s(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    work = tmp_path_factory.mktemp("audio")
    out = work / "sample15s.wav"
    env_sample = os.getenv("LOCAL_TRANSCRIBE_SAMPLE_AUDIO")
    if env_sample:
        source = Path(env_sample)
        if not source.exists():
            pytest.skip(f"LOCAL_TRANSCRIBE_SAMPLE_AUDIO not found: {source}")
    elif SAY.exists():
        source = work / "spoken.aiff"
        subprocess.run([str(SAY), "-o", str(source), SPOKEN_SAMPLE_TEXT], check=True)
    else:
        pytest.skip("set LOCAL_TRANSCRIBE_SAMPLE_AUDIO; /usr/bin/say is macOS-only")
    _to_wav_15s(source, out)
    return out
