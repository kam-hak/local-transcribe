import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_transcribe import cli
from local_transcribe.cli import PlannedFile, confirm_batch
from local_transcribe.planner import make_plan
from local_transcribe.types import AcousticReport, SpeechSpan, TranscribeConfig

EN_MODEL = "parakeet-unified-en-0.6b-Q8_0.gguf"


def run_local_transcribe(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "local_transcribe.cli", *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )


@pytest.mark.metal
def test_plan_prints_breakdown_and_writes_nothing(sample15s):
    proc = run_local_transcribe("--plan", str(sample15s))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert "en: " in proc.stderr
    assert EN_MODEL in proc.stderr
    assert not Path(str(sample15s) + ".transcript.jsonl").exists()


@pytest.mark.metal
def test_stdout_carries_only_jsonl(sample15s):
    proc = run_local_transcribe("-y", "-o", "-", str(sample15s))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["schema_version"] == 2
    assert payload["chunks"] and payload["utterances"] and payload["words"]
    assert payload["provenance"]["asr_model"] == EN_MODEL
    assert payload["source_file"] == str(sample15s)
    assert "Receipt" in proc.stderr
    assert "Segmentation:" in proc.stderr
    assert not Path(str(sample15s) + ".transcript.jsonl").exists()


def test_missing_input_exits_2(tmp_path):
    proc = run_local_transcribe("--plan", str(tmp_path / "nope.wav"))
    assert proc.returncode == 2
    assert "not found" in proc.stderr


def test_explicit_jsonl_output_needs_one_input(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    assert cli.main(["-o", str(tmp_path / "x.jsonl"), str(a), str(b)]) == 2


def _words_payload() -> dict:
    words = []
    t = 0
    for sentence in ("Hello there my friend.", "This is a test.", "Goodbye now."):
        for token in sentence.split():
            words.append(
                {
                    "text": token,
                    "start": t,
                    "end": t + 300,
                    "speaker": "SPEAKER_00",
                    "language": "en",
                }
            )
            t += 350
        t += 3000
    return {
        "schema_version": 2,
        "text": "",
        "utterances": [],
        "chunks": [],
        "words": words,
        "provenance": {"engine": "local"},
    }


def test_rechunk_in_place_and_stdout(tmp_path, capsys):
    path = tmp_path / "sample.transcript.jsonl"
    path.write_text(json.dumps(_words_payload()) + "\n", encoding="utf-8")

    assert cli.main(["rechunk", str(path)]) == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["chunks"] and payload["utterances"]
    assert payload["text"].startswith("Hello there my friend.")
    assert payload["provenance"] == {"engine": "local"}
    assert not path.with_name(path.name + ".partial").exists()

    capsys.readouterr()
    assert cli.main(["rechunk", "-o", "-", str(path)]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["chunks"] == payload["chunks"]


def test_rechunk_without_words_fails(tmp_path):
    path = tmp_path / "bad.transcript.jsonl"
    path.write_text(json.dumps({"text": "no words"}) + "\n", encoding="utf-8")
    assert cli.main(["rechunk", str(path)]) == 1


def _long_tail_item() -> PlannedFile:
    span = SpeechSpan(0, 60_000, (("sw", 0.7), ("en", 0.3)), "sw", 0.7)
    report = AcousticReport(60_000, 60_000, (span,), (("sw", 1.0),), "sw")
    plan = make_plan(report, TranscribeConfig())
    assert plan.cloud_recommended
    return PlannedFile(Path("a.wav"), "a.wav", report, plan, report)


def test_confirm_batch_non_interactive_stays_local():
    item = _long_tail_item()
    cfg = confirm_batch([item], TranscribeConfig(), prompt_allowed=False)
    assert cfg == TranscribeConfig()
    assert item.plan.engine == "local"


def test_confirm_batch_restrict_replans_without_reanalysis(monkeypatch):
    answers = iter(["r", "en"])
    monkeypatch.setattr(cli, "ask", lambda prompt: next(answers))
    item = _long_tail_item()
    cfg = confirm_batch([item], TranscribeConfig(), prompt_allowed=True)
    assert cfg.allowed_langs == ("en",)
    assert item.plan.asr_model == EN_MODEL
    assert not item.plan.cloud_recommended


def test_transcribe_loads_working_directory_env_before_global(tmp_path, monkeypatch):
    from local_transcribe import models

    home = tmp_path / "home"
    (home / ".config" / "local-transcribe").mkdir(parents=True)
    (home / ".config" / "local-transcribe" / ".env").write_text(
        "ELEVENLABS_API_KEY=from-global\nELEVENLABS_BASE_URL=https://global.test/v1\n"
    )
    (tmp_path / ".env").write_text("ELEVENLABS_API_KEY=from-cwd\n")
    (tmp_path / "a.wav").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    for var in ("ELEVENLABS_API_KEY", "ELEVENLABS_BASE_URL"):
        monkeypatch.setenv(var, "unset-after-test")
        monkeypatch.delenv(var)
    monkeypatch.setattr(models, "ModelCache", lambda: None)
    seen = {}

    def fake_preflight(paths, cfg, cache):
        seen["key"] = os.environ.get("ELEVENLABS_API_KEY")
        seen["base_url"] = os.environ.get("ELEVENLABS_BASE_URL")
        return [], []

    monkeypatch.setattr(cli, "run_preflight", fake_preflight)

    assert cli.main(["--engine", "cloud", "--plan", "a.wav"]) == 0
    assert seen == {"key": "from-cwd", "base_url": "https://global.test/v1"}
