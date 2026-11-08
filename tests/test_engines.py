import itertools
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from local_transcribe import analyze as analyze_mod
from local_transcribe.analyze import analyze_audio, build_windows, sample_indices
from local_transcribe.audio import AUDIO_PREP_DESC, decode_audio_pcm, pcm_duration_ms
from local_transcribe.models import ModelCache, resolve_model_path
from local_transcribe.planner import CLOUD_ASR_MODEL, DIARIZER_MODEL, make_plan
from local_transcribe.transcribe import (
    _normalize_elevenlabs_words,
    execute_plan,
    plan_slices,
    transcribe_cloud_words,
)
from local_transcribe.types import (
    SCHEMA_VERSION,
    AcousticReport,
    Plan,
    SpeechSpan,
    TranscribeConfig,
    Word,
)

LOCAL_MODELS = (
    "silero-vad-v6.2-F32.gguf",
    "parakeet-unified-en-0.6b-Q8_0.gguf",
    "lang-id-voxlingua107-ecapa-Q8_0.gguf",
    "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf",
)


@pytest.mark.metal
@pytest.mark.parametrize("filename", LOCAL_MODELS)
def test_resolve_model_path_finds_local_gguf(filename):
    path = resolve_model_path(filename, auto_download=False)
    assert path.name == filename
    assert path.stat().st_size > 0


def test_resolve_model_path_unknown_model_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_TRANSCRIBE_MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no known download URL"):
        resolve_model_path("not-a-model.gguf")


def test_decode_audio_missing_ffmpeg_gives_install_hint(tmp_path, monkeypatch):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="brew install ffmpeg"):
        decode_audio_pcm(audio)


def test_build_windows_merges_splits_and_drops_short():
    windows = build_windows([(0, 500), (800, 2000), (5000, 5300), (10_000, 35_000)])
    assert windows[0] == (0, 2000)
    assert (5000, 5300) not in windows
    long_run = windows[1:]
    assert long_run[0][0] == 10_000 and long_run[-1][1] == 35_000
    assert all(b - a <= 10_000 for a, b in long_run)
    assert all(a2 == b1 for (_, b1), (a2, _) in itertools.pairwise(long_run))


def test_build_windows_keeps_short_segments_when_nothing_else():
    assert build_windows([(0, 300), (2000, 2500)]) == [(0, 300), (2000, 2500)]
    assert build_windows([]) == []


def test_sample_indices_bounded_and_spread():
    assert sample_indices(5) == [0, 1, 2, 3, 4]
    idx = sample_indices(500)
    assert len(idx) == 32 and idx[0] == 0 and idx[-1] == 499


def test_plan_slices_cuts_at_silence():
    slices = plan_slices(70_000, 28_000, [10_000, 25_000, 40_000, 52_000])
    assert slices == [(0, 25_000), (25_000, 52_000), (52_000, 70_000)]
    assert plan_slices(70_000, 28_000, []) == [
        (0, 28_000),
        (28_000, 56_000),
        (56_000, 70_000),
    ]
    assert plan_slices(5000, 28_000, []) == [(0, 5000)]


class _FakeSession:
    def __init__(self, fn):
        self.run = fn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAnalysisCache:
    def __init__(self, n_segments):
        self.langid_calls = 0
        segs = tuple(
            SimpleNamespace(start_sample=i * 48_000, end_sample=i * 48_000 + 24_000)
            for i in range(n_segments)
        )
        self.vad = SimpleNamespace(
            vad_session=lambda: _FakeSession(lambda pcm: SimpleNamespace(segments=segs))
        )
        self.lid = SimpleNamespace(
            langid_info=SimpleNamespace(min_audio_ms=500),
            langid_session=lambda: _FakeSession(self._langid),
        )

    def _langid(self, pcm):
        self.langid_calls += 1
        return SimpleNamespace(
            candidates=(
                SimpleNamespace(code="en", p=0.9),
                SimpleNamespace(code="jpn", p=0.1),
            )
        )

    def get(self, filename, auto_download=True):
        return self.vad if filename == analyze_mod.VAD_MODEL_FILENAME else self.lid


def test_analyze_audio_samples_at_most_32_langid_windows():
    cache = _FakeAnalysisCache(80)
    pcm = np.zeros(80 * 48_000, dtype=np.float32)
    report = analyze_audio(pcm, TranscribeConfig(), cache)
    assert cache.langid_calls == 32
    assert len(report.spans) == 80
    assert report.speech_ms == 80 * 1500
    assert report.dominant_language == "en"
    assert all(s.lang_probs[1][0] == "ja" for s in report.spans)


def test_analyze_audio_empty_pcm_uses_allowed_language():
    report = analyze_audio(
        np.zeros(0, dtype=np.float32),
        TranscribeConfig(allowed_langs=("ja", "en")),
        object(),
    )
    assert report == AcousticReport(0, 0, (), (), "ja")


@pytest.mark.metal
def test_local_metal_end_to_end(sample15s):
    cache = ModelCache()
    pcm = decode_audio_pcm(sample15s)
    assert pcm.dtype == np.float32
    assert abs(len(pcm) - 240_000) <= 1600
    assert pcm_duration_ms(pcm) == pytest.approx(15_000, abs=100)

    cfg = TranscribeConfig(diarize=True)
    report = analyze_audio(pcm, cfg, cache)
    assert report.dominant_language == "en"
    assert report.speech_ms > 0

    plan = make_plan(report, cfg)
    assert plan.engine == "local"
    assert plan.asr_model == "parakeet-unified-en-0.6b-Q8_0.gguf"
    assert plan.diarizer_model == DIARIZER_MODEL

    result = execute_plan(plan, pcm, sample15s, report, cfg, cache)
    assert result.words and result.chunks and result.utterances
    assert all(w.text.strip() and w.speaker and w.language for w in result.words)
    english = sum(w.language == "en" for w in result.words)
    assert english > len(result.words) // 2, "LangID may mis-tag short windows"
    assert result.dominant_language == "en"
    assert result.text.strip()
    assert result.provenance.diarizer_model == DIARIZER_MODEL
    assert result.provenance.audio_prep == AUDIO_PREP_DESC

    payload = result.to_jsonl_dict(sample15s, sample15s.stat().st_size, "audio/wav")
    assert payload["schema_version"] == SCHEMA_VERSION == 2
    assert payload["language_code"] == "en"
    assert payload["provenance"]["engine"] == "local"
    assert payload["provenance"]["diarization_enabled"] is True


SCRIBE_WORDS = [
    {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
    {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing", "speaker_id": "speaker_0"},
    {"text": "there.", "start": 0.5, "end": 0.9, "type": "word", "speaker_id": "speaker_0"},
    {"text": "Konnichiwa", "start": 3.0, "end": 3.6, "type": "word", "speaker_id": "speaker_1"},
    {"text": "friend.", "start": 3.7, "end": 4.1, "type": "word", "speaker_id": "speaker_1"},
]  # fmt: skip


def test_normalize_elevenlabs_words_converts_seconds_and_labels_speakers():
    assert _normalize_elevenlabs_words(SCRIBE_WORDS) == [
        Word("Hello", 0, 400, "A"),
        Word("there.", 500, 900, "A"),
        Word("Konnichiwa", 3000, 3600, "B"),
        Word("friend.", 3700, 4100, "B"),
    ]


@pytest.fixture
def scribe(monkeypatch):
    monkeypatch.delenv("LOCAL_TRANSCRIBE_CLOUD_MODEL", raising=False)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_BASE_URL", "https://scribe.test/v1")
    captured = {}

    def fake_post(url, *, headers, files, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["fields"] = [(k, v[1]) for k, v in files if k != "file"]
        payload = {"text": "Hello there. Konnichiwa friend.", "words": SCRIBE_WORDS}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setattr(httpx, "post", fake_post)
    return captured


def _report(shares):
    spans = (
        SpeechSpan(0, 1000, (("en", 0.9),), "en", 0.9),
        SpeechSpan(3000, 4200, (("ja", 0.8),), "ja", 0.8),
    )
    return AcousticReport(5000, 2200, spans, shares, "en")


def test_transcribe_cloud_words_sends_scribe_fields(scribe, tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00" * 64)
    shares = (("en", 1.0),)
    plan = Plan("cloud", CLOUD_ASR_MODEL, None, False, ("test",), shares, "en")
    terms = tuple(f"term{i}" for i in range(120))
    cfg = TranscribeConfig(diarize=False, vocab_terms=terms)

    words, text = transcribe_cloud_words(audio, plan, _report(shares), cfg)

    assert scribe["url"] == "https://scribe.test/v1/speech-to-text"
    assert scribe["headers"] == {"xi-api-key": "test-key"}
    fields = scribe["fields"]
    assert ("model_id", "scribe_v2") in fields
    assert ("diarize", "false") in fields
    assert ("language_code", "en") in fields
    assert [v for k, v in fields if k == "keyterms"] == list(terms[:100])
    assert text == "Hello there. Konnichiwa friend."
    assert [w.text for w in words] == ["Hello", "there.", "Konnichiwa", "friend."]


def test_transcribe_cloud_words_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    plan = Plan("cloud", CLOUD_ASR_MODEL, None, False, (), (), "en")
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        transcribe_cloud_words(
            tmp_path / "x.mp3", plan, _report(()), TranscribeConfig()
        )


def test_execute_plan_cloud_segments_and_provenance(scribe, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_TRANSCRIBE_CLOUD_MODEL", "scribe_v2_test")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00" * 64)
    shares = (("en", 0.6), ("ja", 0.4))
    plan = Plan("cloud", CLOUD_ASR_MODEL, None, False, ("test",), shares, "en")
    cfg = TranscribeConfig(diarize=True, vocab_terms=("PostgreSQL", "GraphQL"))

    result = execute_plan(
        plan, np.zeros(80_000, dtype=np.float32), audio, _report(shares), cfg
    )

    fields = scribe["fields"]
    assert ("model_id", "scribe_v2_test") in fields
    assert ("diarize", "true") in fields
    assert not any(k == "language_code" for k, _ in fields)
    assert [v for k, v in fields if k == "keyterms"] == ["PostgreSQL", "GraphQL"]

    assert [w.text for w in result.words] == [
        "Hello",
        "there.",
        "Konnichiwa",
        "friend.",
    ]
    assert [w.language for w in result.words] == ["en", "en", "ja", "ja"]
    assert [w.speaker for w in result.words] == ["A", "A", "B", "B"]
    assert [u.speaker for u in result.utterances] == ["A", "B"]
    assert result.chunks and result.text == "Hello there. Konnichiwa friend."
    assert result.duration_ms == 5000

    prov = result.provenance
    assert prov.engine == "cloud"
    assert prov.asr_model == "scribe_v2_test"
    assert prov.diarizer_model == "scribe_v2_test_diarize"
    assert prov.diarization_enabled is True
    assert prov.lang_distribution == shares
    assert prov.vocab_count == 2
    assert prov.vocab_sha256 and len(prov.vocab_sha256) == 64
    assert prov.chunking_strategy == "uax29_uterm_pause_v1"
    assert prov.audio_prep == AUDIO_PREP_DESC
