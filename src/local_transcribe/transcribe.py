"""Execute a Plan on the local transcribe.cpp engine or ElevenLabs Scribe v2."""

from __future__ import annotations

import hashlib
import itertools
import os
import time
from pathlib import Path

import httpx
import numpy as np
import transcribe_cpp

from local_transcribe.analyze import analyze_audio
from local_transcribe.audio import (
    AUDIO_PREP_DESC,
    SAMPLE_RATE,
    decode_audio_pcm,
    pcm_duration_ms,
)
from local_transcribe.models import ModelCache
from local_transcribe.planner import CLOUD_ASR_MODEL, make_plan
from local_transcribe.segment import join_tokens, segment_words, speaker_label
from local_transcribe.types import (
    AcousticReport,
    Plan,
    Provenance,
    SpeakerTurn,
    SpeechSpan,
    TranscribeConfig,
    TranscriptResult,
    Word,
)

CHUNKING_STRATEGY = "uax29_uterm_pause_v1"
LONG_INPUT_MS = 15 * 60 * 1000
LONG_CHUNK_MS = 10 * 60 * 1000
SHORT_WINDOW_CHUNK_MS = 28_000
WINDOW_HEADROOM_MS = 2000
_TIMESTAMP_RANK = {"none": 0, "segment": 1, "word": 2, "token": 3}
_SENTENCEPIECE_SPACE = "▁"

ELEVENLABS_MAX_UPLOAD_BYTES = 3 * 1024**3
ELEVENLABS_MAX_KEYTERMS = 100


def _silence_cuts(spans: tuple[SpeechSpan, ...]) -> list[int]:
    """Midpoints (ms) of the silences between consecutive speech spans."""
    ordered = sorted(spans, key=lambda s: s.start_ms)
    return [
        (a.end_ms + b.start_ms) // 2
        for a, b in itertools.pairwise(ordered)
        if b.start_ms > a.end_ms
    ]


def plan_slices(total_ms: int, limit_ms: int, cuts: list[int]) -> list[tuple[int, int]]:
    """Split [0, total_ms) into slices of at most `limit_ms`, cutting at the silence nearest each limit."""
    slices = []
    cursor = 0
    while total_ms - cursor > limit_ms:
        target = cursor + limit_ms
        usable = [c for c in cuts if cursor + limit_ms // 4 < c <= target]
        end = max(usable) if usable else target
        slices.append((cursor, end))
        cursor = end
    slices.append((cursor, total_ms))
    return slices


def _words_from_tokens(tokens, offset_ms: int) -> list[Word]:
    words: list[Word] = []
    cur_text, cur_start, cur_end = "", 0, 0
    for tok in tokens:
        starts_word = tok.text[:1] in (" ", _SENTENCEPIECE_SPACE)
        piece = tok.text.replace(_SENTENCEPIECE_SPACE, " ").strip()
        if not piece:
            continue
        if cur_text and starts_word:
            words.append(Word(cur_text, cur_start + offset_ms, cur_end + offset_ms))
            cur_text = ""
        if not cur_text:
            cur_start = tok.t0_ms
        cur_text += piece
        cur_end = tok.t1_ms
    if cur_text:
        words.append(Word(cur_text, cur_start + offset_ms, cur_end + offset_ms))
    return words


def _spread_words_over_segments(segments, offset_ms: int) -> list[Word]:
    words: list[Word] = []
    for seg in segments:
        parts = seg.text.split()
        span = max(0, seg.t1_ms - seg.t0_ms)
        for i, part in enumerate(parts):
            start = seg.t0_ms + span * i // len(parts)
            end = seg.t0_ms + span * (i + 1) // len(parts)
            words.append(Word(part, start + offset_ms, end + offset_ms))
    return words


def _words_from_result(res, offset_ms: int) -> list[Word]:
    if res.words:
        return [
            Word(w.text.strip(), w.t0_ms + offset_ms, w.t1_ms + offset_ms)
            for w in res.words
            if w.text.strip()
        ]
    if res.tokens:
        return _words_from_tokens(res.tokens, offset_ms)
    return _spread_words_over_segments(res.segments, offset_ms)


def _asr_kwargs(model, plan: Plan, cfg: TranscribeConfig) -> dict:
    caps = model.capabilities
    rank = _TIMESTAMP_RANK.get(caps.max_timestamp_kind, 0)
    timestamps = "word" if rank >= 2 else ("segment" if rank == 1 else "none")
    language = (
        plan.dominant_language
        if len(caps.languages) > 1 and plan.dominant_language in caps.languages
        else None
    )
    vocabulary = (
        list(cfg.vocab_terms)
        if cfg.vocab_terms and model.supports("vocabulary")
        else None
    )
    return {"timestamps": timestamps, "language": language, "vocabulary": vocabulary}


def _run_slices(
    session, pcm: np.ndarray, slices: list[tuple[int, int]], kwargs: dict
) -> tuple[list[Word], str]:
    words: list[Word] = []
    texts: list[str] = []
    for start_ms, end_ms in slices:
        piece = pcm[start_ms * SAMPLE_RATE // 1000 : end_ms * SAMPLE_RATE // 1000]
        try:
            res = session.run(piece, **kwargs)
        except transcribe_cpp.InputTooShort:
            continue
        words.extend(_words_from_result(res, start_ms))
        texts.append(res.text)
    return words, join_tokens(texts)


def _transcribe_local(
    model, pcm: np.ndarray, report: AcousticReport, kwargs: dict
) -> tuple[list[Word], str]:
    total_ms = pcm_duration_ms(pcm)
    cuts = _silence_cuts(report.spans)
    max_audio_ms = model.capabilities.max_audio_ms
    window_ms = (
        max_audio_ms - WINDOW_HEADROOM_MS
        if max_audio_ms > WINDOW_HEADROOM_MS
        else SHORT_WINDOW_CHUNK_MS
    )
    if max_audio_ms > 0:
        limit_ms = window_ms
    elif total_ms > LONG_INPUT_MS:
        limit_ms = LONG_CHUNK_MS
    else:
        limit_ms = total_ms

    with model.session() as session:
        try:
            return _run_slices(
                session, pcm, plan_slices(total_ms, limit_ms, cuts), kwargs
            )
        except transcribe_cpp.InputTooLong:
            if limit_ms <= window_ms:
                raise
            return _run_slices(
                session, pcm, plan_slices(total_ms, window_ms, cuts), kwargs
            )


def _diarize(model, pcm: np.ndarray) -> tuple[SpeakerTurn, ...]:
    with model.diarize_session() as session:
        segments = session.run(pcm)
    return tuple(
        SpeakerTurn(
            start_ms=s.t0_ms, end_ms=s.t1_ms, speaker=speaker_label(s.speaker_id)
        )
        for s in segments
    )


def cloud_model_id() -> str:
    return os.getenv("LOCAL_TRANSCRIBE_CLOUD_MODEL") or CLOUD_ASR_MODEL


def _to_millis(seconds: float) -> int:
    return round(float(seconds) * 1000)


def _normalize_elevenlabs_words(raw_words: list[dict]) -> list[Word]:
    """Map Scribe word tokens to `Word`s in ms, dropping spacing tokens and labelling speakers A, B, ..."""
    speakers: dict[str, str] = {}
    words: list[Word] = []
    for raw in raw_words:
        text = str(raw.get("text") or "").strip()
        if raw.get("type") == "spacing" or not text:
            continue
        if raw.get("start") is None or raw.get("end") is None:
            continue
        speaker_id = str(raw.get("speaker_id") or "speaker_0")
        if speaker_id not in speakers:
            speakers[speaker_id] = speaker_label(len(speakers))
        words.append(
            Word(
                text=text,
                start=_to_millis(raw["start"]),
                end=_to_millis(raw["end"]),
                speaker=speakers[speaker_id],
            )
        )
    return words


def transcribe_cloud_words(
    audio_path: Path, plan: Plan, report: AcousticReport, cfg: TranscribeConfig
) -> tuple[list[Word], str]:
    """Upload `audio_path` to ElevenLabs Scribe v2 and return (words, text)."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    size = audio_path.stat().st_size
    if size > ELEVENLABS_MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large for ElevenLabs direct upload ({size} bytes); limit is 3 GB."
        )
    base_url = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io/v1")

    fields = [
        ("model_id", cloud_model_id()),
        ("diarize", str(cfg.diarize).lower()),
        ("tag_audio_events", "false"),
    ]
    # Without language_code, Scribe v2 detects the language and follows code-switching.
    if len(report.lang_shares) <= 1:
        fields.append(("language_code", plan.dominant_language))
    fields.extend(
        ("keyterms", term) for term in cfg.vocab_terms[:ELEVENLABS_MAX_KEYTERMS]
    )

    with open(audio_path, "rb") as audio_file:
        multipart: list[tuple[str, tuple]] = [(k, (None, v)) for k, v in fields]
        multipart.append(
            ("file", (audio_path.name, audio_file, "application/octet-stream"))
        )
        response = httpx.post(
            f"{base_url}/speech-to-text",
            headers={"xi-api-key": api_key},
            files=multipart,
            timeout=1800.0,
        )
    response.raise_for_status()
    payload = response.json()
    words = _normalize_elevenlabs_words(payload.get("words") or [])
    return words, str(payload.get("text") or "").strip()


def _vocab_sha256(cfg: TranscribeConfig) -> str | None:
    if cfg.vocab_source:
        return cfg.vocab_source
    if not cfg.vocab_terms:
        return None
    return hashlib.sha256("\n".join(cfg.vocab_terms).encode("utf-8")).hexdigest()


def execute_plan(
    plan: Plan,
    pcm: np.ndarray,
    audio_path: Path,
    report: AcousticReport,
    cfg: TranscribeConfig,
    model_cache: ModelCache | None = None,
) -> TranscriptResult:
    t0 = time.monotonic()
    speaker_turns: tuple[SpeakerTurn, ...] = ()
    if plan.engine == "local":
        cache = model_cache or ModelCache()
        asr_model = cache.get(plan.asr_model)
        raw_words, res_text = _transcribe_local(
            asr_model, pcm, report, _asr_kwargs(asr_model, plan, cfg)
        )
        if cfg.diarize and plan.diarizer_model:
            speaker_turns = _diarize(cache.get(plan.diarizer_model), pcm)
        asr_model_id = plan.asr_model
        diarizer_model = plan.diarizer_model if cfg.diarize else None
    else:
        raw_words, res_text = transcribe_cloud_words(
            Path(audio_path), plan, report, cfg
        )
        asr_model_id = cloud_model_id()
        diarizer_model = f"{asr_model_id}_diarize" if cfg.diarize else None

    words, utterances, chunks = segment_words(
        raw_words,
        speaker_turns=speaker_turns,
        speech_spans=report.spans,
        default_language=plan.dominant_language,
    )
    full_text = join_tokens([c.text for c in chunks]) if chunks else res_text.strip()

    provenance = Provenance(
        audio_prep=AUDIO_PREP_DESC,
        engine=plan.engine,
        asr_model=asr_model_id,
        diarizer_model=diarizer_model,
        diarization_enabled=cfg.diarize,
        allowed_langs=cfg.allowed_langs,
        lang_distribution=report.lang_shares,
        vocab_count=len(cfg.vocab_terms),
        vocab_sha256=_vocab_sha256(cfg),
        chunking_strategy=CHUNKING_STRATEGY,
        elapsed_ms=round((time.monotonic() - t0) * 1000),
    )
    return TranscriptResult(
        text=full_text,
        utterances=utterances,
        chunks=chunks,
        words=words,
        dominant_language=plan.dominant_language,
        duration_ms=pcm_duration_ms(pcm) if len(pcm) else report.duration_ms,
        provenance=provenance,
    )


def transcribe_audio_file(
    audio_path: Path,
    cfg: TranscribeConfig | None = None,
    model_cache: ModelCache | None = None,
) -> tuple[AcousticReport, Plan, TranscriptResult]:
    """Decode, analyze, plan and transcribe one file."""
    cfg = cfg or TranscribeConfig()
    cache = model_cache or ModelCache()
    audio_path = Path(audio_path)
    pcm = decode_audio_pcm(audio_path)
    report = analyze_audio(pcm, cfg, cache)
    plan = make_plan(report, cfg, cloud_available=bool(os.getenv("ELEVENLABS_API_KEY")))
    result = execute_plan(plan, pcm, audio_path, report, cfg, cache)
    return report, plan, result
