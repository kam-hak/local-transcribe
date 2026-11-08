"""Acoustic pre-flight: Silero VAD speech windows and sampled VoxLingua107 language ID."""

from __future__ import annotations

import bisect
import itertools
import math
from collections import Counter

import numpy as np

from local_transcribe.audio import SAMPLE_RATE, pcm_duration_ms
from local_transcribe.models import ModelCache
from local_transcribe.planner import renormalize_report, summarize_languages
from local_transcribe.types import (
    AcousticReport,
    SpeechSpan,
    TranscribeConfig,
    normalize_lang_code,
)

VAD_MODEL_FILENAME = "silero-vad-v6.2-F32.gguf"
LANGID_MODEL_FILENAME = "lang-id-voxlingua107-ecapa-Q8_0.gguf"

MERGE_GAP_MS = 400
SPLIT_OVER_MS = 12_000
MAX_WINDOW_MS = 10_000
MIN_WINDOW_MS = 1000
MAX_LANGID_WINDOWS = 32
LANGID_TOP_K = 8


def _fallback_language(cfg: TranscribeConfig) -> str:
    return normalize_lang_code(cfg.allowed_langs[0]) if cfg.allowed_langs else "en"


def _empty_report(duration_ms: int, cfg: TranscribeConfig) -> AcousticReport:
    return AcousticReport(
        duration_ms=duration_ms,
        speech_ms=0,
        spans=(),
        lang_shares=(),
        dominant_language=_fallback_language(cfg),
    )


def build_windows(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge VAD segments across short gaps, split long runs, and drop sub-second windows."""
    merged: list[list[int]] = []
    for start, end in segments:
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= MERGE_GAP_MS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    windows: list[tuple[int, int]] = []
    for start, end in merged:
        dur = end - start
        if dur <= SPLIT_OVER_MS:
            windows.append((start, end))
            continue
        pieces = math.ceil(dur / MAX_WINDOW_MS)
        bounds = [start + round(i * dur / pieces) for i in range(pieces + 1)]
        windows.extend(itertools.pairwise(bounds))

    kept = [w for w in windows if w[1] - w[0] >= MIN_WINDOW_MS]
    return kept or windows


def sample_indices(n: int, k: int = MAX_LANGID_WINDOWS) -> list[int]:
    """`k` evenly spaced indices over range(n), or all of them when n <= k."""
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _langid_probs(
    session, pcm: np.ndarray, min_samples: int, allowed: frozenset[str]
) -> tuple[tuple[str, float], ...]:
    if len(pcm) < min_samples:
        pcm = np.pad(pcm, (0, min_samples - len(pcm)))
    candidates = session.run(pcm).candidates
    probs: dict[str, float] = {}
    for rank, cand in enumerate(candidates):
        code = normalize_lang_code(cand.code)
        if rank < LANGID_TOP_K or code in allowed:
            probs[code] = probs.get(code, 0.0) + float(cand.p)
    return tuple(sorted(probs.items(), key=lambda item: (-item[1], item[0])))


def analyze_audio(
    pcm: np.ndarray,
    cfg: TranscribeConfig,
    model_cache: ModelCache | None = None,
) -> AcousticReport:
    """Return speech spans with language probabilities, renormalized to `cfg.allowed_langs`."""
    duration_ms = pcm_duration_ms(pcm)
    if len(pcm) == 0:
        return _empty_report(0, cfg)
    cache = model_cache or ModelCache()

    with cache.get(VAD_MODEL_FILENAME).vad_session() as vad:
        vad_res = vad.run(pcm)
    segments = [
        (
            round(s.start_sample * 1000 / SAMPLE_RATE),
            round(s.end_sample * 1000 / SAMPLE_RATE),
        )
        for s in vad_res.segments
    ]
    windows = build_windows(segments)
    if not windows:
        return _empty_report(duration_ms, cfg)

    lid_model = cache.get(LANGID_MODEL_FILENAME)
    min_samples = int(lid_model.langid_info.min_audio_ms * SAMPLE_RATE / 1000)
    allowed = frozenset(normalize_lang_code(c) for c in cfg.allowed_langs or ())
    sampled = sample_indices(len(windows))
    sampled_probs: list[tuple[tuple[str, float], ...]] = []
    with lid_model.langid_session() as lid:
        for idx in sampled:
            start, end = windows[idx]
            window_pcm = pcm[start * SAMPLE_RATE // 1000 : end * SAMPLE_RATE // 1000]
            sampled_probs.append(_langid_probs(lid, window_pcm, min_samples, allowed))

    top_counts = Counter(p[0][0] for p in sampled_probs if p)
    majority_probs = max(
        sampled_probs,
        key=lambda p: (top_counts[p[0][0]], p[0][1]) if p else (0, 0.0),
        default=(),
    )
    sampled_set = set(sampled)

    spans = []
    for idx, (start, end) in enumerate(windows):
        pos = bisect.bisect_left(sampled, idx)
        if pos == len(sampled) or (
            pos > 0 and idx - sampled[pos - 1] <= sampled[pos] - idx
        ):
            pos -= 1
        lang_probs = sampled_probs[pos]
        borrows_singleton_outlier = (
            idx not in sampled_set
            and len(sampled) >= 8
            and lang_probs
            and top_counts[lang_probs[0][0]] < 2
        )
        if borrows_singleton_outlier:
            lang_probs = majority_probs
        language, lang_prob = (
            lang_probs[0] if lang_probs else (_fallback_language(cfg), 0.0)
        )
        spans.append(
            SpeechSpan(
                start_ms=start,
                end_ms=end,
                lang_probs=lang_probs,
                language=language,
                lang_prob=lang_prob,
            )
        )

    speech_ms = sum(s.end_ms - s.start_ms for s in spans)
    shares, dominant = summarize_languages(spans, speech_ms)
    raw = AcousticReport(
        duration_ms=duration_ms,
        speech_ms=speech_ms,
        spans=tuple(spans),
        lang_shares=shares,
        dominant_language=dominant,
    )
    return renormalize_report(raw, cfg.allowed_langs)
