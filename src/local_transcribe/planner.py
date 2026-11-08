"""Pure language accounting and execution planning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from local_transcribe.types import (
    AcousticReport,
    Plan,
    ResolvedEngine,
    SpeechSpan,
    TranscribeConfig,
    normalize_lang_code,
)

MIN_SPAN_LANG_PROB = 0.65
AUTO_DETECT_MIN_SHARE = 0.15
SHORT_CLIP_MIN_SHARE = 0.20
SHORT_CLIP_MIN_MS = 30_000
ALLOWED_LANGS_MIN_SHARE = 0.05
CLOUD_ASR_MODEL = "scribe_v2"
DIARIZER_MODEL = "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf"
FALLBACK_LOCAL_MODEL = "whisper-large-v3-turbo-Q8_0.gguf"

# Language sets from transcribe.cpp v0.4.1's model catalog where each model has WER <= 5%.
LOW_WER_LOCAL_MODELS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"en"}), "parakeet-unified-en-0.6b-Q8_0.gguf"),
    (
        frozenset(
            {
                "en",
                "bg",
                "hr",
                "cs",
                "da",
                "nl",
                "et",
                "fi",
                "fr",
                "de",
                "el",
                "hu",
                "it",
                "lv",
                "lt",
                "mt",
                "pl",
                "pt",
                "ro",
                "ru",
                "sk",
                "sl",
                "es",
                "sv",
                "uk",
            }
        ),
        "parakeet-tdt-0.6b-v3-Q8_0.gguf",
    ),
    (
        frozenset(
            {
                "zh",
                "yue",
                "ja",
                "ko",
                "th",
                "vi",
                "id",
                "ms",
                "fil",
                "ar",
                "hi",
                "tr",
                "fa",
            }
        ),
        "qwen3-asr-0.6b-Q8_0.gguf",
    ),
)
_LOW_WER_LANGS = frozenset().union(*(langs for langs, _ in LOW_WER_LOCAL_MODELS))


def auto_detect_min_share(speech_ms: int) -> float:
    """15% of speech, raised to 20% (at most 30s) so one stray window cannot reroute a short clip."""
    if speech_ms <= 0:
        return AUTO_DETECT_MIN_SHARE
    short_clip = min(SHORT_CLIP_MIN_SHARE, SHORT_CLIP_MIN_MS / speech_ms)
    return max(AUTO_DETECT_MIN_SHARE, short_clip)


def summarize_languages(
    spans: Sequence[SpeechSpan],
    speech_ms: int,
    min_share: float | None = None,
) -> tuple[tuple[tuple[str, float], ...], str]:
    """Return (language shares sorted descending, dominant language).

    A language counts when its spans with `lang_prob >= MIN_SPAN_LANG_PROB` cover at
    least `min_share` of `speech_ms`, which defaults to `auto_detect_min_share`.
    """
    if min_share is None:
        min_share = auto_detect_min_share(speech_ms)
    confident: dict[str, int] = defaultdict(int)
    overall: dict[str, int] = defaultdict(int)
    for span in spans:
        dur = max(0, span.end_ms - span.start_ms)
        overall[span.language] += dur
        if span.lang_prob >= MIN_SPAN_LANG_PROB:
            confident[span.language] += dur

    shares = tuple(
        sorted(
            (
                (code, dur / speech_ms)
                for code, dur in confident.items()
                if speech_ms > 0 and dur > 0 and dur / speech_ms >= min_share
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
    if shares:
        return shares, shares[0][0]
    if overall:
        return shares, min(overall.items(), key=lambda item: (-item[1], item[0]))[0]
    return shares, "en"


def _renormalize_span(span: SpeechSpan, allowed: tuple[str, ...]) -> SpeechSpan:
    probs: dict[str, float] = defaultdict(float)
    for code, p in span.lang_probs:
        norm = normalize_lang_code(code)
        if norm in allowed:
            probs[norm] += p
    total = sum(probs.values())
    if total <= 0:
        lang_probs: tuple[tuple[str, float], ...] = ((allowed[0], 1.0),)
    else:
        lang_probs = tuple(
            sorted(
                ((code, p / total) for code, p in probs.items()),
                key=lambda item: (-item[1], item[0]),
            )
        )
    return replace(
        span,
        lang_probs=lang_probs,
        language=lang_probs[0][0],
        lang_prob=lang_probs[0][1],
    )


def renormalize_report(
    report: AcousticReport, allowed_langs: tuple[str, ...] | None
) -> AcousticReport:
    """Restrict every span's language probabilities to `allowed_langs` and recompute shares."""
    if not allowed_langs:
        return report
    allowed = tuple(dict.fromkeys(normalize_lang_code(code) for code in allowed_langs))
    spans = tuple(_renormalize_span(span, allowed) for span in report.spans)
    shares, dominant = summarize_languages(
        spans, report.speech_ms, ALLOWED_LANGS_MIN_SHARE
    )
    return replace(report, spans=spans, lang_shares=shares, dominant_language=dominant)


def _route_local(langs: frozenset[str]) -> tuple[str, bool]:
    """Return (model, low_wer): the first low-WER model covering `langs`, else the fallback."""
    for model_langs, model in LOW_WER_LOCAL_MODELS:
        if langs <= model_langs:
            return model, True
    return FALLBACK_LOCAL_MODEL, False


def _plan_langs(
    lang_shares: tuple[tuple[str, float], ...], dominant: str
) -> frozenset[str]:
    return frozenset(code for code, _ in lang_shares) or frozenset({dominant})


def _assess(
    lang_shares: tuple[tuple[str, float], ...], dominant: str
) -> tuple[str, bool, tuple[str, ...]]:
    """Return (local model, cloud_recommended, reasons) for the significant languages."""
    langs = _plan_langs(lang_shares, dominant)
    model, low_wer = _route_local(langs)
    described = ", ".join(
        f"{code} ({share * 100:.1f}%)" for code, share in lang_shares
    ) or (f"{dominant} (no significant language detected)")
    reasons = [f"Languages: {described}; local model {model}"]
    if low_wer:
        return model, False, tuple(reasons)

    long_tail = [
        (code, share) for code, share in lang_shares if code not in _LOW_WER_LANGS
    ]
    if not lang_shares and dominant not in _LOW_WER_LANGS:
        long_tail = [(dominant, 1.0)]
    for code, share in long_tail:
        reasons.append(
            f"Detected {code} ({share * 100:.1f}% of speech): local model WER > 5%; "
            "consider cloud (ElevenLabs Scribe v2) for higher accuracy"
        )
    if not long_tail:
        reasons.append(
            f"Languages {', '.join(sorted(langs))} need different local models; "
            f"fallback {model} WER > 5%; consider cloud (ElevenLabs Scribe v2) for higher accuracy"
        )
    return model, True, tuple(reasons)


def make_plan(
    report: AcousticReport, cfg: TranscribeConfig, cloud_available: bool = False
) -> Plan:
    """Choose engine and models for a report. `auto` always plans local; the CLI may flip it."""
    model, cloud_recommended, reasons = _assess(
        report.lang_shares, report.dominant_language
    )
    if cfg.engine == "cloud":
        reasons = ("Engine set to cloud (ElevenLabs Scribe v2)",) + reasons
        if not cloud_available:
            reasons += ("Cloud engine requested but not available",)
        return Plan(
            engine="cloud",
            asr_model=CLOUD_ASR_MODEL,
            diarizer_model=None,
            cloud_recommended=cloud_recommended,
            reasons=reasons,
            lang_shares=report.lang_shares,
            dominant_language=report.dominant_language,
        )

    if cloud_recommended and not cloud_available:
        reasons += ("Cloud engine not available; staying local",)
    return Plan(
        engine="local",
        asr_model=model,
        diarizer_model=DIARIZER_MODEL if cfg.diarize else None,
        cloud_recommended=cloud_recommended,
        reasons=reasons,
        lang_shares=report.lang_shares,
        dominant_language=report.dominant_language,
    )


def with_engine(plan: Plan, engine: ResolvedEngine, diarize: bool) -> Plan:
    """Return `plan` switched to `engine`, re-selecting models for it."""
    if engine == "cloud":
        return replace(
            plan,
            engine="cloud",
            asr_model=CLOUD_ASR_MODEL,
            diarizer_model=None,
            reasons=plan.reasons + ("Switched to cloud (ElevenLabs Scribe v2)",),
        )
    model, _ = _route_local(_plan_langs(plan.lang_shares, plan.dominant_language))
    return replace(
        plan,
        engine="local",
        asr_model=model,
        diarizer_model=DIARIZER_MODEL if diarize else None,
        reasons=plan.reasons + (f"Switched to local ({model})",),
    )
