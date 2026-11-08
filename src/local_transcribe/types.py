"""Immutable data contracts for the transcription core."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 2

EngineChoice = Literal["auto", "local", "cloud"]
ResolvedEngine = Literal["local", "cloud"]

_ISO639_2_3_TO_1 = {
    "eng": "en",
    "mya": "my",
    "bur": "my",
    "tha": "th",
    "lao": "lo",
    "khm": "km",
    "zho": "zh",
    "chi": "zh",
    "cmn": "zh",
    "jpn": "ja",
    "kor": "ko",
    "vie": "vi",
    "ind": "id",
    "msa": "ms",
    "may": "ms",
    "zsm": "ms",
    "tgl": "fil",
    "ara": "ar",
    "hin": "hi",
    "tur": "tr",
    "fas": "fa",
    "per": "fa",
    "pes": "fa",
    "fra": "fr",
    "fre": "fr",
    "deu": "de",
    "ger": "de",
    "spa": "es",
    "por": "pt",
    "ita": "it",
    "nld": "nl",
    "dut": "nl",
    "rus": "ru",
    "ukr": "uk",
    "pol": "pl",
    "ces": "cs",
    "cze": "cs",
    "slk": "sk",
    "slo": "sk",
    "slv": "sl",
    "hrv": "hr",
    "bul": "bg",
    "ron": "ro",
    "rum": "ro",
    "hun": "hu",
    "ell": "el",
    "gre": "el",
    "dan": "da",
    "swe": "sv",
    "fin": "fi",
    "est": "et",
    "lav": "lv",
    "lit": "lt",
    "mlt": "mt",
}


def normalize_lang_code(code: str) -> str:
    """Lowercase, drop any region suffix and map 3-letter aliases to ISO 639-1."""
    base = re.split(r"[-_]", code.strip().lower(), maxsplit=1)[0]
    return _ISO639_2_3_TO_1.get(base, base)


@dataclass(frozen=True)
class TranscribeConfig:
    diarize: bool = False
    allowed_langs: tuple[str, ...] | None = None
    engine: EngineChoice = "auto"
    vocab_terms: tuple[str, ...] = ()
    vocab_source: str | None = None


@dataclass(frozen=True)
class SpeechSpan:
    start_ms: int
    end_ms: int
    lang_probs: tuple[tuple[str, float], ...]
    language: str
    lang_prob: float


@dataclass(frozen=True)
class AcousticReport:
    duration_ms: int
    speech_ms: int
    spans: tuple[SpeechSpan, ...]
    lang_shares: tuple[tuple[str, float], ...]
    dominant_language: str


@dataclass(frozen=True)
class Plan:
    engine: ResolvedEngine
    asr_model: str
    diarizer_model: str | None
    cloud_recommended: bool
    reasons: tuple[str, ...]
    lang_shares: tuple[tuple[str, float], ...]
    dominant_language: str


@dataclass(frozen=True)
class SpeakerTurn:
    start_ms: int
    end_ms: int
    speaker: str


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    end: int
    speaker: str | None = None
    language: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "language": self.language,
        }


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: int
    end: int
    text: str
    language: str | None = None

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "language": self.language,
        }


@dataclass(frozen=True)
class Provenance:
    audio_prep: str
    engine: ResolvedEngine
    asr_model: str
    diarizer_model: str | None
    diarization_enabled: bool
    allowed_langs: tuple[str, ...] | None
    lang_distribution: tuple[tuple[str, float], ...]
    vocab_count: int
    vocab_sha256: str | None
    chunking_strategy: str
    elapsed_ms: int

    def to_dict(self) -> dict:
        return {
            "audio_prep": self.audio_prep,
            "engine": self.engine,
            "asr_model": self.asr_model,
            "diarizer_model": self.diarizer_model,
            "diarization_enabled": self.diarization_enabled,
            "allowed_langs": list(self.allowed_langs)
            if self.allowed_langs is not None
            else None,
            "lang_distribution": dict(self.lang_distribution),
            "vocab_count": self.vocab_count,
            "vocab_sha256": self.vocab_sha256,
            "chunking_strategy": self.chunking_strategy,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    utterances: tuple[Segment, ...]
    chunks: tuple[Segment, ...]
    words: tuple[Word, ...]
    dominant_language: str
    duration_ms: int
    provenance: Provenance

    def to_jsonl_dict(
        self, source_file: Path | str, file_size_bytes: int, mime_type: str
    ) -> dict:
        """Build the single-line `.transcript.jsonl` payload read by downstream consumers."""
        return {
            "schema_version": SCHEMA_VERSION,
            "text": self.text,
            "utterances": [u.to_dict() for u in self.utterances],
            "chunks": [c.to_dict() for c in self.chunks],
            "words": [w.to_dict() for w in self.words],
            "source_file": str(source_file),
            "mime_type": mime_type,
            "file_size_bytes": file_size_bytes,
            "duration_seconds": self.duration_ms / 1000.0,
            "language_code": self.dominant_language,
            "provenance": self.provenance.to_dict(),
        }


def load_vocab_terms(
    path: Path | None = None, inline: str = ""
) -> tuple[tuple[str, ...], str | None]:
    """Read vocabulary terms from a file (one per line, `#` comments) and an inline list.

    Inline terms are separated by commas, semicolons or newlines. Terms are NFC-normalized
    and deduplicated in first-seen order. The hash covers the normalized terms joined by
    newlines, or is None when there are no terms.
    """
    raw: list[str] = []
    if path is not None:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                raw.append(line)
    raw.extend(re.split(r"[,;\n]", inline))

    terms: list[str] = []
    seen: set[str] = set()
    for item in raw:
        term = unicodedata.normalize("NFC", item.strip())
        if term and term not in seen:
            seen.add(term)
            terms.append(term)

    if not terms:
        return (), None
    digest = hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()
    return tuple(terms), digest
