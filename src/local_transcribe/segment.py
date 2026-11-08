"""Language-agnostic word attribution and clause/paragraph segmentation."""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from local_transcribe.types import Segment, SpeakerTurn, SpeechSpan, Word

STERM_CHARS = frozenset(".?!။。？！؟।፠⁇⁈⁉")
TRAILING_QUOTES = "\"'’)”]}»"
ATTACH_CHARS = frozenset("၊,;:)]}")
STANDALONE_PUNCT = frozenset("။၊.,?!;:")
SPACE_AFTER_CHARS = frozenset("။၊")
MID_THOUGHT_ENDINGS = (",", "--", "-", "...")
CLAMPED_TOKEN_MS = 400
MAX_SPEAKER_GAP_MS = 5000


def speaker_label(index: int) -> str:
    """Spreadsheet-style labels: 0 -> A, 25 -> Z, 26 -> AA."""
    label = ""
    n = index + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


SCRIPTIO_CONTINUA_BLOCKS = (
    (0x0E00, 0x0E7F),
    (0x0E80, 0x0EFF),
    (0x1000, 0x109F),
    (0x1780, 0x17FF),
    (0x3040, 0x30FF),
    (0x4E00, 0x9FFF),
    (0xA9E0, 0xA9FF),
    (0xAA60, 0xAA7F),
)


def _is_scriptio_continua(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in SCRIPTIO_CONTINUA_BLOCKS)


def _append_token(acc: str, token: str) -> str:
    t = token.strip()
    if not t:
        return acc
    if not acc:
        return t
    prev_ch, next_ch = acc[-1], t[0]
    if next_ch in STERM_CHARS or next_ch in ATTACH_CHARS:
        return acc + t
    if prev_ch in SPACE_AFTER_CHARS:
        return acc + " " + t
    if _is_scriptio_continua(prev_ch) and _is_scriptio_continua(next_ch):
        return acc + t
    return acc + " " + t


def join_tokens(tokens: Sequence[str]) -> str:
    """Join tokens with script-aware spacing (no spaces inside Thai, Khmer, CJK and similar runs)."""
    out = ""
    for tok in tokens:
        out = _append_token(out, tok)
    return out


def _is_sterm(text: str) -> bool:
    s = text.strip().rstrip(TRAILING_QUOTES)
    return bool(s) and s[-1] in STERM_CHARS


def clamp_words(
    words: Sequence[Word], max_token_duration_ms: int = 2000
) -> tuple[Word, ...]:
    """Re-anchor tokens the ASR stretched across prior silence to their final 400ms."""
    out = []
    for w in words:
        start, end = w.start, max(w.start, w.end)
        if end - start > max_token_duration_ms:
            start = end - CLAMPED_TOKEN_MS
        out.append(
            w if (start, end) == (w.start, w.end) else replace(w, start=start, end=end)
        )
    return tuple(out)


class _IntervalIndex:
    """Find the interval with the largest overlap with, or nearest to, a query interval."""

    def __init__(self, intervals: Sequence[tuple[int, int]]):
        self._order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
        self._starts = [intervals[i][0] for i in self._order]
        self._ends = [intervals[i][1] for i in self._order]
        self._prefix_max: list[int] = []
        best = -1
        for pos, end in enumerate(self._ends):
            if best < 0 or end > self._ends[best]:
                best = pos
            self._prefix_max.append(best)

    def best(self, start: int, end: int, max_distance: int) -> int | None:
        """Original index of the max-overlap interval, else nearest within `max_distance` of the midpoint."""
        if not self._starts:
            return None
        best_pos, best_overlap = None, 0
        pos = bisect.bisect_left(self._starts, end) - 1
        while pos >= 0 and self._ends[self._prefix_max[pos]] > start:
            overlap = min(end, self._ends[pos]) - max(start, self._starts[pos])
            if overlap > best_overlap or (overlap == best_overlap and overlap > 0):
                best_pos, best_overlap = pos, overlap
            pos -= 1
        if best_pos is not None:
            return self._order[best_pos]

        mid = (start + end) // 2
        candidates = []
        after = bisect.bisect_right(self._starts, mid)
        if after > 0:
            before = self._prefix_max[after - 1]
            candidates.append((max(0, mid - self._ends[before]), before))
        if after < len(self._starts):
            candidates.append((self._starts[after] - mid, after))
        if not candidates:
            return None
        distance, pos = min(candidates)
        return self._order[pos] if distance <= max_distance else None


def assign_speakers_and_languages(
    words: Sequence[Word],
    speaker_turns: Sequence[SpeakerTurn] = (),
    speech_spans: Sequence[SpeechSpan] = (),
    default_language: str = "en",
) -> tuple[Word, ...]:
    """Attribute each word to a speaker turn and language; speakers are relabelled A, B, ... in arrival order."""
    turn_index = _IntervalIndex([(t.start_ms, t.end_ms) for t in speaker_turns])
    span_index = _IntervalIndex([(s.start_ms, s.end_ms) for s in speech_spans])

    raw_speakers: list[str | None] = []
    prev: str | None = None
    for w in words:
        if speaker_turns:
            hit = turn_index.best(w.start, w.end, MAX_SPEAKER_GAP_MS)
            spk = speaker_turns[hit].speaker if hit is not None else prev
        else:
            spk = w.speaker if w.speaker is not None else prev
        raw_speakers.append(spk)
        prev = spk

    labels: dict[str | None, str] = {}
    out = []
    for w, spk in zip(words, raw_speakers):
        if spk not in labels:
            labels[spk] = speaker_label(len(labels))
        language = w.language
        if language is None:
            hit = span_index.best(w.start, w.end, 0)
            language = (
                speech_spans[hit].language if hit is not None else default_language
            )
        out.append(replace(w, speaker=labels[spk], language=language))
    return tuple(out)


@dataclass(frozen=True)
class _Clause:
    speaker: str
    start: int
    end: int
    gap_after: int
    text: str
    lang_weights: tuple[tuple[str, int], ...]


def _dominant_language(weights: Sequence[tuple[str, int]]) -> str | None:
    totals: dict[str, int] = defaultdict(int)
    for code, weight in weights:
        totals[code] += weight
    if not totals:
        return None
    return max(totals, key=lambda code: totals[code])


def _make_clause(words: Sequence[Word], text: str, gap_after: int) -> _Clause:
    return _Clause(
        speaker=words[0].speaker or "A",
        start=words[0].start,
        end=words[-1].end,
        gap_after=gap_after,
        text=text,
        lang_weights=tuple(
            (w.language, len(w.text.strip())) for w in words if w.language
        ),
    )


def _merge_clauses(clauses: Sequence[_Clause]) -> Segment:
    return Segment(
        speaker=clauses[0].speaker,
        start=clauses[0].start,
        end=clauses[-1].end,
        text=" ".join(c.text for c in clauses),
        language=_dominant_language([lw for c in clauses for lw in c.lang_weights]),
    )


def _split_clauses(
    words: Sequence[Word], max_chars: int, min_clause_chars: int, clause_pause_ms: int
) -> list[_Clause]:
    clauses: list[_Clause] = []
    cur = [words[0]]
    cur_text = join_tokens([words[0].text])
    for w in words[1:]:
        prev = cur[-1]
        token = w.text.strip()
        if token in STANDALONE_PUNCT:
            cur.append(w)
            cur_text = _append_token(cur_text, w.text)
            continue

        gap = max(0, w.start - prev.end)
        cur_chars = len(cur_text)
        sterm = _is_sterm(prev.text) and cur_chars >= min_clause_chars
        mid_thought = prev.text.strip().endswith(MID_THOUGHT_ENDINGS)
        pause_break = (
            gap >= clause_pause_ms and cur_chars >= min_clause_chars and not mid_thought
        )
        # ASR timestamps can collapse to zero gaps on run-on speech.
        runon_break = cur_chars >= max_chars and (
            gap >= 250 or cur_chars >= 2 * max_chars
        )

        if w.speaker != prev.speaker or sterm or pause_break or runon_break:
            clauses.append(_make_clause(cur, cur_text, gap))
            cur = [w]
            cur_text = join_tokens([w.text])
        else:
            cur.append(w)
            cur_text = _append_token(cur_text, w.text)
    clauses.append(_make_clause(cur, cur_text, 0))
    return clauses


def _pack_chunks(
    clauses: Sequence[_Clause], target_chars: int, target_ms: int, para_pause_ms: int
) -> list[list[_Clause]]:
    groups: list[list[_Clause]] = []
    buf = [clauses[0]]
    for cl in clauses[1:]:
        prev = buf[-1]
        cur_chars = sum(len(x.text) for x in buf) + len(buf) - 1
        would_chars = cur_chars + 1 + len(cl.text)
        would_dur = cl.end - buf[0].start
        topic_pause = prev.gap_after >= para_pause_ms and cur_chars >= 80
        budget_full = (
            would_chars > target_chars or (would_dur > target_ms and cur_chars >= 120)
        ) and cur_chars >= 80
        if cl.speaker != prev.speaker or topic_pause or budget_full:
            groups.append(buf)
            buf = [cl]
        else:
            buf.append(cl)
    groups.append(buf)
    return groups


def _group_chars(group: Sequence[_Clause]) -> int:
    return sum(len(c.text) for c in group) + len(group) - 1


def _absorb_orphans(
    groups: list[list[_Clause]], min_chars: int, max_chars: int
) -> list[list[_Clause]]:
    """Fold a short chunk into an adjacent chunk of the same speaker when it fits."""
    out: list[list[_Clause]] = []
    for group in groups:
        if out and out[-1][0].speaker == group[0].speaker:
            prev = out[-1]
            short = _group_chars(group) < min_chars or _group_chars(prev) < min_chars
            if short and _group_chars(prev) + 1 + _group_chars(group) <= max_chars:
                out[-1] = prev + group
                continue
        out.append(list(group))
    return out


def segment_words(
    words: Sequence[Word],
    speaker_turns: Sequence[SpeakerTurn] = (),
    speech_spans: Sequence[SpeechSpan] = (),
    default_language: str = "en",
    *,
    target_chars: int = 300,
    max_chars: int = 450,
    target_ms: int = 24000,
    min_clause_chars: int = 35,
    clause_pause_ms: int = 1200,
    para_pause_ms: int = 2500,
) -> tuple[tuple[Word, ...], tuple[Segment, ...], tuple[Segment, ...]]:
    """Return (attributed words, utterances, chunks).

    Attribution and segmentation use clamped timings; the returned words keep the
    original timestamps. Clauses and chunks always break on a speaker change.
    """
    if not words:
        return (), (), ()
    clamped = clamp_words(words)
    attributed = assign_speakers_and_languages(
        clamped, speaker_turns, speech_spans, default_language
    )
    out_words = tuple(
        replace(orig, speaker=a.speaker, language=a.language)
        for orig, a in zip(words, attributed)
    )

    clauses = _split_clauses(attributed, max_chars, min_clause_chars, clause_pause_ms)
    groups = _pack_chunks(clauses, target_chars, target_ms, para_pause_ms)
    groups = _absorb_orphans(groups, min_clause_chars, max_chars)
    chunks = tuple(_merge_clauses(g) for g in groups)

    turns: list[list[_Clause]] = []
    for cl in clauses:
        if turns and turns[-1][0].speaker == cl.speaker:
            turns[-1].append(cl)
        else:
            turns.append([cl])
    utterances = tuple(_merge_clauses(t) for t in turns)
    return out_words, utterances, chunks


def rechunk_jsonl_payload(payload: dict, **kwargs) -> dict:
    """Recompute `chunks`, `utterances` and `text` of a `.transcript.jsonl` payload from its words."""
    words = [
        Word(
            text=str(w.get("text", "")),
            start=int(float(w.get("start") or 0)),
            end=int(float(w.get("end") or 0)),
            speaker=w.get("speaker"),
            language=w.get("language"),
        )
        for w in payload.get("words") or []
    ]
    _, utterances, chunks = segment_words(words, **kwargs)
    return {
        **payload,
        "text": join_tokens([c.text for c in chunks]),
        "utterances": [u.to_dict() for u in utterances],
        "chunks": [c.to_dict() for c in chunks],
    }
