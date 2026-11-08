import dataclasses
import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from local_transcribe.planner import (
    DIARIZER_MODEL,
    auto_detect_min_share,
    make_plan,
    renormalize_report,
    summarize_languages,
    with_engine,
)
from local_transcribe.segment import (
    assign_speakers_and_languages,
    clamp_words,
    join_tokens,
    rechunk_jsonl_payload,
    segment_words,
    speaker_label,
)
from local_transcribe.types import (
    AcousticReport,
    Provenance,
    Segment,
    SpeakerTurn,
    SpeechSpan,
    TranscribeConfig,
    TranscriptResult,
    Word,
    load_vocab_terms,
    normalize_lang_code,
)

PARAKEET_EN = "parakeet-unified-en-0.6b-Q8_0.gguf"
PARAKEET_V3 = "parakeet-tdt-0.6b-v3-Q8_0.gguf"
QWEN3 = "qwen3-asr-0.6b-Q8_0.gguf"
WHISPER = "whisper-large-v3-turbo-Q8_0.gguf"


def test_load_vocab_terms_normalizes_dedupes_and_hashes(tmp_path):
    nfd = unicodedata.normalize("NFD", "Café")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        f"# header comment\n\nPostgreSQL\n  {nfd}  \nCafé\n# another\nKubernetes\nPostgreSQL\n",
        encoding="utf-8",
    )

    terms, digest = load_vocab_terms(vocab, inline="GraphQL, Kubernetes; SQLite")

    assert terms == ("PostgreSQL", "Café", "Kubernetes", "GraphQL", "SQLite")
    assert all(unicodedata.is_normalized("NFC", t) for t in terms)
    assert digest == hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()


def test_load_vocab_terms_empty():
    assert load_vocab_terms() == ((), None)
    assert load_vocab_terms(inline=" , ;\n") == ((), None)


def test_normalize_lang_code():
    assert normalize_lang_code("eng") == "en"
    assert normalize_lang_code("JPN") == "ja"
    assert normalize_lang_code("en-US") == "en"
    assert normalize_lang_code("yue") == "yue"


def test_types_are_frozen():
    w = Word("hi", 0, 10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.text = "bye"


def _provenance(**overrides):
    base = {
        "audio_prep": "ffmpeg 16kHz mono f32le",
        "engine": "local",
        "asr_model": PARAKEET_EN,
        "diarizer_model": None,
        "diarization_enabled": False,
        "allowed_langs": ("en", "ja"),
        "lang_distribution": (("en", 0.7), ("ja", 0.3)),
        "vocab_count": 2,
        "vocab_sha256": "abc",
        "chunking_strategy": "sentence-pack",
        "elapsed_ms": 1234,
    }
    base.update(overrides)
    return Provenance(**base)


def test_transcript_result_to_jsonl_dict_matches_consumer_contract():
    words = (Word("Hello", 0, 400, "A", "en"), Word("です", 500, 800, "A", "ja"))
    seg = Segment("A", 0, 800, "Hello です", "en")
    result = TranscriptResult(
        text="Hello です",
        utterances=(seg,),
        chunks=(seg,),
        words=words,
        dominant_language="en",
        duration_ms=90_500,
        provenance=_provenance(),
    )

    d = result.to_jsonl_dict(Path("/tmp/a.mp3"), 1234, "audio/mpeg")

    assert d["schema_version"] == 2
    assert d["source_file"] == "/tmp/a.mp3"
    assert d["duration_seconds"] == 90.5
    assert d["language_code"] == "en"
    assert d["words"][1] == {
        "text": "です",
        "start": 500,
        "end": 800,
        "speaker": "A",
        "language": "ja",
    }
    assert d["chunks"][0] == {
        "speaker": "A",
        "start": 0,
        "end": 800,
        "text": "Hello です",
        "language": "en",
    }
    assert d["provenance"]["allowed_langs"] == ["en", "ja"]
    assert d["provenance"]["lang_distribution"] == {"en": 0.7, "ja": 0.3}
    line = json.dumps(d, ensure_ascii=False)
    assert "\n" not in line and "です" in line
    assert json.loads(line) == d


def _span(start_s, end_s, *probs):
    lang_probs = tuple(sorted(probs, key=lambda p: -p[1]))
    return SpeechSpan(
        start_s * 1000, end_s * 1000, lang_probs, lang_probs[0][0], lang_probs[0][1]
    )


def _report(spans):
    speech_ms = sum(s.end_ms - s.start_ms for s in spans)
    shares, dominant = summarize_languages(spans, speech_ms)
    return AcousticReport(speech_ms + 5000, speech_ms, tuple(spans), shares, dominant)


def test_auto_detect_ignores_languages_under_fifteen_percent():
    report = _report([_span(0, 930, ("en", 0.9)), _span(930, 1000, ("cy", 0.9))])
    assert report.lang_shares == (("en", pytest.approx(0.93)),)
    assert make_plan(report, TranscribeConfig()).asr_model == PARAKEET_EN

    report = _report([_span(0, 850, ("en", 0.9)), _span(850, 1000, ("de", 0.9))])
    assert [c for c, _ in report.lang_shares] == ["en", "de"]


def test_auto_detect_min_share():
    assert auto_detect_min_share(60_000) == pytest.approx(0.20), "short clip: 20%"
    assert auto_detect_min_share(180_000) == pytest.approx(30 / 180), "30s floor"
    assert auto_detect_min_share(3_600_000) == pytest.approx(0.15)


def test_short_clip_requires_twenty_percent():
    report = _report([_span(0, 49, ("en", 0.9)), _span(49, 60, ("de", 0.9))])
    assert report.lang_shares == (("en", pytest.approx(49 / 60)),)
    report = _report([_span(0, 47, ("en", 0.9)), _span(47, 60, ("de", 0.9))])
    assert [c for c, _ in report.lang_shares] == ["en", "de"]


def test_explicit_langs_keep_languages_over_five_percent():
    report = _report(
        [
            _span(0, 930, ("en", 0.9), ("es", 0.05)),
            _span(930, 1000, ("es", 0.8), ("en", 0.1)),
            _span(1000, 1030, ("es", 0.8), ("en", 0.1)),
        ]
    )
    assert [c for c, _ in report.lang_shares] == ["en"]

    renorm = renormalize_report(report, ("en", "es"))

    assert [c for c, _ in renorm.lang_shares] == ["en", "es"]
    assert dict(renorm.lang_shares)["es"] == pytest.approx(100 / 1030)

    renorm = renormalize_report(
        _report([_span(0, 970, ("en", 0.9)), _span(970, 1000, ("es", 0.9))]),
        ("en", "es"),
    )
    assert [c for c, _ in renorm.lang_shares] == ["en"], "3% stays under 5%"


def test_low_confidence_spans_do_not_count():
    report = _report(
        [_span(0, 500, ("en", 0.9)), _span(500, 700, ("sw", 0.6), ("en", 0.3))]
    )
    assert [c for c, _ in report.lang_shares] == ["en"]


def test_no_speech_defaults_to_english():
    report = AcousticReport(10_000, 0, (), *summarize_languages((), 0))
    assert report.dominant_language == "en"
    plan = make_plan(report, TranscribeConfig())
    assert plan.asr_model == PARAKEET_EN and not plan.cloud_recommended


def test_pure_english_routes_to_parakeet():
    report = _report(
        [_span(0, 300, ("en", 0.95)), _span(300, 600, ("en", 0.9), ("de", 0.05))]
    )
    plan = make_plan(report, TranscribeConfig(diarize=True))
    assert plan.engine == "local"
    assert plan.asr_model == PARAKEET_EN
    assert plan.diarizer_model == DIARIZER_MODEL
    assert plan.cloud_recommended is False
    assert plan.dominant_language == "en"


def test_english_plus_swahili_routes_to_whisper_and_recommends_cloud():
    report = _report(
        [_span(0, 400, ("en", 0.9)), _span(400, 600, ("sw", 0.8), ("en", 0.1))]
    )
    plan = make_plan(report, TranscribeConfig())
    assert plan.engine == "local"
    assert plan.asr_model == WHISPER
    assert plan.diarizer_model is None
    assert plan.cloud_recommended is True
    assert any(
        "Detected sw (33.3% of speech): local model WER > 5%; consider cloud (ElevenLabs Scribe v2)"
        in r
        for r in plan.reasons
    )


def test_routing_tiers():
    eu = make_plan(
        _report([_span(0, 300, ("en", 0.9)), _span(300, 600, ("de", 0.9))]),
        TranscribeConfig(),
    )
    assert eu.asr_model == PARAKEET_V3 and not eu.cloud_recommended
    asia = make_plan(
        _report([_span(0, 300, ("zh", 0.9)), _span(300, 600, ("ja", 0.9))]),
        TranscribeConfig(),
    )
    assert asia.asr_model == QWEN3 and not asia.cloud_recommended
    mixed = make_plan(
        _report([_span(0, 300, ("en", 0.9)), _span(300, 600, ("zh", 0.9))]),
        TranscribeConfig(),
    )
    assert mixed.asr_model == WHISPER and mixed.cloud_recommended
    assert any("need different local models" in r for r in mixed.reasons)


def test_langs_en_renormalization_drops_false_positives():
    report = _report(
        [
            _span(0, 400, ("en", 0.9), ("de", 0.1)),
            _span(400, 600, ("hi", 0.7), ("en", 0.2), ("ur", 0.1)),
        ]
    )
    assert "hi" in dict(report.lang_shares)

    renorm = renormalize_report(report, ("eng",))

    assert renorm is not report
    assert report.spans[1].language == "hi", "input untouched"
    assert all(
        s.language == "en" and s.lang_prob == pytest.approx(1.0) for s in renorm.spans
    )
    assert renorm.lang_shares == (("en", pytest.approx(1.0)),)
    assert (
        make_plan(renorm, TranscribeConfig(allowed_langs=("en",))).asr_model
        == PARAKEET_EN
    )


def test_renormalize_renormalizes_and_falls_back():
    report = _report(
        [
            _span(0, 100, ("hi", 0.5), ("ja", 0.3), ("en", 0.1)),
            _span(100, 200, ("hi", 1.0)),
        ]
    )
    renorm = renormalize_report(report, ("en", "jpn"))
    assert renorm.spans[0].lang_probs == (
        ("ja", pytest.approx(0.75)),
        ("en", pytest.approx(0.25)),
    )
    assert renorm.spans[1].lang_probs == (("en", 1.0),)
    assert renormalize_report(report, None) is report


def test_cloud_engine_and_with_engine():
    report = _report([_span(0, 400, ("en", 0.9)), _span(400, 600, ("sw", 0.9))])
    cloud = make_plan(
        report, TranscribeConfig(engine="cloud", diarize=True), cloud_available=True
    )
    assert (cloud.engine, cloud.asr_model, cloud.diarizer_model) == (
        "cloud",
        "scribe_v2",
        None,
    )

    local = make_plan(report, TranscribeConfig(diarize=True), cloud_available=True)
    switched = with_engine(local, "cloud", diarize=True)
    assert (switched.engine, switched.asr_model, switched.diarizer_model) == (
        "cloud",
        "scribe_v2",
        None,
    )
    assert switched.lang_shares == local.lang_shares
    back = with_engine(switched, "local", diarize=True)
    assert (back.engine, back.asr_model, back.diarizer_model) == (
        "local",
        WHISPER,
        DIARIZER_MODEL,
    )


def _words(spec, speaker=None, language=None):
    """spec: list of (text, start_ms, end_ms)."""
    return [Word(t, s, e, speaker, language) for t, s, e in spec]


def _timed(tokens, start=0, step=300, dur=250, speaker=None):
    out, t = [], start
    for tok in tokens:
        out.append(Word(tok, t, t + dur, speaker))
        t += step
    return out, t


def test_speaker_label():
    assert [speaker_label(i) for i in (0, 1, 25, 26, 27, 51, 52)] == [
        "A",
        "B",
        "Z",
        "AA",
        "AB",
        "AZ",
        "BA",
    ]


def test_join_tokens_script_aware():
    assert (
        join_tokens(["今日", "の", "Central", "Data", "Unit", "会議", "。"])
        == "今日の Central Data Unit 会議。"
    )
    assert (
        join_tokens(["Hello", ",", "world", "!", "How", "are", "you", "?"])
        == "Hello, world! How are you?"
    )
    assert join_tokens(["ไป", "ไหน", "日本", "語"]) == "ไปไหน日本語"
    assert join_tokens(["สวัสดี", "ครับ", "Bangkok", "ใช่"]) == "สวัสดีครับ Bangkok ใช่"
    assert join_tokens([]) == "" and join_tokens(["", "  "]) == ""


def test_clamp_words():
    words = [Word("ok", 0, 300), Word("会議", 1_000, 86_000), Word("bad", 500, 400)]
    out = clamp_words(words)
    assert out[0] is words[0]
    assert (out[1].start, out[1].end) == (85_600, 86_000)
    assert out[2].start <= out[2].end


def test_english_hesitation_pause_stays_in_one_clause():
    first, t = _timed(
        [
            "Okay,",
            "so",
            "I'm",
            "doing",
            "an",
            "experiment.",
            "Um,",
            "I'm",
            "not",
            "gonna",
            "phrase",
            "them--",
            "I'm",
            "not",
            "gonna",
            "create,",
        ]
    )
    second, t = _timed(
        [
            "uh,",
            "flashcards,",
            "but",
            "what",
            "I",
            "do",
            "want",
            "to",
            "do",
            "is",
            "have",
            "these",
            "as",
            "lists.",
        ],
        start=t + 1_500,
    )
    third, _ = _timed(
        [
            "Then",
            "I",
            "can",
            "review",
            "them",
            "later",
            "on",
            "and",
            "remember",
            "what",
            "mattered",
            "most.",
        ],
        start=t + 3_000,
    )

    words, utterances, chunks = segment_words(first + second + third)

    assert all(w.speaker == "A" and w.language == "en" for w in words)
    assert len(utterances) == 1
    assert any("I'm not gonna create, uh, flashcards" in c.text for c in chunks)
    assert not [c for c in chunks if len(c.text) < 20]


FIRST_THOUGHT = "This is the first complete thought here and it keeps going for a while without a stop"
SECOND_THOUGHT = (
    "and here another thought begins after a long break in the recording session"
)


def test_long_pause_without_mid_thought_starts_new_chunk():
    first, t = _timed(FIRST_THOUGHT.split())
    second, _ = _timed(SECOND_THOUGHT.split(), start=t + 3_000)
    _, utterances, chunks = segment_words(first + second)
    assert len(utterances) == 1
    assert [c.text for c in chunks] == [FIRST_THOUGHT, SECOND_THOUGHT]


def test_long_pause_after_comma_does_not_split():
    first, t = _timed((FIRST_THOUGHT + ",").split())
    second, _ = _timed(SECOND_THOUGHT.split(), start=t + 3_000)
    _, _, chunks = segment_words(first + second)
    assert [c.text for c in chunks] == [f"{FIRST_THOUGHT}, {SECOND_THOUGHT}"]


def test_zero_gap_runon_is_capped():
    words, _ = _timed(["word"] * 400, step=10, dur=10)
    _, _, chunks = segment_words(words)
    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= 2 * 450 + len(" word")


def test_scriptio_continua_code_switched_with_clamping():
    sentence = ["今日", "の", "会議", "で", "は", "Central", "Data", "Unit", "の", "来年",
                "の", "予算", "に", "つい", "て", "話し", "ます", "。"]  # fmt: skip
    first = sentence * 2
    words, t = _timed(first, step=200, dur=180)
    silence_absorbing = Word("その", t, t + 85_000)
    words.append(silence_absorbing)
    tail = ["後", "Data", "を", "毎日", "集め", "て", "保存", "し", "ます", "。"]
    rest, _ = _timed(tail + tail[1:], start=t + 85_100, step=200, dur=180)
    words += rest

    out_words, utterances, chunks = segment_words(words, default_language="ja")

    stretched = out_words[len(first)]
    assert (stretched.start, stretched.end) == (
        silence_absorbing.start,
        silence_absorbing.end,
    ), "returned words keep original timing"
    sentence_text = "今日の会議では Central Data Unit の来年の予算について話します。"
    assert [c.text for c in chunks] == [
        f"{sentence_text} {sentence_text}",
        "その後 Data を毎日集めて保存します。 Data を毎日集めて保存します。",
    ]
    assert chunks[1].start == t + 85_000 - 400
    assert all(c.language == "ja" for c in chunks)
    assert len(utterances) == 1 and " 。" not in utterances[0].text


def test_multi_speaker_turns_are_never_swallowed():
    a1, t = _timed(
        [
            "So",
            "I",
            "think",
            "the",
            "budget",
            "should",
            "go",
            "to",
            "the",
            "field",
            "teams",
            "first",
            "because",
        ]
    )
    b, t2 = _timed(["Right."], start=t + 100)
    a2, end = _timed(
        [
            "they",
            "know",
            "the",
            "ground",
            "situation",
            "best",
            "and",
            "can",
            "act",
            "quickly.",
        ],
        start=t2 + 100,
    )
    turns = [
        SpeakerTurn(0, t, "spk3"),
        SpeakerTurn(t, t2, "spk1"),
        SpeakerTurn(t2, end, "spk3"),
    ]

    words, utterances, chunks = segment_words(a1 + b + a2, speaker_turns=turns)

    assert [u.speaker for u in utterances] == ["A", "B", "A"]
    assert [c.speaker for c in chunks] == ["A", "B", "A"]
    assert utterances[1].text == "Right."
    assert all(c.text.count("Right.") <= 1 for c in chunks)
    assert words[len(a1)].speaker == "B"


def test_assign_speakers_existing_ids_and_nearest_turn():
    words = _words([("a", 0, 100), ("b", 200, 300), ("c", 400, 500)])
    words = [
        dataclasses.replace(w, speaker=s)
        for w, s in zip(words, ["speaker_2", "speaker_0", "speaker_2"])
    ]
    assert [w.speaker for w in assign_speakers_and_languages(words)] == ["A", "B", "A"]

    words = _words(
        [("x", 0, 100), ("y", 1_000, 1_100), ("w", 5_000, 5_100), ("z", 20_000, 20_100)]
    )
    turns = [SpeakerTurn(3_000, 4_000, "s1"), SpeakerTurn(0, 500, "s0")]
    out = assign_speakers_and_languages(words, turns)
    assert [w.speaker for w in out] == ["A", "A", "B", "B"], (
        "y and w take the nearest turn; z is over 5s away and keeps the previous speaker"
    )


def test_assign_languages_from_spans():
    words = _words([("hello", 0, 400), ("です", 1_000, 1_300), ("x", 9_000, 9_100)])
    words[2] = dataclasses.replace(words[2], language="th")
    spans = [
        _span(0, 0.5, ("en", 0.9)),
        SpeechSpan(900, 1_500, (("ja", 0.8),), "ja", 0.8),
    ]
    out = assign_speakers_and_languages(
        words, speech_spans=spans, default_language="fr"
    )
    assert [w.language for w in out] == ["en", "ja", "th"]


def test_rechunk_jsonl_payload_preserves_other_fields():
    words, _ = _timed(
        [
            "Okay,",
            "so",
            "this",
            "is",
            "a",
            "short",
            "test",
            "of",
            "the",
            "rechunk",
            "function",
            "working",
            "end",
            "to",
            "end.",
        ]
    )
    payload = {
        "text": "old",
        "chunks": [{"text": "old"}],
        "words": [
            {"text": w.text, "start": w.start, "end": w.end, "speaker": "A"}
            for w in words
        ],
        "language_code": "eng",
    }
    out = rechunk_jsonl_payload(payload)
    assert out["language_code"] == "eng" and out["words"] is payload["words"]
    assert (
        out["text"]
        == "Okay, so this is a short test of the rechunk function working end to end."
    )
    assert out["chunks"][0]["text"] == out["text"]
    assert payload["text"] == "old"


ENGLISH_TALK = [
    "Good morning, everyone, and thanks for joining the quarterly planning call.",
    "Today I want to cover the budget, the hiring plan, and the release schedule.",
    "Let's start with the budget, because it shapes everything else we decide.",
    "We spent about eighty percent of what we planned, mostly on infrastructure.",
    "The database migration took longer than expected, so some costs moved into this quarter.",
    "On hiring, we closed two of the four open roles, and the other two are in final interviews.",
    "Now, the release schedule.",
    "The beta goes out on the fifteenth, and the general release follows four weeks later, if the beta is stable.",
    "Are there any questions before we move on to the roadmap?",
]

JAPANESE_ENGLISH_TALK = [
    ["今日", "は", "新しい", "Data", "Pipeline", "の", "設計", "に", "つい", "て", "説明", "し", "ます", "。"],
    ["まず、", "PostgreSQL", "から", "毎晩", "データ", "を", "読み", "込み", "ます", "。"],
    ["次", "に", "Kubernetes", "の", "Job", "で", "集計", "を", "実行", "し", "ます", "。"],
    ["結果", "は", "GraphQL", "の", "API", "で", "ダッシュボード", "に", "表示", "さ", "れ", "ます", "。"],
    ["質問", "が", "あれ", "ば", "どうぞ", "。"],
]  # fmt: skip


def _jsonl_fixture(sentences: list[list[str]], language: str) -> str:
    words, t = [], 0
    for i, tokens in enumerate(sentences):
        for token in tokens:
            words.append(
                {
                    "text": token,
                    "start": t,
                    "end": t + 280,
                    "speaker": "spk0",
                    "language": language,
                }
            )
            t += 320
        t += 2_800 if i % 3 == 2 else 700
    payload = {
        "schema_version": 2,
        "text": "",
        "utterances": [],
        "chunks": [],
        "words": words,
        "language_code": language,
        "provenance": {"engine": "local"},
    }
    return json.dumps(payload, ensure_ascii=False) + "\n"


@pytest.mark.parametrize(
    ("jsonl", "expected_text"),
    [
        pytest.param(
            _jsonl_fixture([s.split() for s in ENGLISH_TALK], "en"),
            " ".join(ENGLISH_TALK),
            id="solo-english",
        ),
        pytest.param(
            _jsonl_fixture(JAPANESE_ENGLISH_TALK, "ja"),
            " ".join(join_tokens(s) for s in JAPANESE_ENGLISH_TALK),
            id="japanese-english-code-switched",
        ),
    ],
)
def test_realistic_transcript_rechunk_keeps_chunk_contract(jsonl, expected_text):
    payload = json.loads(jsonl)

    out = rechunk_jsonl_payload(payload)

    assert out["words"] == payload["words"]
    assert out["text"] == expected_text
    assert out["chunks"] and out["utterances"]
    assert not [c for c in out["chunks"] if len(c["text"].strip()) < 20]
    assert max(len(c["text"]) for c in out["chunks"]) <= 1_000
    assert all(
        isinstance(c["start"], int) and isinstance(c["end"], int) for c in out["chunks"]
    )
    assert all(c["speaker"] == "A" for c in out["chunks"])
    assert all(c["language"] == payload["language_code"] for c in out["chunks"])
    json.loads(json.dumps(out, ensure_ascii=False))
