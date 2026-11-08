"""`local-transcribe`: stateless Unix CLI for local-first transcription and re-chunking.

stdout carries only transcript JSONL (with `-o -`); every table, prompt and receipt goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from local_transcribe.planner import make_plan, renormalize_report, with_engine
from local_transcribe.segment import rechunk_jsonl_payload
from local_transcribe.types import (
    AcousticReport,
    Plan,
    TranscribeConfig,
    TranscriptResult,
    load_vocab_terms,
    normalize_lang_code,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

TRANSCRIPT_SUFFIX = ".transcript.jsonl"


class UsageError(Exception):
    """Invalid CLI usage or a missing input; maps to exit code 2."""


def eprint(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


def ask(prompt: str) -> str:
    """Prompt on stderr and read one line from stdin; EOF reads as empty."""
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return sys.stdin.readline().strip()


def interactive(args: argparse.Namespace) -> bool:
    return not args.yes and sys.stdin.isatty()


def cloud_available() -> bool:
    return bool(os.getenv("ELEVENLABS_API_KEY"))


def parse_langs(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    codes: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        code = normalize_lang_code(part)
        if not re.fullmatch(r"[a-z]{2,3}", code):
            raise UsageError(f"Invalid language code: {part!r}")
        codes.append(code)
    return tuple(dict.fromkeys(codes)) or None


def add_transcribe_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d", "--diarize", action="store_true", help="enable speaker diarization"
    )
    parser.add_argument(
        "-l", "--langs", help="comma-separated ISO-639-1 language codes, e.g. en,es"
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "local", "cloud"],
        default="auto",
        help="execution engine (default: auto)",
    )
    parser.add_argument(
        "--vocab", type=Path, help="custom vocabulary / keyterms file (one per line)"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="non-interactive: never prompt"
    )


def build_config(args: argparse.Namespace) -> TranscribeConfig:
    """Build the run config from parsed options. Call after `.env` is loaded."""
    terms: tuple[str, ...] = ()
    vocab_sha: str | None = None
    if args.vocab is not None:
        if not args.vocab.is_file():
            raise UsageError(f"Vocabulary file not found: {args.vocab}")
        terms, vocab_sha = load_vocab_terms(args.vocab)
    if args.engine == "cloud" and not cloud_available():
        raise UsageError("--engine cloud requires ELEVENLABS_API_KEY")
    return TranscribeConfig(
        diarize=args.diarize,
        allowed_langs=parse_langs(args.langs),
        engine=args.engine,
        vocab_terms=terms,
        vocab_source=vocab_sha,
    )


@dataclass
class PlannedFile:
    path: Path
    label: str
    report: AcousticReport
    plan: Plan
    unrestricted_report: AcousticReport
    sha: str | None = None


def fmt_duration(ms: int) -> str:
    total = max(0, round(ms / 1000))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def fmt_langs(lang_shares: Sequence[tuple[str, float]], dominant: str) -> str:
    if not lang_shares:
        return f"{dominant} (low confidence)"
    return ", ".join(f"{code}: {share * 100:.1f}%" for code, share in lang_shares)


def diarizer_label(plan: Plan, cfg: TranscribeConfig) -> str:
    if not cfg.diarize:
        return "off"
    if plan.engine == "cloud":
        return "Scribe v2"
    return "Sortformer" if plan.diarizer_model else "off"


def analyze_file(
    path: Path, cfg: TranscribeConfig, cache, label: str, sha: str | None = None
) -> PlannedFile:
    from local_transcribe.analyze import analyze_audio
    from local_transcribe.audio import decode_audio_pcm

    pcm = decode_audio_pcm(path)
    report = analyze_audio(pcm, cfg, cache)
    plan = make_plan(report, cfg, cloud_available=cloud_available())
    return PlannedFile(path, label, report, plan, report, sha)


def print_plan_table(items: Sequence[PlannedFile], cfg: TranscribeConfig) -> None:
    show_sha = any(item.sha for item in items)
    header = ["File"] + (["SHA256"] if show_sha else [])
    header += ["Duration", "Speech", "Languages", "Engine / ASR model", "Diarizer"]
    rows = []
    for item in items:
        row = [item.label] + ([(item.sha or "")[:12]] if show_sha else [])
        row += [
            fmt_duration(item.report.duration_ms),
            fmt_duration(item.report.speech_ms),
            fmt_langs(item.report.lang_shares, item.report.dominant_language),
            f"{item.plan.engine}: {item.plan.asr_model}",
            diarizer_label(item.plan, cfg),
        ]
        rows.append(row)
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    eprint("\nPre-flight plan")
    eprint(line(header))
    eprint(line(["-" * w for w in widths]))
    for row in rows:
        eprint(line(row))
    for item in items:
        if item.plan.cloud_recommended:
            eprint(f"! {item.label}:")
            for reason in item.plan.reasons[1:]:
                eprint(f"    {reason}")
    eprint("")


def _restrict(
    items: list[PlannedFile], cfg: TranscribeConfig, langs: tuple[str, ...]
) -> TranscribeConfig:
    cfg = replace(cfg, allowed_langs=langs)
    for item in items:
        item.report = renormalize_report(item.unrestricted_report, langs)
        item.plan = make_plan(item.report, cfg, cloud_available=cloud_available())
    return cfg


def confirm_batch(
    items: list[PlannedFile], cfg: TranscribeConfig, prompt_allowed: bool
) -> TranscribeConfig:
    """Resolve cloud recommendations once for the whole batch; may update plans and cfg in place."""
    flagged = [item for item in items if item.plan.cloud_recommended]
    if not flagged or cfg.engine != "auto":
        return cfg
    if not prompt_allowed:
        eprint(
            f"Note: {len(flagged)} file(s) flagged for cloud; staying local "
            "(non-interactive). Pass --engine cloud to upload them."
        )
        return cfg

    while True:
        flagged = [item for item in items if item.plan.cloud_recommended]
        if not flagged:
            return cfg
        cloud_note = "" if cloud_available() else " (ELEVENLABS_API_KEY not set)"
        eprint(f"{len(flagged)} file(s) flagged: local model WER may exceed 5%.")
        eprint("  [L] Continue with local model (default)")
        eprint(
            f"  [c] Switch flagged files to cloud (ElevenLabs Scribe v2){cloud_note}"
        )
        eprint("  [r] Restrict languages (e.g. en,es) and re-plan")
        choice = ask("Choice [L/c/r]: ").lower()
        if choice in ("", "l", "local"):
            return cfg
        if choice in ("c", "cloud"):
            if not cloud_available():
                eprint("ELEVENLABS_API_KEY not set; cloud unavailable.")
                continue
            for item in flagged:
                item.plan = with_engine(item.plan, "cloud", cfg.diarize)
            return cfg
        if choice in ("r", "restrict"):
            try:
                langs = parse_langs(ask("Languages (comma-separated, e.g. en,es): "))
            except UsageError as exc:
                eprint(exc)
                continue
            if not langs:
                continue
            cfg = _restrict(items, cfg, langs)
            print_plan_table(items, cfg)
            continue
        eprint(f"Unrecognized choice: {choice!r}")


def run_preflight(
    paths: Sequence[Path],
    cfg: TranscribeConfig,
    cache,
    labels: Callable[[Path], tuple[str, str | None]] = lambda p: (p.name, None),
) -> tuple[list[PlannedFile], list[tuple[Path, str]]]:
    """Analyze and plan every path; returns (planned, failures)."""
    planned: list[PlannedFile] = []
    failed: list[tuple[Path, str]] = []
    for path in paths:
        label, sha = labels(path)
        eprint(f"Analyzing {label} ...")
        try:
            planned.append(analyze_file(path, cfg, cache, label, sha))
        except Exception as exc:
            eprint(f"  ✗ {label}: {exc}")
            failed.append((path, str(exc)))
    return planned, failed


def transcribe_planned(
    item: PlannedFile, cfg: TranscribeConfig, cache
) -> TranscriptResult:
    from local_transcribe.audio import decode_audio_pcm
    from local_transcribe.transcribe import execute_plan

    pcm = decode_audio_pcm(item.path)
    return execute_plan(item.plan, pcm, item.path, item.report, cfg, cache)


def mime_type_for(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def jsonl_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def write_atomic(dest: Path, text: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    with open(partial, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, dest)


def describe_result(result: TranscriptResult) -> list[str]:
    """Receipt lines listing every transformation applied to produce `result`."""
    prov = result.provenance
    langs = fmt_langs(prov.lang_distribution, result.dominant_language)
    lang_filter = ",".join(prov.allowed_langs) if prov.allowed_langs else "none"
    diar = f"on: {prov.diarizer_model}" if prov.diarization_enabled else "off"
    vocab = f"{prov.vocab_count} terms"
    if prov.vocab_sha256:
        vocab += f" (sha256: {prov.vocab_sha256[:12]})"
    return [
        f"Audio prep:    {prov.audio_prep}, {fmt_duration(result.duration_ms)}",
        f"Languages:     {langs} (filter: {lang_filter})",
        f"Engine:        {prov.engine}: {prov.asr_model}",
        f"Diarization:   {diar}",
        f"Vocabulary:    {vocab}",
        (
            f"Segmentation:  {prov.chunking_strategy} -> {len(result.chunks)} chunks, "
            f"{len(result.utterances)} utterances, {len(result.words)} words"
        ),
    ]


def _resolve_destinations(
    inputs: Sequence[Path], output: str | None, suffix: str
) -> list[Path | None]:
    """Map each input to its output path; None means stdout."""
    if output == "-":
        return [None] * len(inputs)
    if output is None:
        return [Path(str(p) + suffix) for p in inputs]
    out = Path(output)
    if out.suffix == ".jsonl" and not out.is_dir():
        if len(inputs) != 1:
            raise UsageError("An explicit .jsonl output path needs exactly one input")
        return [out]
    dests = [out / (p.name + suffix) for p in inputs]
    if len(set(dests)) != len(dests):
        raise UsageError(f"Inputs with the same name would collide in {out}")
    return dests


def _check_inputs(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise UsageError(f"Input file not found: {path}")


def load_env() -> None:
    """Load `.env` from the working directory, then the global config file.

    Shell variables win over both files, and the working-directory file wins over the global one.
    """
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path.home() / ".config" / "local-transcribe" / ".env")


def cmd_transcribe(args: argparse.Namespace) -> int:
    load_env()
    inputs = [Path(p) for p in args.files]
    _check_inputs(inputs)
    dests = _resolve_destinations(inputs, args.output, TRANSCRIPT_SUFFIX)
    cfg = build_config(args)

    from local_transcribe.models import ModelCache

    cache = ModelCache()
    planned, failed = run_preflight(inputs, cfg, cache)
    if planned:
        print_plan_table(planned, cfg)
    if args.plan:
        return EXIT_FAILED if failed else EXIT_OK

    cfg = confirm_batch(planned, cfg, interactive(args))
    dest_for = dict(zip(inputs, dests))
    receipts: list[tuple[PlannedFile, Path | None, TranscriptResult]] = []
    for item in planned:
        eprint(
            f"Transcribing {item.label} ({item.plan.engine}: {item.plan.asr_model}) ..."
        )
        try:
            result = transcribe_planned(item, cfg, cache)
            payload = result.to_jsonl_dict(
                source_file=item.path,
                file_size_bytes=item.path.stat().st_size,
                mime_type=mime_type_for(item.path),
            )
            dest = dest_for[item.path]
            if dest is None:
                sys.stdout.write(jsonl_line(payload))
                sys.stdout.flush()
            else:
                write_atomic(dest, jsonl_line(payload))
            receipts.append((item, dest, result))
        except Exception as exc:
            eprint(f"  ✗ {item.label}: {exc}")
            failed.append((item.path, str(exc)))

    eprint("\nReceipt")
    for item, dest, result in receipts:
        eprint(f"✓ {item.path} -> {dest if dest is not None else '<stdout>'}")
        for line in describe_result(result):
            eprint(f"    {line}")
        eprint(f"    Elapsed:       {result.provenance.elapsed_ms / 1000:.1f}s")
    for path, error in failed:
        eprint(f"✗ {path}: {error}")
    eprint(f"{len(receipts)} transcribed, {len(failed)} failed")
    return EXIT_FAILED if failed else EXIT_OK


def cmd_rechunk(args: argparse.Namespace) -> int:
    inputs = [Path(p) for p in args.files]
    _check_inputs(inputs)
    if args.output in (None, "-"):
        dests: list[Path | None] = (
            [None] * len(inputs) if args.output == "-" else list(inputs)
        )
    else:
        out = Path(args.output)
        if out.exists() and not out.is_dir():
            raise UsageError(f"rechunk output must be a directory or '-': {out}")
        dests = [out / p.name for p in inputs]

    failed = 0
    for path, dest in zip(inputs, dests):
        try:
            lines = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                payload = json.loads(raw)
                if not payload.get("words"):
                    raise ValueError("no word timestamps to re-segment")
                lines.append(jsonl_line(rechunk_jsonl_payload(payload)))
            if not lines:
                raise ValueError("empty transcript")
            if dest is None:
                sys.stdout.write("".join(lines))
                sys.stdout.flush()
            else:
                write_atomic(dest, "".join(lines))
            chunks = sum(len(json.loads(line)["chunks"]) for line in lines)
            eprint(
                f"✓ {path} -> {dest if dest is not None else '<stdout>'} ({chunks} chunks)"
            )
        except Exception as exc:
            eprint(f"✗ {path}: {exc}")
            failed += 1
    return EXIT_FAILED if failed else EXIT_OK


def _transcribe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-transcribe",
        description="Transcribe audio files locally (transcribe.cpp) or via ElevenLabs Scribe v2. "
        "Use `local-transcribe rechunk FILE.jsonl ...` to re-segment existing transcripts.",
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help="audio file(s)")
    parser.add_argument(
        "-o",
        "--output",
        help="output directory, .jsonl file (one input), or '-' for stdout "
        "(default: <FILE>.transcript.jsonl next to each input)",
    )
    add_transcribe_options(parser)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the acoustic pre-flight plan and exit without transcribing",
    )
    return parser


def _rechunk_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-transcribe rechunk",
        description="Re-segment existing .transcript.jsonl files from their word timestamps.",
    )
    parser.add_argument("files", nargs="+", metavar="FILE.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        help="output directory or '-' for stdout (default: rewrite in place atomically)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["rechunk"]:
        parser, command = _rechunk_parser(), cmd_rechunk
        argv = argv[1:]
    else:
        parser, command = _transcribe_parser(), cmd_transcribe
    args = parser.parse_args(argv)
    try:
        return command(args)
    except UsageError as exc:
        eprint(f"{parser.prog}: error: {exc}")
        return EXIT_USAGE
    except KeyboardInterrupt:
        eprint("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
