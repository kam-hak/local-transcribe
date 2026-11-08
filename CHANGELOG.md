# Changelog

## 0.1.0 (2026-11-08)

Initial 0.1.0 pre-release.

- `local-transcribe` CLI with local Metal transcription through `transcribe.cpp`.
- Acoustic pre-flight with Silero VAD and VoxLingua107 language ID, and per-language routing to Parakeet Unified En, Parakeet TDT v3, Qwen3-ASR, or Whisper Large v3 Turbo.
- Automatic language detection with a 15% minimum share (up to 20% on short clips), or a 5% minimum share for languages passed with `--langs`.
- Local speaker diarization with Sortformer 4spk v2.1.
- Optional ElevenLabs Scribe v2 cloud engine, called directly over HTTPS with no other SDK.
- `.env` loading from the working directory and from `~/.config/local-transcribe/.env`.
- Script-aware sentence and paragraph chunking, and `local-transcribe rechunk` for existing transcripts.
- `.transcript.jsonl` output with `schema_version: 2` and a provenance record.
