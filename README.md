# local-transcribe

Local-first audio transcription for on Silicone Macs. 

A CLI-tool that includes voice activity detection, language identification, speaker diarization, and semantic chunking. All data remains local and on-device through Metal (`transcribe.cpp`). The output is a `.transcript.jsonl` files.

## Privacy and sensitive recordings

> [!WARNING]
> Local transcription runs entirely on your Mac and never sends audio over the network. This covers `--engine local` and the default `--engine auto`, unless you accept the cloud prompt.
>
> ElevenLabs Scribe v2 (`--engine cloud`) is an optional cloud fallback for languages where open local models have a word error rate above 5%. **ElevenLabs is not private.** When you select the cloud engine, `local-transcribe` uploads the audio file to ElevenLabs servers at `https://api.elevenlabs.io/v1/speech-to-text`.
>
> Do not pass `--engine cloud` or accept the interactive cloud prompt for confidential, regulated, or sensitive recordings unless your organization authorizes cloud processing. For sensitive data, pass `--engine local`. Running with `-y` and `--engine auto` also never uploads, because `-y` skips the prompt and keeps every file local.

## Install

With [Homebrew](https://brew.sh), two commands install the CLI and its dependencies:

```bash
brew install uv ffmpeg
uv tool install git+https://github.com/kam-hak/local-transcribe.git
```

`uv tool install` puts `local-transcribe` on your `PATH` in its own environment. The `transcribe-cpp` dependency pulls the prebuilt `transcribe-cpp-native` wheel, which bundles the Metal build of `transcribe.cpp`, so you do not need a compiler or Xcode.

You need:

- An Apple Silicon Mac (M1 or newer) with Metal GPU support.
- `ffmpeg` on your `PATH`.
- Disk space for GGUF models. The CLI downloads each model from Hugging Face the first time a file needs it, into `~/.local/share/local-transcribe/models`. English with diarization needs about 900 MB for Parakeet Unified En (731 MB), Sortformer (139 MB), VoxLingua107 (27 MB), and Silero VAD (1 MB). If the Hex dictation app is installed, the CLI also reuses models that it finds in `~/Library/Application Support/Hex/models`.

## Optional cloud setup (ElevenLabs)

The cloud engine needs an ElevenLabs API key. Save the key in one of three places:

- `~/.config/local-transcribe/.env`, which applies in every directory.
- `.env` in the directory where you run `local-transcribe`.
- An exported `ELEVENLABS_API_KEY` variable in your shell.

To save the key globally, run:

```bash
mkdir -p ~/.config/local-transcribe
echo 'ELEVENLABS_API_KEY=your_api_key_here' >> ~/.config/local-transcribe/.env
chmod 600 ~/.config/local-transcribe/.env
```

If a variable is set in more than one place, the shell environment wins, then the working-directory `.env`, then the global file.

`local-transcribe` calls ElevenLabs only when you pass `--engine cloud` or choose `[c]` at an interactive prompt.


## Features

- **Local-first by default.** Every stage runs offline on the Apple Silicon GPU. The cloud engine runs only when you pass `--engine cloud` or accept an interactive prompt.
- **Language auto-detection or explicit `--langs`.**
  - **Auto-detect (default)** samples speech windows with Silero VAD + VoxLingua107 (`p >= 0.65`) and keeps languages above a **15% share** (20% on clips under 150s) to filter out false-positive noise.
  - **Explicit (`-l en` or `-l en,es`)** restricts detection to the listed languages with a **5% threshold**.
- **Automatic model routing.** Selects the highest-accuracy local GGUF model for the detected language set:

  | Languages detected | Local model |
  | --- | --- |
  | English only | Parakeet Unified En |
  | 25 European languages (including English) | Parakeet TDT v3 |
  | Chinese, Cantonese, Japanese, Korean, Thai, Vietnamese, Indonesian, Malay, Filipino, Arabic, Hindi, Turkish, Persian | Qwen3-ASR |
  | Any other language or unsupported mix | Whisper Large v3 Turbo (flags file if expected WER > 5%) |

- **Speaker diarization (`-d`).** Labels words by speaker (`A`, `B`, `C`, ...) using NVIDIA Sortformer 4spk v2.1 locally or Scribe v2 in the cloud.
- **Semantic chunking.** Groups tokens into sentences and ~300-character paragraph chunks with script-aware spacing, silence re-anchoring, and pause-aware punctuation boundaries. Run `local-transcribe rechunk` to re-segment existing `.transcript.jsonl` files in place.

- **Dry run through `plan`**

The plan flag (`--plan`) performs a dry run, stating the audio duration, speech duration, language(s) detected, ASR model expected, and diarizer. 

```console
$ local-transcribe --plan -d interview.wav
Analyzing interview.wav ...

Pre-flight plan
File           Duration  Speech  Languages  Engine / ASR model                         Diarizer
-------------  --------  ------  ---------  -----------------------------------------  ----------
interview.wav  00:15     00:11   en: 83.0%  local: parakeet-unified-en-0.6b-Q8_0.gguf  Sortformer
```

Without `--plan`, the CLI writes the transcript next to the input as `interview.wav.transcript.jsonl` and prints a receipt on `stderr` that lists every step that produced it.

## Models

The CLI runs GGUF conversions of these models from [handy-computer on Hugging Face](https://huggingface.co/handy-computer). The dates are when each model was published on Hugging Face, and the benchmark figures come from each model card.

| Model | Upstream ID | Released | License | Benchmark |
| --- | --- | --- | --- | --- |
| Parakeet Unified En 0.6B | [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | Apr 2026 | NVIDIA Open Model License | Unified FastConformer RNN-T for English. Offline WER averages 5.9% across the eight Hugging Face Open ASR Leaderboard datasets, and is 1.63% on LibriSpeech test-clean. |
| Parakeet TDT 0.6B v3 | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Aug 2025 | CC-BY-4.0 | Transcribes 25 European languages and detects the language itself. |
| Qwen3-ASR 0.6B | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Jan 2026 | Apache-2.0 | Language ID and ASR for 30 languages and 22 Chinese dialects, across Asian, Middle Eastern, and European languages. GGUF: [`handy-computer/Qwen3-ASR-0.6B-gguf`](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf). |
| Whisper Large v3 Turbo | [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | Oct 2024 | MIT | An 809M-parameter Whisper Large v3 with the decoder cut from 32 layers to 4. Error rates vary widely by language, and long-tail languages are much less accurate than English. |
| Sortformer 4spk v2.1 | [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | Oct 2025 | NVIDIA Open Model License | Streaming speaker diarization for up to four speakers. |
| VoxLingua107 ECAPA-TDNN | [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) | 2021 | Apache-2.0 | Spoken language identification for 107 languages, with a 6.7% error rate on the VoxLingua107 development set. |

## Quickstart

1. Inspect the language mix and the planned models without transcribing.

   ```bash
   local-transcribe --plan -d recording.mp3
   ```

2. Transcribe locally with speaker diarization.

   ```bash
   local-transcribe -d recording.mp3
   ```

3. Restrict detection to the languages you expect. The CLI then ignores false positives from other languages.

   ```bash
   local-transcribe -d -l en,ja interview.m4a
   ```

4. Stream JSONL to `stdout` for a Unix pipeline. Tables, prompts, and receipts go to `stderr`, so `stdout` carries only transcript lines.

   ```bash
   local-transcribe -y -o - recording.wav | jq .provenance
   ```

5. Re-chunk an existing transcript to apply the current segmenter to it.

   ```bash
   local-transcribe rechunk recording.mp3.transcript.jsonl
   ```

## CLI reference

```text
local-transcribe [options] FILE [FILE ...]
local-transcribe rechunk [-o OUTPUT] FILE.jsonl [FILE.jsonl ...]
```

| Option | Meaning |
| --- | --- |
| `-o`, `--output` | Output directory, a `.jsonl` path (one input only), or `-` for `stdout`. The default is `<FILE>.transcript.jsonl` next to each input. For `rechunk`, a directory or `-`. The default rewrites each file in place. |
| `-d`, `--diarize` | Label words by speaker. |
| `-l`, `--langs` | Comma-separated ISO 639-1 codes, such as `en,es`. Language ID considers only these languages. Without this option, the CLI detects languages automatically. |
| `--engine` | `auto` (default), `local`, or `cloud`. `auto` plans local and asks before it switches flagged files to the cloud. |
| `--vocab` | A file with one term per line. Lines that start with `#` are comments. Local models that support vocabulary receive every term. Scribe v2 receives the first 100 terms as keyterms. |
| `--plan` | Print the pre-flight plan and exit without transcribing. |
| `-y`, `--yes` | Never prompt. Files flagged for the cloud stay local. |

When a file is flagged for the cloud in an interactive run, the CLI offers three choices. You can continue locally (the default), switch flagged files to Scribe v2, or restrict languages and re-plan.

| Exit code | Meaning |
| --- | --- |
| `0` | Every file succeeded. |
| `1` | At least one file failed. The other files still finish. |
| `2` | Usage error, such as a missing input or `--engine cloud` without an API key. |

The CLI reads environment variables from the shell, from `.env` in the working directory, and from `~/.config/local-transcribe/.env`.

| Variable | Purpose |
| --- | --- |
| `LOCAL_TRANSCRIBE_MODELS_DIR` | Directory for downloaded GGUF models. The default is `~/.local/share/local-transcribe/models`. |
| `ELEVENLABS_API_KEY` | Enables `--engine cloud` and the cloud choice in the prompt. |
| `ELEVENLABS_BASE_URL` | ElevenLabs API base URL. The default is `https://api.elevenlabs.io/v1`. |
| `LOCAL_TRANSCRIBE_CLOUD_MODEL` | Scribe model ID. The default is `scribe_v2`. |

## Output schema

Each `.transcript.jsonl` file holds one JSON object per line with `schema_version: 2`. All `start` and `end` values are integer milliseconds.

| Field | Content |
| --- | --- |
| `schema_version` | `2` |
| `text` | Full transcript, joined from the chunks. |
| `chunks` | Paragraph chunks. Each has `speaker`, `start`, `end`, `text`, and `language`. |
| `utterances` | Speaker turns, with the same fields as chunks. |
| `words` | Each word with `text`, `start`, `end`, `speaker`, and `language`. |
| `source_file`, `mime_type`, `file_size_bytes` | The input audio file. |
| `duration_seconds` | Audio duration. |
| `language_code` | Dominant ISO 639-1 language. |
| `provenance` | `audio_prep`, `engine`, `asr_model`, `diarizer_model`, `diarization_enabled`, `allowed_langs`, `lang_distribution`, `vocab_count`, `vocab_sha256`, `chunking_strategy`, and `elapsed_ms`. |

## Development

```bash
git clone https://github.com/kam-hak/local-transcribe.git
cd local-transcribe
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -m "not metal"
pytest
```

`pytest -m "not metal"` runs the unit tests on any platform, and CI runs the same command. A plain `pytest` also runs the `metal` tests, which need the GGUF models and `ffmpeg`. The `metal` tests synthesize a 15-second English clip with macOS `say`. To test against your own recording instead, set `LOCAL_TRANSCRIBE_SAMPLE_AUDIO` to its path.

## License

MIT. See [LICENSE](LICENSE).
