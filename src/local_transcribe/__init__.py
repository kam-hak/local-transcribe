"""Local-first transcription core: data contracts, planning and segmentation."""

from importlib import import_module

# Importing transcribe_cpp initializes Metal, so engine symbols load on first access.
_LAZY_EXPORTS = {
    "MODEL_URLS": "local_transcribe.models",
    "ModelCache": "local_transcribe.models",
    "resolve_model_path": "local_transcribe.models",
    "AUDIO_PREP_DESC": "local_transcribe.audio",
    "SAMPLE_RATE": "local_transcribe.audio",
    "SUPPORTED_AUDIO_EXTENSIONS": "local_transcribe.audio",
    "decode_audio_pcm": "local_transcribe.audio",
    "pcm_duration_ms": "local_transcribe.audio",
    "analyze_audio": "local_transcribe.analyze",
    "execute_plan": "local_transcribe.transcribe",
    "transcribe_audio_file": "local_transcribe.transcribe",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
