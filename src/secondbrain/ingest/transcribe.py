"""Optional local transcription with faster-whisper (install the 'whisper' extra)."""

import logging
from pathlib import Path

from secondbrain.config import Settings
from secondbrain.errors import TranscribeError
from secondbrain.models import Segment, Transcript

logger = logging.getLogger(__name__)

_GPU_HINT = (
    "If this is a GPU library error, either set WHISPER_DEVICE=cpu in .env or install the "
    "NVIDIA cuBLAS/cuDNN libraries (README, 'Whisper on the GPU')."
)


def transcribe_audio(audio_path: Path, settings: Settings) -> Transcript:
    """Transcribe one audio file. Slow on CPU: expect minutes for a long video."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "faster-whisper is not installed. Run: uv sync --extra whisper"
        ) from exc

    logger.info(
        "Transcribing %s with Whisper '%s' (device=%s, compute_type=%s)",
        audio_path.name,
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    try:
        model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root=str(settings.model_cache_dir),
        )
        raw_segments, info = model.transcribe(str(audio_path), vad_filter=True)
        segments = [
            Segment(text=s.text.strip(), start=float(s.start), end=float(s.end))
            for s in raw_segments
        ]
    except Exception as exc:
        raise TranscribeError(f"Whisper failed: {exc}. {_GPU_HINT}") from exc
    if not segments:
        raise TranscribeError("Whisper found no speech in the audio.")
    return Transcript(segments=segments, source="whisper", language=info.language)
