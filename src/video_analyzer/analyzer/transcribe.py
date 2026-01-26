"""Audio transcription using provider abstraction."""

from ..config import settings
from ..ingestion.processor import chunk_audio, get_audio_duration
from ..storage.media import get_audio_path, load_transcript, save_transcript
from .provider import get_provider


def transcribe_audio(video_id: str, force: bool = False) -> list[dict]:
    """Transcribe audio and return timestamped segments."""
    # Check for cached transcript
    if not force:
        cached = load_transcript(video_id)
        if cached:
            return cached

    audio_path = get_audio_path(video_id)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    provider = get_provider()

    # Check if we need to chunk
    duration = get_audio_duration(audio_path)
    chunk_seconds = settings.audio_chunk_minutes * 60

    if duration <= chunk_seconds:
        # Single chunk - transcribe directly
        segments = provider.transcribe(audio_path)
    else:
        # Multiple chunks - transcribe each and merge
        chunks = chunk_audio(video_id)
        segments = []
        time_offset = 0.0

        for chunk_path in chunks:
            chunk_segments = provider.transcribe(chunk_path)

            # Adjust timestamps for chunk offset
            for seg in chunk_segments:
                seg["start"] += time_offset
                seg["end"] += time_offset
                segments.append(seg)

            # Update offset for next chunk
            chunk_duration = get_audio_duration(chunk_path)
            time_offset += chunk_duration

    # Save transcript
    save_transcript(video_id, segments)

    return segments


def get_transcript_at_time(segments: list[dict], timestamp: float) -> str | None:
    """Get transcript text at a specific timestamp."""
    for seg in segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return seg["text"]
    return None


def get_transcript_range(
    segments: list[dict],
    start_time: float,
    end_time: float,
) -> str:
    """Get combined transcript text for a time range."""
    texts = []
    for seg in segments:
        # Check for overlap
        if seg["end"] >= start_time and seg["start"] <= end_time:
            texts.append(seg["text"])
    return " ".join(texts)
