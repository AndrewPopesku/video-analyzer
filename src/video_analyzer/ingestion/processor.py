"""Media processing using FFmpeg."""

import subprocess
from pathlib import Path

from ..config import settings
from ..storage.media import (
    ensure_video_dir,
    get_audio_path,
    get_frames_dir,
    get_video_path,
)


def extract_audio(video_id: str, video_path: Path | None = None) -> Path:
    """Extract audio from video as MP3."""
    if video_path is None:
        video_path = get_video_path(video_id)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ensure_video_dir(video_id)
    audio_path = get_audio_path(video_id)

    if audio_path.exists():
        return audio_path

    subprocess.run(
        [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # no video
            "-acodec", "libmp3lame",
            "-ab", "128k",
            "-ar", "44100",
            "-y",  # overwrite
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )

    return audio_path


def extract_frames(
    video_id: str,
    video_path: Path | None = None,
    fps: float | None = None,
) -> list[Path]:
    """Extract frames from video at specified rate."""
    if video_path is None:
        video_path = get_video_path(video_id)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if fps is None:
        fps = settings.frame_rate

    ensure_video_dir(video_id)
    frames_dir = get_frames_dir(video_id)

    # Check if frames already extracted
    existing_frames = sorted(frames_dir.glob("*.jpg"))
    if existing_frames:
        return existing_frames

    # Extract frames with FFmpeg
    output_pattern = str(frames_dir / "%04d.jpg")

    subprocess.run(
        [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",  # quality (2 is high quality)
            "-y",
            output_pattern,
        ],
        check=True,
        capture_output=True,
    )

    return sorted(frames_dir.glob("*.jpg"))


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def chunk_audio(
    video_id: str,
    chunk_minutes: int | None = None,
) -> list[Path]:
    """Split audio into chunks for processing long videos."""
    if chunk_minutes is None:
        chunk_minutes = settings.audio_chunk_minutes

    audio_path = get_audio_path(video_id)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = get_audio_duration(audio_path)
    chunk_seconds = chunk_minutes * 60

    # If audio is shorter than chunk size, return original
    if duration <= chunk_seconds:
        return [audio_path]

    chunks_dir = audio_path.parent / "audio_chunks"
    chunks_dir.mkdir(exist_ok=True)

    # Check for existing chunks
    existing_chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
    if existing_chunks:
        return existing_chunks

    # Split into chunks
    chunks = []
    chunk_num = 0
    start_time = 0

    while start_time < duration:
        chunk_path = chunks_dir / f"chunk_{chunk_num:03d}.mp3"

        subprocess.run(
            [
                "ffmpeg",
                "-i", str(audio_path),
                "-ss", str(start_time),
                "-t", str(chunk_seconds),
                "-acodec", "libmp3lame",
                "-ab", "128k",
                "-y",
                str(chunk_path),
            ],
            check=True,
            capture_output=True,
        )

        chunks.append(chunk_path)
        chunk_num += 1
        start_time += chunk_seconds

    return chunks
