"""Local media file management."""

import json
import shutil
from pathlib import Path
from typing import Any

from ..config import settings


def get_video_dir(video_id: str) -> Path:
    """Get directory path for a video's files."""
    return settings.data_directory / video_id


def ensure_video_dir(video_id: str) -> Path:
    """Create and return video directory."""
    video_dir = get_video_dir(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "frames").mkdir(exist_ok=True)
    return video_dir


def get_video_path(video_id: str) -> Path:
    """Get path to downloaded video file."""
    return get_video_dir(video_id) / "video.mp4"


def get_audio_path(video_id: str) -> Path:
    """Get path to extracted audio file."""
    return get_video_dir(video_id) / "audio.mp3"


def get_frames_dir(video_id: str) -> Path:
    """Get path to frames directory."""
    return get_video_dir(video_id) / "frames"


def get_frame_path(video_id: str, frame_number: int) -> Path:
    """Get path to a specific frame image."""
    return get_frames_dir(video_id) / f"{frame_number:04d}.jpg"


def get_transcript_path(video_id: str) -> Path:
    """Get path to transcript JSON file."""
    return get_video_dir(video_id) / "transcript.json"


def get_analysis_path(video_id: str) -> Path:
    """Get path to analysis JSON file."""
    return get_video_dir(video_id) / "analysis.json"


def save_transcript(video_id: str, transcript: list[dict]) -> Path:
    """Save transcript to JSON file."""
    path = get_transcript_path(video_id)
    path.write_text(json.dumps(transcript, indent=2))
    return path


def load_transcript(video_id: str) -> list[dict] | None:
    """Load transcript from JSON file."""
    path = get_transcript_path(video_id)
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_analysis(video_id: str, analysis: dict[str, Any]) -> Path:
    """Save analysis results to JSON file."""
    path = get_analysis_path(video_id)
    path.write_text(json.dumps(analysis, indent=2))
    return path


def load_analysis(video_id: str) -> dict[str, Any] | None:
    """Load analysis results from JSON file."""
    path = get_analysis_path(video_id)
    if path.exists():
        return json.loads(path.read_text())
    return None


def list_frames(video_id: str) -> list[Path]:
    """List all frame files for a video."""
    frames_dir = get_frames_dir(video_id)
    if not frames_dir.exists():
        return []
    return sorted(frames_dir.glob("*.jpg"))


def cleanup_video(video_id: str, keep_analysis: bool = True) -> None:
    """Remove video files to save space."""
    video_dir = get_video_dir(video_id)
    if not video_dir.exists():
        return

    # Remove large files but keep analysis
    video_path = get_video_path(video_id)
    if video_path.exists():
        video_path.unlink()

    if not keep_analysis:
        shutil.rmtree(video_dir)


def get_storage_stats() -> dict:
    """Get storage usage statistics."""
    data_dir = settings.data_directory
    if not data_dir.exists():
        return {"total_size": 0, "video_count": 0, "videos": []}

    videos = []
    total_size = 0

    for video_dir in data_dir.iterdir():
        if video_dir.is_dir() and video_dir.name != ".gitkeep":
            size = sum(f.stat().st_size for f in video_dir.rglob("*") if f.is_file())
            videos.append({"video_id": video_dir.name, "size_bytes": size})
            total_size += size

    return {
        "total_size": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "video_count": len(videos),
        "videos": videos,
    }
