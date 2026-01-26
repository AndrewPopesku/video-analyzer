"""Interfaces (Protocols) for dependency injection and testing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class VideoInfo:
    """Video metadata from a media source."""

    video_id: str
    url: str
    title: str
    channel: str
    duration_seconds: int
    thumbnail_url: str | None = None
    description: str | None = None


class MediaDownloader(Protocol):
    """Protocol for downloading media from external sources."""

    def get_info(self, url: str) -> VideoInfo:
        """Get video metadata without downloading."""
        ...

    def download(self, url: str) -> tuple[VideoInfo, Path]:
        """Download video and return info and path."""
        ...


class MediaProcessor(Protocol):
    """Protocol for processing media files (audio/video extraction)."""

    def extract_audio(self, video_id: str, video_path: Path) -> Path:
        """Extract audio from video file."""
        ...

    def extract_frames(self, video_id: str, video_path: Path) -> list[Path]:
        """Extract frames from video file."""
        ...
