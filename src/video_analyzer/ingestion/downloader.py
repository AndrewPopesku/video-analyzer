"""YouTube video downloader using yt-dlp."""

import re
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path

from ..storage.media import ensure_video_dir, get_video_path


@dataclass
class VideoInfo:
    """Video metadata from YouTube."""

    video_id: str
    url: str
    title: str
    channel: str
    duration_seconds: int
    thumbnail_url: str | None = None
    description: str | None = None


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_video_info(url: str) -> VideoInfo:
    """Get video metadata without downloading."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    result = subprocess.run(
        [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            url,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    return VideoInfo(
        video_id=video_id,
        url=url,
        title=data.get("title", "Unknown"),
        channel=data.get("channel", data.get("uploader", "Unknown")),
        duration_seconds=int(data.get("duration", 0)),
        thumbnail_url=data.get("thumbnail"),
        description=data.get("description"),
    )


def download_video(url: str, max_quality: str = "720p") -> tuple[VideoInfo, Path]:
    """Download video and return info and path."""
    info = get_video_info(url)
    video_dir = ensure_video_dir(info.video_id)
    output_path = get_video_path(info.video_id)

    if output_path.exists():
        return info, output_path

    # Download with yt-dlp
    subprocess.run(
        [
            "yt-dlp",
            "-f", f"bestvideo[height<={max_quality[:-1]}]+bestaudio/best[height<={max_quality[:-1]}]",
            "-o", str(output_path),
            "--merge-output-format", "mp4",
            url,
        ],
        check=True,
    )

    return info, output_path
