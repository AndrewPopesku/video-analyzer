"""Ingestion module for downloading and processing videos."""

from .downloader import download_video, get_video_info
from .processor import extract_audio, extract_frames

__all__ = ["download_video", "get_video_info", "extract_audio", "extract_frames"]
