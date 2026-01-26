"""Configuration settings for video analyzer."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection
    ai_provider: Literal["gemini", "groq"] = "gemini"

    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"

    # Gemini rate limiting
    gemini_rpm: int = 15  # requests per minute
    gemini_daily_limit: int = 1500  # requests per day
    gemini_request_interval: float = 4.0  # seconds between requests

    # Groq API
    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_text_model: str = "openai/gpt-oss-120b"

    # Groq rate limiting (more generous)
    groq_rpm: int = 30
    groq_daily_limit: int = 14400
    groq_request_interval: float = 2.0  # seconds between requests

    # Paths
    data_dir: Path = Path("data")
    db_path: Path = Path("data/video_analyzer.db")

    # Processing
    frame_rate: float = 0.2  # frames per second (1 frame per 5 seconds)
    audio_chunk_minutes: int = 10  # chunk audio into 10-minute segments
    max_frames_per_batch: int = 5  # frames to analyze in single request

    # Frame deduplication
    enable_deduplication: bool = True
    dedup_method: Literal["hash", "embedding"] = "embedding"  # embedding is more accurate
    dedup_threshold: float = 0.95  # Cosine similarity for embedding (0-1), Hamming for hash
    dedup_algorithm: Literal["ahash", "phash", "dhash", "colorhash"] = "ahash"  # Only for hash method
    dedup_hash_size: int = 8  # Hash size (8 or 16), only for hash method
    dedup_sequential_only: bool = True  # Only for hash method (embedding always uses non-sequential)

    @property
    def data_directory(self) -> Path:
        """Get absolute data directory path."""
        return self.data_dir.resolve()

    @property
    def database_path(self) -> Path:
        """Get absolute database path."""
        return self.db_path.resolve()


settings = Settings()
