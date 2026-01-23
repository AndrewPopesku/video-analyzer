"""AI provider abstraction layer."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..config import settings


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[dict]:
        """Transcribe audio and return list of segments.

        Returns:
            List of dicts with keys: start, end, text
        """
        pass

    @abstractmethod
    def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze a single image with a prompt."""
        pass

    @abstractmethod
    def analyze_images(self, image_paths: list[Path], prompt: str) -> str:
        """Analyze multiple images with a prompt."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def detect_scenes_and_shots(
        self, image_paths: list[Path], timestamps: list[float]
    ) -> list[dict]:
        """Detect scenes and shots from a list of keyframes with their timestamps.

        Args:
            image_paths: List of paths to keyframe images
            timestamps: List of timestamps (in seconds) corresponding to each frame

        Returns:
            List of dicts with keys: start_time, end_time, label, description
        """
        pass

    @abstractmethod
    def extract_entities(self, text: str, image_paths: list[Path]) -> list[dict]:
        """Extract named entities (Brands, Locations, People) from text and visuals."""
        pass

    @abstractmethod
    def extract_topics_and_keywords(self, text: str) -> dict:
        """Extract high-level topics and keywords from text."""
        pass

    @abstractmethod
    def perform_ocr(self, image_path: Path) -> list[dict]:
        """Perform OCR on an image and return text with bounding boxes."""
        pass


# Singleton provider instance
_provider: AIProvider | None = None


def get_provider() -> AIProvider:
    """Get or create the configured AI provider."""
    global _provider

    if _provider is None:
        if settings.ai_provider == "groq":
            from .groq import GroqProvider
            _provider = GroqProvider()
        else:
            from .gemini import GeminiProvider
            _provider = GeminiProvider()

    return _provider


def reset_provider() -> None:
    """Reset the provider (useful when switching providers)."""
    global _provider
    _provider = None
