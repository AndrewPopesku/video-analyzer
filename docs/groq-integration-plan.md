# Plan: Add Groq API Support

## Overview

Add Groq as an alternative AI provider alongside Gemini. Groq offers:
- **Whisper** for audio transcription (216x real-time speed, $0.04/hour)
- **Llama 4 Vision** for image analysis (Scout 17B or Maverick models)
- **Llama 3.3 70B** for text reasoning/hook detection

## Why Groq?

| Feature | Gemini | Groq |
|---------|--------|------|
| Transcription | Native (Gemini Flash) | Whisper Large v3 Turbo |
| Vision | Native | Llama 4 Scout/Maverick |
| Reasoning | Native | Llama 3.3 70B |
| Speed | Moderate | Very fast (LPU) |
| Free tier | 15 RPM, 1500/day | 30 RPM, 14,400/day |
| Reliability | Some issues | Generally stable |

## Implementation Plan

### 1. Add Groq dependency

```toml
# pyproject.toml
"groq>=0.33.0",
```

### 2. Update config.py

```python
class Settings(BaseSettings):
    # Provider selection
    ai_provider: str = "gemini"  # "gemini" or "groq"

    # Gemini settings (existing)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Groq settings (new)
    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_text_model: str = "llama-3.3-70b-versatile"

    # Groq rate limits (more generous)
    groq_rpm: int = 30
    groq_daily_limit: int = 14400
```

### 3. Create analyzer/groq.py

```python
"""Groq API client for transcription, vision, and reasoning."""

from groq import Groq
import base64

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    def transcribe(self, audio_path: Path) -> dict:
        """Transcribe audio using Whisper."""
        with open(audio_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                file=f,
                model=settings.groq_whisper_model,
                response_format="verbose_json",  # includes timestamps
                timestamp_granularities=["segment"],
            )
        return transcription

    def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze image using Llama Vision."""
        image_data = base64.b64encode(image_path.read_bytes()).decode()

        response = self.client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    def generate(self, prompt: str, system: str = None) -> str:
        """Generate text using Llama."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=settings.groq_text_model,
            messages=messages,
        )
        return response.choices[0].message.content
```

### 4. Create analyzer/provider.py (abstraction layer)

```python
"""AI provider abstraction layer."""

from abc import ABC, abstractmethod
from pathlib import Path

class AIProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[dict]:
        """Return list of {start, end, text} segments."""
        pass

    @abstractmethod
    def analyze_image(self, image_path: Path, prompt: str) -> str:
        pass

    @abstractmethod
    def analyze_images(self, image_paths: list[Path], prompt: str) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str, system: str = None) -> str:
        pass


def get_provider() -> AIProvider:
    """Get configured AI provider."""
    if settings.ai_provider == "groq":
        from .groq import GroqProvider
        return GroqProvider()
    else:
        from .gemini import GeminiProvider
        return GeminiProvider()
```

### 5. Update CLI

```python
@app.command()
def analyze(
    url: str,
    provider: str = typer.Option(None, "--provider", "-p", help="AI provider: gemini or groq"),
):
    # Override provider if specified
    if provider:
        settings.ai_provider = provider
```

### 6. Update existing analyzers

- `transcribe.py` → use `provider.transcribe()`
- `vision.py` → use `provider.analyze_image()` / `provider.analyze_images()`
- `hooks.py` → use `provider.generate()`
- `reranker.py` → use `provider.generate()`

## File Changes Summary

| File | Change |
|------|--------|
| `pyproject.toml` | Add `groq>=0.33.0` |
| `config.py` | Add Groq settings |
| `analyzer/groq.py` | New - Groq client |
| `analyzer/provider.py` | New - abstraction layer |
| `analyzer/gemini.py` | Wrap in GeminiProvider class |
| `analyzer/transcribe.py` | Use provider abstraction |
| `analyzer/vision.py` | Use provider abstraction |
| `analyzer/hooks.py` | Use provider abstraction |
| `search/reranker.py` | Use provider abstraction |
| `cli.py` | Add `--provider` flag |

## Usage After Implementation

```bash
# Use Gemini (default)
video-analyzer analyze https://youtube.com/...

# Use Groq
video-analyzer analyze https://youtube.com/... --provider groq

# Set default provider via env
export AI_PROVIDER=groq
video-analyzer analyze https://youtube.com/...
```

## Notes

- Groq Whisper returns timestamps natively (no need to ask LLM to generate them)
- Groq vision supports up to 5 images per request
- Groq has much higher rate limits than Gemini free tier
- Can mix providers in future (e.g., Groq for transcription, Gemini for vision)
