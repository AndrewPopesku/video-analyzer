"""Groq API provider for transcription, vision, and text generation."""

import base64
import time
from pathlib import Path

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from ..config import settings
from ..storage.database import increment_quota
from .provider import AIProvider


class GroqAPIError(Exception):
    """Raised when Groq API returns an error."""
    pass


def _should_retry(exception: Exception) -> bool:
    """Determine if we should retry based on exception type."""
    if isinstance(exception, GroqAPIError):
        return False

    error_str = str(exception).lower()
    # Common Groq error messages/types that shouldn't be retried
    non_retryable = [
        "api_key",
        "invalid",
        "unauthorized",
        "forbidden",
        "quota",
        "not found",
        "bad request",
        "400",
        "401",
        "403",
    ]
    return not any(term in error_str for term in non_retryable)


class GroqProvider(AIProvider):
    """Groq API provider using Whisper and Llama models."""

    def __init__(self):
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.client = Groq(api_key=settings.groq_api_key)
        self._last_request_time: float = 0

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < settings.groq_request_interval:
            time.sleep(settings.groq_request_interval - elapsed)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def transcribe(self, audio_path: Path) -> list[dict]:
        """Transcribe audio using Whisper and return timestamped segments."""
        self._wait_for_rate_limit()

        try:
            with open(audio_path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_path.name, f),
                    model=settings.groq_whisper_model,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            self._last_request_time = time.time()
            increment_quota()

            # Convert Groq response to our format
            segments = []
            if hasattr(transcription, "segments") and transcription.segments:
                for seg in transcription.segments:
                    # transcription.segments can be a list of objects or dicts
                    if isinstance(seg, dict):
                        segments.append({
                            "start": seg.get("start", 0.0),
                            "end": seg.get("end", 0.0),
                            "text": seg.get("text", ""),
                        })
                    else:
                        segments.append({
                            "start": getattr(seg, "start", 0.0),
                            "end": getattr(seg, "end", 0.0),
                            "text": getattr(seg, "text", ""),
                        })
            else:
                # Fallback if no segments
                segments.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": getattr(transcription, "text", str(transcription)),
                })

            return segments

        except Exception as e:
            if not _should_retry(e):
                raise GroqAPIError(f"Groq transcription error: {e}") from e
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze a single image using Llama Vision."""
        self._wait_for_rate_limit()

        try:
            image_data = base64.b64encode(image_path.read_bytes()).decode()
            mime_type = "image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"

            response = self.client.chat.completions.create(
                model=settings.groq_vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=1024,
            )

            self._last_request_time = time.time()
            increment_quota()

            return response.choices[0].message.content

        except Exception as e:
            if not _should_retry(e):
                raise GroqAPIError(f"Groq vision error: {e}") from e
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def analyze_images(self, image_paths: list[Path], prompt: str) -> str:
        """Analyze multiple images using Llama Vision (max 5 images)."""
        self._wait_for_rate_limit()

        try:
            # Groq supports up to 5 images
            paths_to_use = image_paths[:5]

            content = []
            for image_path in paths_to_use:
                image_data = base64.b64encode(image_path.read_bytes()).decode()
                mime_type = "image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                })

            content.append({"type": "text", "text": prompt})

            response = self.client.chat.completions.create(
                model=settings.groq_vision_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
            )

            self._last_request_time = time.time()
            increment_quota()

            return response.choices[0].message.content

        except Exception as e:
            if not _should_retry(e):
                raise GroqAPIError(f"Groq vision error: {e}") from e
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text using Llama."""
        self._wait_for_rate_limit()

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=settings.groq_text_model,
                messages=messages,
                max_tokens=4096,
            )

            self._last_request_time = time.time()
            increment_quota()

            return response.choices[0].message.content

        except Exception as e:
            if not _should_retry(e):
                raise GroqAPIError(f"Groq generation error: {e}") from e
            raise

    def detect_scenes_and_shots(
        self, image_paths: list[Path], timestamps: list[float]
    ) -> list[dict]:
        """Detect scenes and shots from keyframes with timestamps."""
        if not image_paths or not timestamps:
            return [
                {
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "label": "Full Video",
                    "description": "No frames available for analysis",
                }
            ]

        # Groq supports only 5 images at a time, so sample evenly
        max_frames = 5
        if len(image_paths) > max_frames:
            indices = [
                int(i * (len(image_paths) - 1) / (max_frames - 1))
                for i in range(max_frames)
            ]
            selected_images = [image_paths[i] for i in indices]
            selected_timestamps = [timestamps[i] for i in indices]
        else:
            selected_images = image_paths
            selected_timestamps = timestamps

        frame_info = "\n".join(
            [
                f"- Frame {i}: {ts:.1f}s ({int(ts // 60)}:{int(ts % 60):02d})"
                for i, ts in enumerate(selected_timestamps)
            ]
        )

        prompt = f"""Analyze these video frames to identify distinct scenes.

FRAME TIMESTAMPS:
{frame_info}

For each scene, provide:
- "start_time": Timestamp in seconds where scene begins
- "end_time": Timestamp in seconds where scene ends
- "label": Short scene title (3-5 words)
- "description": 1-2 sentence description of what's happening

Return JSON: {{"scenes": [...]}}

Use the ACTUAL timestamps above. Return ONLY the JSON object."""

        import json
        import re

        response = self.analyze_images(selected_images, prompt)
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                scenes = data.get("scenes", [])
                validated_scenes = []
                for s in scenes:
                    validated_scenes.append(
                        {
                            "start_time": float(s.get("start_time", 0)),
                            "end_time": float(s.get("end_time", 0)),
                            "label": s.get("label", "Scene"),
                            "description": s.get("description", ""),
                        }
                    )
                return validated_scenes if validated_scenes else []
            return []
        except Exception:
            return [
                {
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "label": "Full Video",
                    "description": "Scene detection unavailable",
                }
            ]

    def extract_entities(self, text: str, image_paths: list[Path]) -> list[dict]:
        """Extract named entities (Brands, Locations, People) from text and visuals."""
        if not text:
            return []

        system = "You are a named entity recognition expert. Identify Brands, Locations, and People mentioned in the transcript or visible in the visuals."
        prompt = f"""Transcript:
{text}

Identify unique entities mentioned:
- Brands (e.g., Apple, Nike, YouTube)
- Locations (e.g., New York, The Studio, California)
- People (e.g., Steve Jobs, The Host, MrBeast)

Return a JSON list of objects, each with:
- "name": Name of the entity
- "type": One of "Brand", "Location", "Person"
- "count": How many times it was mentioned/seen
- "description": A brief 1-sentence description or context

Example:
[
  {{"name": "Tesla", "type": "Brand", "count": 3, "description": "Electric vehicle manufacturer mentioned during the intro."}}
]

Return ONLY the JSON list."""

        # For now, we mainly use text as Groq vision is limited in batch size/complex reasoning over many images
        response = self.generate(prompt, system=system)
        try:
            import json
            import re
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(response)
        except Exception:
            return []

    def extract_topics_and_keywords(self, text: str) -> dict:
        """Extract high-level topics and keywords from text."""
        if not text:
            return {"topics": [], "keywords": []}

        system = "You are a video metadata expert. Extract high-level topics and keywords from the provided transcript."
        prompt = f"""Transcript:
{text}

Return a JSON object with:
- "topics": List of high-level topics discussed (max 5, each with "name" and "confidence")
- "keywords": List of important keywords (max 10, each with "text" and "confidence")

Example:
{{
  "topics": [
    {{"name": "Product Review", "confidence": 0.95}},
    {{"name": "Tech Comparison", "confidence": 0.8}}
  ],
  "keywords": [
    {{"text": "iPhone 15", "confidence": 1.0}},
    {{"text": "Camera quality", "confidence": 0.9}}
  ]
}}

Return ONLY the JSON object."""

        response = self.generate(prompt, system=system)
        try:
            # Simple JSON extraction
            import json
            import re
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(response)
        except Exception:
            return {"topics": [], "keywords": []}

    def perform_ocr(self, image_path: Path) -> list[dict]:
        """Perform OCR on an image and return text with bounding boxes."""
        prompt = """Identify ALL text visible in this image. For each text snippet, provide the text and its approximate location.

Return a JSON list of objects:
- "text": The detected text
- "left": X coordinate (0.0 to 1.0)
- "top": Y coordinate (0.0 to 1.0)
- "width": Relative width (0.0 to 1.0)
- "height": Relative height (0.0 to 1.0)

Return ONLY the JSON list."""

        response = self.analyze_image(image_path, prompt)
        try:
            import json
            import re
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(response)
        except Exception:
            return []
