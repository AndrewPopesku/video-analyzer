"""Gemini API provider with rate limiting and retry logic."""

import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from ..config import settings
from ..storage.database import get_today_quota, increment_quota
from .provider import AIProvider


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class QuotaExceeded(Exception):
    """Raised when daily quota is exceeded."""
    pass


class GeminiAPIError(Exception):
    """Raised when Gemini API returns an error."""
    pass


def _should_retry(exception: Exception) -> bool:
    """Determine if we should retry based on exception type."""
    if isinstance(exception, GeminiAPIError):
        return False
    error_str = str(exception).lower()
    non_retryable = ["api_key", "invalid", "unauthorized", "forbidden", "quota"]
    return not any(term in error_str for term in non_retryable)


class GeminiProvider(AIProvider):
    """Gemini API provider with rate limiting."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self._last_request_time: float = 0

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < settings.gemini_request_interval:
            time.sleep(settings.gemini_request_interval - elapsed)

    def _check_quota(self) -> None:
        """Check if daily quota is exceeded."""
        quota = get_today_quota()
        if quota.request_count >= settings.gemini_daily_limit:
            raise QuotaExceeded(
                f"Daily quota exceeded: {quota.request_count}/{settings.gemini_daily_limit} requests"
            )

        if quota.request_count >= settings.gemini_daily_limit * 0.8:
            remaining = settings.gemini_daily_limit - quota.request_count
            print(f"Warning: Only {remaining} API requests remaining today")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def transcribe(self, audio_path: Path) -> list[dict]:
        """Transcribe audio using Gemini and return timestamped segments."""
        self._check_quota()
        self._wait_for_rate_limit()

        prompt = """Transcribe this audio with timestamps. Return a JSON array of segments.

Each segment should have:
- "start": start time in seconds (float)
- "end": end time in seconds (float)
- "text": the transcribed text

Example format:
[
  {"start": 0.0, "end": 5.2, "text": "Hello and welcome to this video"},
  {"start": 5.2, "end": 10.1, "text": "Today we're going to talk about..."}
]

Important:
- Create segments at natural speech boundaries (sentences, pauses)
- Each segment should be 3-10 seconds long
- Include all spoken words accurately
- Return ONLY the JSON array, no other text"""

        try:
            file_size = audio_path.stat().st_size
            max_inline_size = 15 * 1024 * 1024

            config = types.GenerateContentConfig(
                system_instruction="You are a precise audio transcription assistant. Always return valid JSON.",
            )

            if file_size < max_inline_size:
                audio_data = audio_path.read_bytes()
                mime_type = "audio/mpeg" if audio_path.suffix.lower() == ".mp3" else "audio/wav"

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=audio_data, mime_type=mime_type),
                        prompt,
                    ],
                    config=config,
                )
            else:
                audio_file = self.client.files.upload(file=str(audio_path))
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[audio_file, prompt],
                    config=config,
                )

            self._last_request_time = time.time()
            increment_quota()

            return self._parse_transcript_response(response.text)

        except Exception as e:
            if not _should_retry(e):
                raise GeminiAPIError(f"Gemini transcription error: {e}") from e
            raise

    def _parse_transcript_response(self, response: str) -> list[dict]:
        """Parse Gemini response into transcript segments."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            segments = json.loads(text)
            if isinstance(segments, list):
                for seg in segments:
                    if not all(k in seg for k in ["start", "end", "text"]):
                        raise ValueError(f"Invalid segment: {seg}")
                return segments
        except (json.JSONDecodeError, ValueError):
            pass

        return [{"start": 0.0, "end": 0.0, "text": response}]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Analyze a single image."""
        self._check_quota()
        self._wait_for_rate_limit()

        try:
            image_data = image_path.read_bytes()
            mime_type = "image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    prompt,
                ],
            )

            self._last_request_time = time.time()
            increment_quota()

            return response.text

        except Exception as e:
            if not _should_retry(e):
                raise GeminiAPIError(f"Gemini vision error: {e}") from e
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def analyze_images(self, image_paths: list[Path], prompt: str) -> str:
        """Analyze multiple images."""
        self._check_quota()
        self._wait_for_rate_limit()

        try:
            parts = []
            for image_path in image_paths:
                image_data = image_path.read_bytes()
                mime_type = "image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))

            parts.append(prompt)

            # Max 3600 images for Gemini, but let's be reasonable
            response = self.client.models.generate_content(
                model=self.model,
                contents=parts,
            )

            self._last_request_time = time.time()
            increment_quota()

            return response.text

        except Exception as e:
            if not _should_retry(e):
                raise GeminiAPIError(f"Gemini vision error: {e}") from e
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text from a prompt."""
        self._check_quota()
        self._wait_for_rate_limit()

        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
            ) if system else None

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            self._last_request_time = time.time()
            increment_quota()

            return response.text

        except Exception as e:
            if not _should_retry(e):
                raise GeminiAPIError(f"Gemini generation error: {e}") from e
            raise

    def detect_scenes_and_shots(
        self, image_paths: list[Path], timestamps: list[float]
    ) -> list[dict]:
        """Detect scenes and shots from a list of keyframes with their timestamps."""
        if not image_paths or not timestamps:
            return [
                {
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "label": "Full Video",
                    "description": "No frames available for analysis",
                }
            ]

        # Sample up to 30 frames evenly across the video for good coverage
        max_frames = 30
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

        # Build timestamp info for the prompt
        frame_info = "\n".join(
            [
                f"- Frame {i}: {ts:.1f}s ({int(ts // 60)}:{int(ts % 60):02d})"
                for i, ts in enumerate(selected_timestamps)
            ]
        )

        prompt = f"""Analyze this sequence of video frames to identify distinct scenes.

FRAME TIMESTAMPS:
{frame_info}

A 'scene' is a coherent segment where the content, setting, or topic remains consistent.
Look for significant changes in:
- Location or setting
- Visual style (e.g., switch from talking head to B-roll footage)
- Topic or activity being shown
- Camera angle or shot type

For each scene, provide:
- "start_time": The timestamp (in seconds) where this scene begins
- "end_time": The timestamp (in seconds) where this scene ends
- "label": A short title (3-5 words) describing the scene type
- "description": A detailed 1-2 sentence description of what is happening in this scene

Return a JSON object with a "scenes" array:
{{
  "scenes": [
    {{
      "start_time": 0.0,
      "end_time": 45.0,
      "label": "Introduction",
      "description": "The host greets viewers from a home office setup, introducing today's topic."
    }}
  ]
}}

Important:
- Use the ACTUAL timestamps from the list above, not frame indices
- Ensure scenes are contiguous (end_time of one scene should match start_time of next)
- Provide meaningful descriptions that explain WHAT is happening
- Return ONLY the JSON object."""

        response = self.analyze_images(selected_images, prompt)
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                scenes = data.get("scenes", [])
                # Validate and return scenes with correct structure
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
                    "label": "Analysis Failed",
                    "description": "Could not parse scene detection response",
                }
            ]

    def extract_entities(self, text: str, image_paths: list[Path]) -> list[dict]:
        """Extract named entities (Brands, Locations, People) from text and visuals."""
        if not text:
            return []

        # Gemini can handle multiple images, but we'll limit to a few key ones if there are many
        # for entity extraction
        selected_images = image_paths[:5] if len(image_paths) > 5 else image_paths

        prompt = f"""Transcript:
{text}

Identify unique entities (Brands, Locations, People) mentioned in the transcript or visible in the frames.

Return a JSON list of objects:
- "name": Name of the entity
- "type": One of "Brand", "Location", "Person"
- "count": How many times it was mentioned/seen
- "description": Context or description

Return ONLY the JSON list."""

        response = self.analyze_images(selected_images, prompt)
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

Return ONLY the JSON object."""

        response = self.generate(prompt, system=system)
        try:
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



# Legacy compatibility - keep get_client for existing code during transition
_client: GeminiProvider | None = None


def get_client() -> GeminiProvider:
    """Get or create Gemini client instance (legacy compatibility)."""
    global _client
    if _client is None:
        _client = GeminiProvider()
    return _client
