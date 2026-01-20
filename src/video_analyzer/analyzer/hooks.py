"""Hook detection using content analysis."""

import json
from dataclasses import dataclass

from .provider import get_provider
from .transcribe import get_transcript_range


@dataclass
class DetectedHook:
    """A detected attention hook."""

    timestamp: float
    end_timestamp: float | None
    hook_type: str  # visual, audio, content, pattern_interrupt
    description: str
    confidence: float
    transcript_snippet: str | None = None


HOOK_DETECTION_PROMPT = """Analyze this video content for attention hooks - moments designed to capture and retain viewer attention.

TRANSCRIPT (with timestamps):
{transcript}

FRAME ANALYSIS:
{frame_analysis}

Identify hooks in these categories:

1. **Visual Hooks**: Scene changes, text overlays, face close-ups, motion spikes
2. **Audio Hooks**: Energy shifts, music drops, silence-to-speech, questions to viewer
3. **Content Patterns**:
   - Pattern interrupts ("But here's the thing...", "Plot twist...")
   - Open loops / cliffhangers ("What happens next will surprise you")
   - Contrast and surprise elements
   - Direct viewer engagement ("Have you ever...")

4. **First 30 Seconds Analysis**: Specifically analyze how the video hooks viewers in the opening

Return a JSON array of detected hooks:
[
  {{
    "timestamp": 0.0,
    "end_timestamp": 5.0,
    "hook_type": "content",
    "description": "Opens with a provocative question to engage viewers",
    "confidence": 0.85,
    "transcript_snippet": "Have you ever wondered why..."
  }}
]

Guidelines:
- Focus on the most impactful hooks (quality over quantity)
- Confidence should reflect how likely this is to retain viewer attention
- Include transcript snippet when the hook involves speech
- timestamp is start of hook in seconds

Return ONLY the JSON array."""


INTRO_ANALYSIS_PROMPT = """Analyze this video's opening hook (first 30-60 seconds).

TRANSCRIPT:
{transcript}

FRAMES (in order):
{frame_descriptions}

Evaluate the intro's hook effectiveness:

1. **Hook Type**: What technique is used? (question, bold claim, story tease, visual surprise, etc.)
2. **Speed to Hook**: How quickly does the video grab attention?
3. **Promise**: What does the intro promise the viewer?
4. **Thumbnail-Title Alignment**: Based on the opening, does it likely match expectations set by title/thumbnail?

Return JSON:
{{
  "hook_technique": "description of the hook technique used",
  "hook_timestamp": 0.0,
  "speed_rating": "fast|medium|slow",
  "promise": "what the video promises to deliver",
  "effectiveness_score": 0.0-1.0,
  "improvement_suggestion": "how the hook could be stronger"
}}

Return ONLY the JSON object."""


def detect_hooks(
    transcript: list[dict],
    frame_analysis: list[dict],
) -> list[DetectedHook]:
    """Detect attention hooks from transcript and frame analysis."""
    provider = get_provider()

    # Format transcript for prompt
    transcript_text = _format_transcript(transcript)

    # Format frame analysis for prompt
    frame_text = _format_frame_analysis(frame_analysis)

    response = provider.generate(
        HOOK_DETECTION_PROMPT.format(
            transcript=transcript_text,
            frame_analysis=frame_text,
        ),
        system="You are a video content strategist expert at identifying attention hooks. Always return valid JSON.",
    )

    hooks = _parse_hooks_response(response)

    return hooks


def analyze_intro(
    transcript: list[dict],
    frame_analysis: list[dict],
    intro_duration: float = 60.0,
) -> dict:
    """Analyze the video's intro/hook effectiveness."""
    provider = get_provider()

    # Filter to intro segment
    intro_transcript = [s for s in transcript if s["start"] < intro_duration]
    intro_frames = [f for f in frame_analysis if f.get("timestamp", 0) < intro_duration]

    transcript_text = _format_transcript(intro_transcript)
    frame_text = "\n".join(
        f"[{f.get('timestamp', 0):.1f}s] {f.get('description', 'No description')}"
        for f in intro_frames
    )

    response = provider.generate(
        INTRO_ANALYSIS_PROMPT.format(
            transcript=transcript_text,
            frame_descriptions=frame_text,
        ),
        system="You are a video content strategist. Always return valid JSON.",
    )

    return _parse_intro_response(response)


def _format_transcript(transcript: list[dict]) -> str:
    """Format transcript for prompt."""
    lines = []
    for seg in transcript:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")
    return "\n".join(lines)


def _format_frame_analysis(frame_analysis: list[dict]) -> str:
    """Format frame analysis for prompt."""
    lines = []
    for frame in frame_analysis:
        timestamp = frame.get("timestamp", 0)
        desc = frame.get("description", "No description")
        scene = frame.get("scene_type", "unknown")
        hook = "VISUAL HOOK" if frame.get("visual_hook") else ""
        lines.append(f"[{timestamp:.1f}s] ({scene}) {desc} {hook}".strip())
    return "\n".join(lines)


def _parse_hooks_response(response: str) -> list[DetectedHook]:
    """Parse hooks detection response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [
                DetectedHook(
                    timestamp=h.get("timestamp", 0),
                    end_timestamp=h.get("end_timestamp"),
                    hook_type=h.get("hook_type", "unknown"),
                    description=h.get("description", ""),
                    confidence=h.get("confidence", 0.5),
                    transcript_snippet=h.get("transcript_snippet"),
                )
                for h in data
            ]
    except json.JSONDecodeError:
        pass

    return []


def _parse_intro_response(response: str) -> dict:
    """Parse intro analysis response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    return {
        "hook_technique": "unknown",
        "effectiveness_score": 0.0,
        "improvement_suggestion": "Analysis failed",
    }
