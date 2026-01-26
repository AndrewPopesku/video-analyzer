"""Hook detection using content analysis."""

import json
from dataclasses import dataclass

from . import prompts
from .provider import get_provider


@dataclass
class DetectedHook:
    """A detected attention hook."""

    timestamp: float
    end_timestamp: float | None
    hook_type: str  # visual, audio, content, pattern_interrupt
    description: str
    confidence: float
    transcript_snippet: str | None = None


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
        prompts.hook_detection(transcript_text, frame_text),
        system=prompts.SYSTEM_HOOK_DETECTION,
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
        prompts.intro_analysis(transcript_text, frame_text),
        system=prompts.SYSTEM_INTRO_ANALYSIS,
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
