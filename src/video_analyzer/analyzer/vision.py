"""Frame analysis using provider abstraction."""

import json
from pathlib import Path

from ..config import settings
from ..storage.media import list_frames
from .provider import get_provider


FRAME_ANALYSIS_PROMPT = """Analyze this video frame and provide a detailed description.

Return a JSON object with:
- "description": A 1-2 sentence description of what's happening in the frame
- "scene_type": One of: "talking_head", "b_roll", "text_overlay", "product_shot", "screen_recording", "animation", "other"
- "objects": List of key objects/elements visible (max 5)
- "text_visible": Any text visible in the frame (or null if none)
- "emotion": If a person is visible, their apparent emotion (or null)
- "visual_hook": Boolean - is this frame visually attention-grabbing?
- "hook_reason": If visual_hook is true, briefly explain why

Example:
{
  "description": "A person speaking directly to camera with an excited expression",
  "scene_type": "talking_head",
  "objects": ["person", "microphone", "ring light"],
  "text_visible": null,
  "emotion": "excited",
  "visual_hook": true,
  "hook_reason": "Strong emotion and direct eye contact"
}

Return ONLY the JSON object, no other text."""


BATCH_ANALYSIS_PROMPT = """Analyze these video frames in sequence. They are from the same video, shown in chronological order.

For each frame, provide:
- "frame_index": The frame number (0-indexed)
- "description": What's happening in this frame
- "scene_type": One of: "talking_head", "b_roll", "text_overlay", "product_shot", "screen_recording", "animation", "other"
- "objects": Key objects visible (max 3)
- "visual_hook": Boolean - is this frame attention-grabbing?
- "transition": Did the scene change significantly from the previous frame?

Return a JSON array of analysis objects, one per frame.
Return ONLY the JSON array, no other text."""


def analyze_frame(frame_path: Path) -> dict:
    """Analyze a single frame."""
    provider = get_provider()
    response = provider.analyze_image(frame_path, FRAME_ANALYSIS_PROMPT)
    return _parse_frame_response(response)


def analyze_frames_batch(frame_paths: list[Path]) -> list[dict]:
    """Analyze multiple frames in a single request."""
    if not frame_paths:
        return []

    provider = get_provider()
    response = provider.analyze_images(frame_paths, BATCH_ANALYSIS_PROMPT)
    return _parse_batch_response(response, len(frame_paths))


def analyze_frames(video_id: str, batch_size: int | None = None) -> list[dict]:
    """Analyze all frames for a video."""
    if batch_size is None:
        batch_size = settings.max_frames_per_batch

    frames = list_frames(video_id)
    if not frames:
        return []

    all_results = []

    # Process in batches
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]

        if len(batch) == 1:
            result = analyze_frame(batch[0])
            result["frame_index"] = i
            result["frame_path"] = str(batch[0])
            result["timestamp"] = _frame_number_to_timestamp(i, settings.frame_rate)
            all_results.append(result)
        else:
            results = analyze_frames_batch(batch)
            for j, result in enumerate(results):
                frame_idx = i + j
                result["frame_index"] = frame_idx
                result["frame_path"] = str(frames[frame_idx]) if frame_idx < len(frames) else None
                result["timestamp"] = _frame_number_to_timestamp(frame_idx, settings.frame_rate)
                all_results.append(result)

    return all_results


def _frame_number_to_timestamp(frame_number: int, fps: float) -> float:
    """Convert frame number to timestamp in seconds."""
    return frame_number / fps


def _parse_frame_response(response: str) -> dict:
    """Parse single frame analysis response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback
    return {
        "description": response[:200],
        "scene_type": "other",
        "objects": [],
        "visual_hook": False,
    }


def _parse_batch_response(response: str, expected_count: int) -> list[dict]:
    """Parse batch frame analysis response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        results = json.loads(text)
        if isinstance(results, list):
            return results
    except json.JSONDecodeError:
        pass

    # Fallback - return empty results
    return [{"description": "Analysis failed", "scene_type": "other", "objects": [], "visual_hook": False} for _ in range(expected_count)]
