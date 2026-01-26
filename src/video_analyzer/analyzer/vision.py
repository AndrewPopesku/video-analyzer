"""Frame analysis using provider abstraction."""

import json
from pathlib import Path

from ..config import settings
from ..storage.media import list_frames
from .deduplication import (
    DeduplicationResult,
    apply_analysis_to_duplicates,
    get_deduplicator,
)
from .prompts import FRAME_ANALYSIS, FRAME_BATCH_ANALYSIS
from .provider import get_provider


def analyze_frame(frame_path: Path) -> dict:
    """Analyze a single frame."""
    provider = get_provider()
    response = provider.analyze_image(frame_path, FRAME_ANALYSIS)
    return _parse_frame_response(response)


def analyze_frames_batch(frame_paths: list[Path]) -> list[dict]:
    """Analyze multiple frames in a single request."""
    if not frame_paths:
        return []

    provider = get_provider()
    response = provider.analyze_images(frame_paths, FRAME_BATCH_ANALYSIS)
    return _parse_batch_response(response, len(frame_paths))


def analyze_frames(
    video_id: str,
    batch_size: int | None = None,
    enable_deduplication: bool | None = None,
) -> tuple[list[dict], DeduplicationResult | None]:
    """
    Analyze all frames for a video with optional deduplication.

    Args:
        video_id: The video identifier
        batch_size: Number of frames per batch (default from settings)
        enable_deduplication: Override deduplication setting (default from settings)

    Returns:
        Tuple of (analysis_results, dedup_result).
        dedup_result is None if deduplication was disabled.
    """
    if batch_size is None:
        batch_size = settings.max_frames_per_batch
    if enable_deduplication is None:
        enable_deduplication = settings.enable_deduplication

    all_frames = list_frames(video_id)
    if not all_frames:
        return [], None

    # Deduplication step
    dedup_result: DeduplicationResult | None = None
    frames_to_analyze = all_frames
    duplicate_map: dict[Path, Path] = {}

    if enable_deduplication:
        if settings.dedup_method == "embedding":
            deduplicator = get_deduplicator(
                method="embedding",
                threshold=settings.dedup_threshold,
            )
        else:
            deduplicator = get_deduplicator(
                method="hash",
                threshold=settings.dedup_threshold,
                algorithm=settings.dedup_algorithm,
                hash_size=settings.dedup_hash_size,
                sequential_only=settings.dedup_sequential_only,
            )
        dedup_result = deduplicator.deduplicate_frames(all_frames)
        frames_to_analyze = dedup_result.unique_frames
        duplicate_map = dedup_result.duplicate_map

    # Analyze only unique frames
    unique_results: list[dict] = []
    for i in range(0, len(frames_to_analyze), batch_size):
        batch = frames_to_analyze[i : i + batch_size]

        if len(batch) == 1:
            result = analyze_frame(batch[0])
            result["frame_path"] = str(batch[0])
            unique_results.append(result)
        else:
            results = analyze_frames_batch(batch)
            for j, result in enumerate(results):
                if i + j < len(frames_to_analyze):
                    result["frame_path"] = str(frames_to_analyze[i + j])
                unique_results.append(result)

    # Apply analysis to duplicates and build complete results
    if enable_deduplication and dedup_result:
        all_results = apply_analysis_to_duplicates(
            unique_results, duplicate_map, all_frames, settings.frame_rate
        )
    else:
        # No deduplication - add metadata to results
        all_results = []
        for i, result in enumerate(unique_results):
            result["frame_index"] = i
            result["timestamp"] = _frame_number_to_timestamp(i, settings.frame_rate)
            result["is_duplicate"] = False
            all_results.append(result)

    return all_results, dedup_result


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
    return [
        {
            "description": "Analysis failed",
            "scene_type": "other",
            "objects": [],
            "visual_hook": False,
        }
        for _ in range(expected_count)
    ]
