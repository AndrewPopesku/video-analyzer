"""Frame deduplication using perceptual hashing or embeddings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image


@dataclass
class DeduplicationResult:
    """Result of frame deduplication analysis."""

    unique_frames: list[Path]
    duplicate_map: dict[Path, Path]  # {duplicate: reference}
    hashes: dict[Path, str]  # {path: hash_hex}
    stats: dict = field(default_factory=dict)


class FrameDeduplicator:
    """Detects duplicate/similar frames to reduce LLM API calls."""

    def __init__(
        self,
        threshold: int = 8,
        algorithm: str = "ahash",
        hash_size: int = 8,
        sequential_only: bool = True,
    ):
        """
        Initialize deduplicator.

        Args:
            threshold: Hamming distance threshold. Lower = stricter matching.
                      For hash_size=8: max distance is 64.
                      Recommended: 8-12 for ahash, 10-15 for phash/dhash.
            algorithm: Hash algorithm - "ahash" (default, best for video),
                      "phash", "dhash", or "colorhash".
            hash_size: Hash size for precision (8 or 16). Ignored for colorhash.
            sequential_only: If True, only compare adjacent frames (O(n)).
                           If False, compare all frames (O(n²)).
        """
        self.threshold = threshold
        self.algorithm = algorithm
        self.hash_size = hash_size
        self.sequential_only = sequential_only

    def compute_hash(self, frame_path: Path) -> str:
        """Compute perceptual hash for a frame image."""
        img = Image.open(frame_path)

        if self.algorithm == "dhash":
            h = imagehash.dhash(img, hash_size=self.hash_size)
        elif self.algorithm == "phash":
            h = imagehash.phash(img, hash_size=self.hash_size)
        elif self.algorithm == "colorhash":
            h = imagehash.colorhash(img)
        else:  # default to ahash (average hash) - best for video with animation
            h = imagehash.average_hash(img, hash_size=self.hash_size)

        return str(h)

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes."""
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2

    def is_similar(self, hash1: str, hash2: str) -> bool:
        """Check if two frames are similar based on threshold."""
        return self.hamming_distance(hash1, hash2) <= self.threshold

    def deduplicate_frames(self, frame_paths: list[Path]) -> DeduplicationResult:
        """
        Analyze frames and identify duplicates.

        Returns DeduplicationResult with:
        - unique_frames: Frames that need LLM analysis
        - duplicate_map: Maps duplicate frames to their reference frame
        - hashes: All computed hashes
        - stats: Deduplication statistics
        """
        if not frame_paths:
            return DeduplicationResult([], {}, {}, {"total_frames": 0})

        # Compute hashes for all frames
        hashes: dict[Path, str] = {}
        for path in frame_paths:
            hashes[path] = self.compute_hash(path)

        unique_frames: list[Path] = []
        duplicate_map: dict[Path, Path] = {}

        if self.sequential_only:
            # Compare only adjacent frames (O(n))
            unique_frames.append(frame_paths[0])

            for i in range(1, len(frame_paths)):
                current = frame_paths[i]
                previous = frame_paths[i - 1]

                if self.is_similar(hashes[current], hashes[previous]):
                    # Current is duplicate - find the original reference
                    ref = previous
                    while ref in duplicate_map:
                        ref = duplicate_map[ref]
                    duplicate_map[current] = ref
                else:
                    unique_frames.append(current)
        else:
            # Compare all frames (O(n²))
            for path in frame_paths:
                is_dup = False
                for unique in unique_frames:
                    if self.is_similar(hashes[path], hashes[unique]):
                        duplicate_map[path] = unique
                        is_dup = True
                        break
                if not is_dup:
                    unique_frames.append(path)

        total = len(frame_paths)
        duplicates = len(duplicate_map)
        stats = {
            "total_frames": total,
            "unique_frames": len(unique_frames),
            "duplicates": duplicates,
            "dedup_ratio": duplicates / total if total > 0 else 0,
        }

        return DeduplicationResult(unique_frames, duplicate_map, hashes, stats)


def apply_analysis_to_duplicates(
    analysis_results: list[dict],
    duplicate_map: dict[Path, Path],
    all_frame_paths: list[Path],
    fps: float,
) -> list[dict]:
    """
    Create analysis entries for duplicate frames by copying from reference.

    Args:
        analysis_results: Results from analyzing unique frames
        duplicate_map: {duplicate_path: reference_path}
        all_frame_paths: All original frame paths in order
        fps: Frame rate for timestamp calculation

    Returns:
        Complete analysis list with all frames in original order
    """
    # Build lookup from path to analysis
    path_to_analysis: dict[Path, dict] = {}
    for result in analysis_results:
        frame_path = result.get("frame_path")
        if frame_path:
            path_to_analysis[Path(frame_path)] = result

    complete_results: list[dict] = []
    for i, path in enumerate(all_frame_paths):
        timestamp = i / fps if fps > 0 else 0.0

        if path in duplicate_map:
            # Copy analysis from reference frame
            ref_path = duplicate_map[path]
            ref_analysis = path_to_analysis.get(ref_path, {})

            dup_analysis = ref_analysis.copy()
            dup_analysis["frame_index"] = i
            dup_analysis["frame_path"] = str(path)
            dup_analysis["timestamp"] = timestamp
            dup_analysis["is_duplicate"] = True
            dup_analysis["reference_frame_path"] = str(ref_path)
            complete_results.append(dup_analysis)
        else:
            # Original analysis - ensure it has the right metadata
            analysis = path_to_analysis.get(path, {}).copy()
            analysis["frame_index"] = i
            analysis["frame_path"] = str(path)
            analysis["timestamp"] = timestamp
            analysis["is_duplicate"] = False
            complete_results.append(analysis)

    return complete_results


class EmbeddingDeduplicator:
    """Detects duplicate frames using CLIP embeddings for semantic similarity."""

    _model = None  # Class-level model cache

    def __init__(
        self,
        threshold: float = 0.95,
        model_name: str = "clip-ViT-B-32",
    ):
        """
        Initialize embedding-based deduplicator.

        Args:
            threshold: Cosine similarity threshold (0.0-1.0). Higher = stricter.
                      Recommended: 0.90-0.95 for scene deduplication.
            model_name: CLIP model to use. Options:
                      - "clip-ViT-B-32" (default, fast)
                      - "clip-ViT-L-14" (more accurate, slower)
        """
        self.threshold = threshold
        self.model_name = model_name
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Load model once and cache it."""
        if EmbeddingDeduplicator._model is None:
            from sentence_transformers import SentenceTransformer

            EmbeddingDeduplicator._model = SentenceTransformer(self.model_name)

    @property
    def model(self):
        return EmbeddingDeduplicator._model

    def compute_embedding(self, frame_path: Path) -> np.ndarray:
        """Compute CLIP embedding for a frame."""
        img = Image.open(frame_path)
        return self.model.encode(img)

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def is_similar(self, emb1: np.ndarray, emb2: np.ndarray) -> bool:
        """Check if two frames are similar based on threshold."""
        return self.cosine_similarity(emb1, emb2) >= self.threshold

    def deduplicate_frames(self, frame_paths: list[Path]) -> DeduplicationResult:
        """
        Analyze frames using embeddings and identify duplicates.

        Uses non-sequential comparison (O(n²)) since embeddings capture
        semantic similarity that may not be present in adjacent frames.
        """
        if not frame_paths:
            return DeduplicationResult([], {}, {}, {"total_frames": 0})

        # Compute embeddings for all frames
        embeddings: dict[Path, np.ndarray] = {}
        for path in frame_paths:
            embeddings[path] = self.compute_embedding(path)

        unique_frames: list[Path] = []
        unique_embeddings: list[np.ndarray] = []
        duplicate_map: dict[Path, Path] = {}

        # Non-sequential comparison - compare each frame to all unique frames
        for path in frame_paths:
            emb = embeddings[path]
            is_dup = False

            for i, unique_emb in enumerate(unique_embeddings):
                if self.is_similar(emb, unique_emb):
                    duplicate_map[path] = unique_frames[i]
                    is_dup = True
                    break

            if not is_dup:
                unique_frames.append(path)
                unique_embeddings.append(emb)

        total = len(frame_paths)
        duplicates = len(duplicate_map)

        # Store embeddings as hex strings for compatibility with hash-based approach
        hashes = {path: emb.tobytes().hex()[:32] for path, emb in embeddings.items()}

        stats = {
            "total_frames": total,
            "unique_frames": len(unique_frames),
            "duplicates": duplicates,
            "dedup_ratio": duplicates / total if total > 0 else 0,
            "method": "embedding",
        }

        return DeduplicationResult(unique_frames, duplicate_map, hashes, stats)


def get_deduplicator(
    method: str = "hash",
    threshold: Optional[float] = None,
    **kwargs,
):
    """
    Factory function to get the appropriate deduplicator.

    Args:
        method: "hash" for perceptual hashing, "embedding" for CLIP embeddings
        threshold: Similarity threshold (Hamming distance for hash, cosine for embedding)
        **kwargs: Additional arguments passed to the deduplicator

    Returns:
        FrameDeduplicator or EmbeddingDeduplicator instance
    """
    if method == "embedding":
        return EmbeddingDeduplicator(
            threshold=threshold if threshold is not None else 0.95,
            **kwargs,
        )
    else:
        return FrameDeduplicator(
            threshold=int(threshold) if threshold is not None else 10,
            **kwargs,
        )
