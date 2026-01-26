"""Video Analyzer Service - Core business logic."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from sqlmodel import select

from .analyzer.provider import AIProvider
from .analyzer.transcribe import transcribe_audio
from .analyzer.vision import analyze_frames
from .analyzer.hooks import detect_hooks
from .interfaces import MediaDownloader, MediaProcessor, VideoInfo
from .search.reranker import search_videos as search_videos_impl
from .storage.database import get_session, get_all_videos, get_video_by_video_id
from .storage.models import (
    Video,
    Transcript,
    Frame,
    Hook,
    Scene,
    Entity,
    Topic,
    Keyword,
    OCRResult,
)


@dataclass
class AnalysisResult:
    """Result of video analysis."""

    video: Video
    transcript_count: int
    frame_count: int
    hook_count: int
    dedup_stats: dict | None = None


@dataclass
class SearchResult:
    """Search result item."""

    video_id: str
    video_title: str
    timestamp: float
    content_type: str
    content: str
    relevance_score: float


class VideoAlreadyAnalyzedError(Exception):
    """Raised when video has already been analyzed."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        super().__init__(f"Video already analyzed: {video_id}")


class VideoAnalyzerService:
    """Service layer for video analysis operations."""

    def __init__(
        self,
        downloader: MediaDownloader,
        processor: MediaProcessor,
        provider: AIProvider,
    ):
        self.downloader = downloader
        self.processor = processor
        self.provider = provider

    def analyze_video(
        self,
        url: str,
        on_progress: Callable[[str], None] | None = None,
        skip_if_exists: bool = True,
    ) -> AnalysisResult:
        """
        Full video analysis pipeline.

        Args:
            url: YouTube video URL
            on_progress: Optional callback for progress updates
            skip_if_exists: If True, raise VideoAlreadyAnalyzedError for completed videos

        Returns:
            AnalysisResult with video and statistics

        Raises:
            VideoAlreadyAnalyzedError: If video already analyzed and skip_if_exists=True
        """

        def progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # Step 1: Get video info
        progress("Getting video info...")
        info = self.downloader.get_info(url)

        # Step 2: Check if already analyzed
        with get_session() as session:
            existing = session.exec(
                select(Video).where(Video.video_id == info.video_id)
            ).first()
            if existing and existing.status == "completed" and skip_if_exists:
                raise VideoAlreadyAnalyzedError(info.video_id)

        # Step 3: Create/update video record
        video_db_id = self._create_or_update_video(info, url)

        try:
            # Step 4: Download video
            progress("Downloading video...")
            _, video_path = self.downloader.download(url)

            # Step 5: Extract audio
            progress("Extracting audio...")
            _ = self.processor.extract_audio(info.video_id, video_path)

            # Step 6: Extract frames
            progress("Extracting frames...")
            _ = self.processor.extract_frames(info.video_id, video_path)

            # Step 7: Transcribe audio
            progress("Transcribing audio...")
            transcript = transcribe_audio(info.video_id)

            # Step 8: Save transcript
            self._save_transcript(video_db_id, transcript)

            # Step 9: Analyze frames with deduplication
            progress("Analyzing frames...")
            frame_analysis, dedup_result = analyze_frames(info.video_id)

            # Step 10: Save frame analysis
            self._save_frames(video_db_id, frame_analysis, dedup_result)

            # Step 11: Detect hooks
            progress("Detecting hooks...")
            hooks = detect_hooks(transcript, frame_analysis)

            # Step 12: Save hooks
            self._save_hooks(video_db_id, hooks)

            # Step 13: Extract Topics and Keywords
            progress("Extracting topics and keywords...")
            topics_data = self.provider.extract_topics_and_keywords(transcript)
            self._save_topics_keywords(video_db_id, topics_data)

            # Step 14: Extract Entities
            progress("Extracting entities...")
            keyframe_paths = [
                Path(f["frame_path"]) for f in frame_analysis if f.get("frame_path")
            ][:10]
            entities = self.provider.extract_entities(transcript, keyframe_paths)
            self._save_entities(video_db_id, entities)

            # Step 15: Perform OCR
            progress("Performing OCR...")
            self._perform_ocr(video_db_id, frame_analysis)

            # Step 16: Scene Detection
            progress("Detecting scenes...")
            self._detect_scenes(video_db_id, frame_analysis)

            # Step 17: Mark as completed
            progress("Finalizing...")
            self._mark_completed(video_db_id)

            # Get final video record
            with get_session() as session:
                video = session.get(Video, video_db_id)

            # Build dedup stats
            dedup_stats = None
            if dedup_result and dedup_result.stats:
                dedup_stats = dedup_result.stats

            return AnalysisResult(
                video=video,
                transcript_count=len(transcript),
                frame_count=len(frame_analysis),
                hook_count=len(hooks),
                dedup_stats=dedup_stats,
            )

        except Exception:
            # Mark as failed on any error
            self._mark_failed(video_db_id)
            raise

    def get_video(self, video_id: str) -> Video | None:
        """Retrieve a video by its YouTube ID."""
        return get_video_by_video_id(video_id)

    def list_videos(self) -> list[Video]:
        """List all analyzed videos."""
        return get_all_videos()

    def search_videos(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search across all analyzed videos."""
        results = search_videos_impl(query, limit)
        return [
            SearchResult(
                video_id=r.video_id,
                video_title=r.video_title,
                timestamp=r.timestamp,
                content_type=r.content_type,
                content=r.content,
                relevance_score=r.relevance_score,
            )
            for r in results
        ]

    # --- Private helper methods ---

    def _create_or_update_video(self, info: VideoInfo, url: str) -> int:
        """Create or update video record, return DB ID."""
        with get_session() as session:
            video = session.exec(
                select(Video).where(Video.video_id == info.video_id)
            ).first()
            if not video:
                video = Video(
                    video_id=info.video_id,
                    url=url,
                    title=info.title,
                    channel=info.channel,
                    duration_seconds=info.duration_seconds,
                    thumbnail_url=info.thumbnail_url,
                    status="processing",
                )
                session.add(video)
            else:
                video.status = "processing"
            session.commit()
            session.refresh(video)
            return video.id

    def _save_transcript(self, video_db_id: int, transcript: list[dict]) -> None:
        """Save transcript segments to database."""
        with get_session() as session:
            for seg in transcript:
                t = Transcript(
                    video_id=video_db_id,
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"],
                )
                session.add(t)
            session.commit()

    def _save_frames(
        self, video_db_id: int, frame_analysis: list[dict], dedup_result
    ) -> None:
        """Save frame analysis to database."""
        with get_session() as session:
            # First pass: save all frames and build path->id map
            frame_id_map: dict[str, int] = {}
            for fa in frame_analysis:
                frame_path = fa.get("frame_path", "")
                # Get hash from dedup_result if available
                phash = None
                if dedup_result and dedup_result.hashes:
                    phash = dedup_result.hashes.get(Path(frame_path))

                f = Frame(
                    video_id=video_db_id,
                    timestamp=fa.get("timestamp", 0),
                    frame_path=frame_path,
                    description=fa.get("description"),
                    objects=str(fa.get("objects", [])),
                    scene_type=fa.get("scene_type"),
                    perceptual_hash=phash,
                    is_duplicate=fa.get("is_duplicate", False),
                )
                session.add(f)
                session.flush()  # Get the ID
                frame_id_map[frame_path] = f.id

            # Second pass: set reference_frame_id for duplicates
            for fa in frame_analysis:
                if fa.get("is_duplicate") and fa.get("reference_frame_path"):
                    frame_path = fa.get("frame_path", "")
                    ref_path = fa.get("reference_frame_path")
                    if frame_path in frame_id_map and ref_path in frame_id_map:
                        frame = session.get(Frame, frame_id_map[frame_path])
                        if frame:
                            frame.reference_frame_id = frame_id_map[ref_path]

            session.commit()

    def _save_hooks(self, video_db_id: int, hooks: list) -> None:
        """Save detected hooks to database."""
        with get_session() as session:
            for h in hooks:
                hook = Hook(
                    video_id=video_db_id,
                    timestamp=h.timestamp,
                    end_timestamp=h.end_timestamp,
                    hook_type=h.hook_type,
                    description=h.description,
                    confidence=h.confidence,
                    transcript_snippet=h.transcript_snippet,
                )
                session.add(hook)
            session.commit()

    def _save_topics_keywords(self, video_db_id: int, topics_data: dict) -> None:
        """Save extracted topics and keywords."""
        with get_session() as session:
            for t_data in topics_data.get("topics", []):
                topic = Topic(
                    video_id=video_db_id,
                    name=t_data["name"],
                    confidence=t_data.get("confidence", 1.0),
                )
                session.add(topic)
            for k_data in topics_data.get("keywords", []):
                keyword = Keyword(
                    video_id=video_db_id,
                    text=k_data["text"],
                    confidence=k_data.get("confidence", 1.0),
                )
                session.add(keyword)
            session.commit()

    def _save_entities(self, video_db_id: int, entities: list[dict]) -> None:
        """Save extracted entities."""
        with get_session() as session:
            for e_data in entities:
                entity = Entity(
                    video_id=video_db_id,
                    name=e_data["name"],
                    type=e_data["type"],
                    count=e_data.get("count", 1),
                    description=e_data.get("description"),
                )
                session.add(entity)
            session.commit()

    def _perform_ocr(self, video_db_id: int, frame_analysis: list[dict]) -> None:
        """Perform OCR on relevant frames."""
        with get_session() as session:
            # Map timestamps to frame IDs
            stmt = select(Frame).where(Frame.video_id == video_db_id)
            db_frames = session.exec(stmt).all()
            frame_map = {f.timestamp: f.id for f in db_frames}

            for fa in frame_analysis:
                if fa.get("text_visible") or fa.get("frame_index", 0) % 10 == 0:
                    image_path = Path(fa["frame_path"])
                    ocr_data = self.provider.perform_ocr(image_path)
                    for ocr in ocr_data:
                        res = OCRResult(
                            video_id=video_db_id,
                            frame_id=frame_map.get(fa["timestamp"]),
                            text=ocr["text"],
                            left=ocr.get("left", 0.0),
                            top=ocr.get("top", 0.0),
                            width=ocr.get("width", 0.0),
                            height=ocr.get("height", 0.0),
                        )
                        session.add(res)
            session.commit()

    def _detect_scenes(self, video_db_id: int, frame_analysis: list[dict]) -> None:
        """Detect scenes from frames."""
        all_frame_data = [
            (Path(f["frame_path"]), f.get("timestamp", 0.0))
            for f in frame_analysis
            if f.get("frame_path")
        ]
        scene_frame_paths = [f[0] for f in all_frame_data]
        scene_timestamps = [f[1] for f in all_frame_data]

        scenes_data = self.provider.detect_scenes_and_shots(
            scene_frame_paths, scene_timestamps
        )

        with get_session() as session:
            for s_data in scenes_data:
                scene = Scene(
                    video_id=video_db_id,
                    start_time=s_data.get("start_time", 0.0),
                    end_time=s_data.get("end_time", 0.0),
                    label=s_data.get("label", "Scene"),
                    description=s_data.get("description", ""),
                )
                session.add(scene)
            session.commit()

    def _mark_completed(self, video_db_id: int) -> None:
        """Mark video as completed."""
        with get_session() as session:
            video = session.get(Video, video_db_id)
            video.status = "completed"
            video.analyzed_at = datetime.utcnow()
            session.add(video)
            session.commit()

    def _mark_failed(self, video_db_id: int) -> None:
        """Mark video as failed."""
        with get_session() as session:
            video = session.get(Video, video_db_id)
            if video:
                video.status = "failed"
                session.add(video)
                session.commit()
