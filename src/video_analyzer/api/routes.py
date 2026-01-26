"""FastAPI routes for video analyzer API."""

from fastapi import FastAPI, HTTPException

from video_analyzer.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    SearchRequest,
    SearchResultResponse,
    VideoListResponse,
    VideoResponse,
)

from ..ingestion.downloader import YouTubeDownloader
from ..ingestion.processor import FFmpegProcessor
from ..analyzer.provider import get_provider
from ..service import VideoAnalyzerService, VideoAlreadyAnalyzedError

app = FastAPI(
    title="Video Analyzer API",
    description="YouTube video analysis using Gemini/Groq AI",
    version="0.1.0",
)


def _create_service() -> VideoAnalyzerService:
    """Create a VideoAnalyzerService with default dependencies."""
    return VideoAnalyzerService(
        downloader=YouTubeDownloader(),
        processor=FFmpegProcessor(),
        provider=get_provider(),
    )


@app.get("/")
async def root():
    """API root endpoint."""
    return {"message": "Video Analyzer API", "version": "0.1.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    """
    Analyze a YouTube video.

    This endpoint runs the full analysis pipeline synchronously.
    For long videos, consider using background tasks in production.
    """
    service = _create_service()

    try:
        result = service.analyze_video(request.url, skip_if_exists=False)
        return AnalyzeResponse(
            video_id=result.video.video_id,
            title=result.video.title,
            status=result.video.status,
            transcript_count=result.transcript_count,
            frame_count=result.frame_count,
            hook_count=result.hook_count,
        )
    except VideoAlreadyAnalyzedError as e:
        # Return existing video info
        video = service.get_video(e.video_id)
        if video:
            return AnalyzeResponse(
                video_id=video.video_id,
                title=video.title,
                status=video.status,
                transcript_count=0,  # Not re-counted
                frame_count=0,
                hook_count=0,
            )
        raise HTTPException(status_code=404, detail="Video not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos", response_model=list[VideoListResponse])
async def list_videos():
    """List all analyzed videos."""
    service = _create_service()
    videos = service.list_videos()
    return [
        VideoListResponse(
            video_id=v.video_id,
            title=v.title,
            channel=v.channel,
            duration_seconds=v.duration_seconds,
            status=v.status,
            created_at=v.created_at.isoformat() if v.created_at else None,
        )
        for v in videos
    ]


@app.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(video_id: str):
    """Get video details."""
    service = _create_service()
    video = service.get_video(video_id)

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return VideoResponse(
        video_id=video.video_id,
        url=video.url,
        title=video.title,
        channel=video.channel,
        duration_seconds=video.duration_seconds,
        status=video.status,
        created_at=video.created_at.isoformat() if video.created_at else None,
        analyzed_at=video.analyzed_at.isoformat() if video.analyzed_at else None,
    )


@app.post("/search", response_model=list[SearchResultResponse])
async def search(request: SearchRequest):
    """Search across analyzed videos."""
    service = _create_service()
    results = service.search_videos(request.query, request.limit)
    return [
        SearchResultResponse(
            video_id=r.video_id,
            video_title=r.video_title,
            timestamp=r.timestamp,
            content_type=r.content_type,
            content=r.content,
            relevance_score=r.relevance_score,
        )
        for r in results
    ]
