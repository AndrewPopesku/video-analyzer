"""FastAPI routes for video analyzer API."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..storage.database import get_all_videos, get_video_by_video_id, get_today_quota
from ..search.reranker import search_videos

app = FastAPI(
    title="Video Analyzer API",
    description="YouTube video analysis using Gemini API",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SearchResultResponse(BaseModel):
    video_id: str
    video_title: str
    timestamp: float
    content_type: str
    content: str
    relevance_score: float


@app.get("/")
async def root():
    """API root endpoint."""
    return {"message": "Video Analyzer API", "version": "0.1.0"}


@app.get("/videos")
async def list_videos():
    """List all analyzed videos."""
    videos = get_all_videos()
    return [
        {
            "video_id": v.video_id,
            "title": v.title,
            "channel": v.channel,
            "duration_seconds": v.duration_seconds,
            "status": v.status,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in videos
    ]


@app.get("/videos/{video_id}")
async def get_video(video_id: str):
    """Get video details."""
    video = get_video_by_video_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "video_id": video.video_id,
        "url": video.url,
        "title": video.title,
        "channel": video.channel,
        "duration_seconds": video.duration_seconds,
        "status": video.status,
        "created_at": video.created_at.isoformat() if video.created_at else None,
        "analyzed_at": video.analyzed_at.isoformat() if video.analyzed_at else None,
    }


@app.post("/search", response_model=list[SearchResultResponse])
async def search(request: SearchRequest):
    """Search across analyzed videos."""
    results = search_videos(request.query, request.limit)
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


@app.get("/quota")
async def quota():
    """Get today's API quota usage."""
    q = get_today_quota()
    return {
        "date": q.date,
        "request_count": q.request_count,
        "daily_limit": 1500,
        "remaining": 1500 - q.request_count,
        "usage_percent": round(q.request_count / 1500 * 100, 1),
    }
