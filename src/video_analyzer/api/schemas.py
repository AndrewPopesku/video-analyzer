from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    video_id: str
    title: str
    status: str
    transcript_count: int
    frame_count: int
    hook_count: int


class VideoResponse(BaseModel):
    video_id: str
    url: str | None
    title: str
    channel: str
    duration_seconds: int
    status: str
    created_at: str | None
    analyzed_at: str | None


class VideoListResponse(BaseModel):
    video_id: str
    title: str
    channel: str
    duration_seconds: int
    status: str
    created_at: str | None


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


class QuotaResponse(BaseModel):
    date: str
    request_count: int
    daily_limit: int
    remaining: int
    usage_percent: float
