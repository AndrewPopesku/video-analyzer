"""SQLModel data models for video analyzer."""

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship


class Video(SQLModel, table=True):
    """Represents an analyzed video."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: str = Field(unique=True, index=True)  # YouTube video ID
    url: str
    title: str
    channel: str
    duration_seconds: int
    thumbnail_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    analyzed_at: Optional[datetime] = None
    status: str = Field(default="pending")  # pending, processing, completed, failed

    # Relationships
    transcripts: list["Transcript"] = Relationship(back_populates="video")
    frames: list["Frame"] = Relationship(back_populates="video")
    hooks: list["Hook"] = Relationship(back_populates="video")
    moments: list["Moment"] = Relationship(back_populates="video")
    scenes: list["Scene"] = Relationship(back_populates="video")
    shots: list["Shot"] = Relationship(back_populates="video")
    entities: list["Entity"] = Relationship(back_populates="video")
    topics: list["Topic"] = Relationship(back_populates="video")
    keywords: list["Keyword"] = Relationship(back_populates="video")
    ocr_results: list["OCRResult"] = Relationship(back_populates="video")


class Transcript(SQLModel, table=True):
    """Timestamped transcript segment."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    start_time: float  # seconds
    end_time: float
    text: str
    confidence: Optional[float] = None

    video: Optional[Video] = Relationship(back_populates="transcripts")


class Frame(SQLModel, table=True):
    """Analyzed video frame."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    timestamp: float  # seconds
    frame_path: str  # relative path to frame image
    description: Optional[str] = None  # Gemini vision description
    objects: Optional[str] = None  # JSON list of detected objects
    scene_type: Optional[str] = None  # e.g., "talking_head", "b_roll", "text_overlay"

    # Deduplication fields
    perceptual_hash: Optional[str] = None  # hex string of pHash
    is_duplicate: bool = Field(default=False)
    reference_frame_id: Optional[int] = Field(default=None, foreign_key="frame.id")

    video: Optional[Video] = Relationship(back_populates="frames")


class Hook(SQLModel, table=True):
    """Detected attention hook in video."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    timestamp: float  # seconds
    end_timestamp: Optional[float] = None
    hook_type: str  # visual, audio, content, pattern_interrupt
    description: str
    confidence: float = Field(default=0.5)
    transcript_snippet: Optional[str] = None
    frame_id: Optional[int] = Field(default=None, foreign_key="frame.id")

    video: Optional[Video] = Relationship(back_populates="hooks")


class Moment(SQLModel, table=True):
    """Key moment or searchable segment."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    start_time: float
    end_time: float
    title: str
    description: str
    keywords: Optional[str] = None  # JSON list of keywords

    video: Optional[Video] = Relationship(back_populates="moments")


class Scene(SQLModel, table=True):
    """Video scene (collection of shots)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    start_time: float
    end_time: float
    label: Optional[str] = None
    description: Optional[str] = None  # Detailed description of scene content

    video: Optional[Video] = Relationship(back_populates="scenes")


class Shot(SQLModel, table=True):
    """Video shot (camera cut)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    scene_id: Optional[int] = Field(default=None, foreign_key="scene.id")
    start_time: float
    end_time: float
    keyframe_path: Optional[str] = None

    video: Optional[Video] = Relationship(back_populates="shots")


class Entity(SQLModel, table=True):
    """Named entity (Brand, Location, Person)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    name: str
    type: str  # Brand, Location, Person
    count: int = Field(default=1)
    description: Optional[str] = None

    video: Optional[Video] = Relationship(back_populates="entities")


class Topic(SQLModel, table=True):
    """High-level video topic."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    name: str
    confidence: float = Field(default=1.0)

    video: Optional[Video] = Relationship(back_populates="topics")


class Keyword(SQLModel, table=True):
    """Video keyword."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    text: str
    confidence: float = Field(default=1.0)

    video: Optional[Video] = Relationship(back_populates="keywords")


class OCRResult(SQLModel, table=True):
    """Text detected in a frame."""

    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    frame_id: int = Field(foreign_key="frame.id", index=True)
    text: str
    left: float
    top: float
    width: float
    height: float

    video: Optional[Video] = Relationship(back_populates="ocr_results")


class QuotaUsage(SQLModel, table=True):
    """Track API quota usage."""

    id: Optional[int] = Field(default=None, primary_key=True)
    date: str = Field(index=True)  # YYYY-MM-DD
    request_count: int = Field(default=0)
    token_count: int = Field(default=0)
    last_request_at: Optional[datetime] = None
