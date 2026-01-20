"""Database operations and session management."""

from contextlib import contextmanager
from typing import Generator

from sqlmodel import SQLModel, Session, create_engine, select

from ..config import settings
from .models import QuotaUsage, Video

# Create engine lazily
_engine = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{settings.database_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engine


def init_db() -> None:
    """Initialize database tables."""
    SQLModel.metadata.create_all(get_engine())


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get database session context manager."""
    with Session(get_engine()) as session:
        yield session


def get_video_by_video_id(video_id: str) -> Video | None:
    """Get video by YouTube video ID."""
    with get_session() as session:
        statement = select(Video).where(Video.video_id == video_id)
        return session.exec(statement).first()


def get_all_videos() -> list[Video]:
    """Get all analyzed videos."""
    with get_session() as session:
        statement = select(Video).order_by(Video.created_at.desc())
        return list(session.exec(statement).all())


def get_today_quota() -> QuotaUsage:
    """Get or create today's quota usage record."""
    from datetime import date

    today = date.today().isoformat()
    with get_session() as session:
        statement = select(QuotaUsage).where(QuotaUsage.date == today)
        quota = session.exec(statement).first()
        if not quota:
            quota = QuotaUsage(date=today)
            session.add(quota)
            session.commit()
            session.refresh(quota)
        return quota


def increment_quota(tokens: int = 0) -> QuotaUsage:
    """Increment today's quota usage."""
    from datetime import datetime

    quota = get_today_quota()
    with get_session() as session:
        statement = select(QuotaUsage).where(QuotaUsage.id == quota.id)
        quota = session.exec(statement).first()
        quota.request_count += 1
        quota.token_count += tokens
        quota.last_request_at = datetime.utcnow()
        session.add(quota)
        session.commit()
        session.refresh(quota)
        return quota
