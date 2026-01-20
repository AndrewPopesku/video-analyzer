"""Semantic search with AI re-ranking."""

import json
from dataclasses import dataclass

from sqlmodel import select

from ..analyzer.provider import get_provider
from ..storage.database import get_session
from ..storage.models import Video, Transcript, Frame, Hook, Moment


@dataclass
class SearchResult:
    """A search result with relevance score."""

    video_id: str
    video_title: str
    timestamp: float
    content_type: str  # transcript, frame, hook, moment
    content: str
    relevance_score: float


RERANK_PROMPT = """Given the search query and candidate results, re-rank them by relevance.

QUERY: {query}

CANDIDATES:
{candidates}

For each candidate, assign a relevance score from 0.0 to 1.0 based on:
- Semantic relevance to the query (not just keyword matching)
- How well it answers what the user is looking for
- Context and intent behind the query

Return a JSON array of objects with "index" and "score":
[
  {{"index": 0, "score": 0.95}},
  {{"index": 1, "score": 0.72}},
  ...
]

Only include candidates with score >= 0.3. Order by score descending.
Return ONLY the JSON array."""


def search_videos(query: str, limit: int = 10) -> list[SearchResult]:
    """Search across all analyzed videos using Gemini re-ranking."""
    # Gather candidate results from database
    candidates = _gather_candidates(query, limit * 3)  # Get more candidates for re-ranking

    if not candidates:
        return []

    # Re-rank with Gemini
    reranked = rerank_results(query, candidates)

    return reranked[:limit]


def _gather_candidates(query: str, limit: int) -> list[SearchResult]:
    """Gather candidate results from database using basic text matching."""
    candidates = []
    query_lower = query.lower()
    keywords = query_lower.split()

    with get_session() as session:
        # Search transcripts
        transcripts = session.exec(select(Transcript)).all()
        for t in transcripts:
            video = session.get(Video, t.video_id)
            if not video:
                continue

            text_lower = t.text.lower()
            if any(kw in text_lower for kw in keywords):
                candidates.append(
                    SearchResult(
                        video_id=video.video_id,
                        video_title=video.title,
                        timestamp=t.start_time,
                        content_type="transcript",
                        content=t.text,
                        relevance_score=0.0,
                    )
                )

        # Search frame descriptions
        frames = session.exec(select(Frame)).all()
        for f in frames:
            video = session.get(Video, f.video_id)
            if not video or not f.description:
                continue

            desc_lower = f.description.lower()
            if any(kw in desc_lower for kw in keywords):
                candidates.append(
                    SearchResult(
                        video_id=video.video_id,
                        video_title=video.title,
                        timestamp=f.timestamp,
                        content_type="frame",
                        content=f.description,
                        relevance_score=0.0,
                    )
                )

        # Search hooks
        hooks = session.exec(select(Hook)).all()
        for h in hooks:
            video = session.get(Video, h.video_id)
            if not video:
                continue

            desc_lower = h.description.lower()
            snippet_lower = (h.transcript_snippet or "").lower()
            if any(kw in desc_lower or kw in snippet_lower for kw in keywords):
                candidates.append(
                    SearchResult(
                        video_id=video.video_id,
                        video_title=video.title,
                        timestamp=h.timestamp,
                        content_type="hook",
                        content=f"{h.hook_type}: {h.description}",
                        relevance_score=0.0,
                    )
                )

        # Search moments
        moments = session.exec(select(Moment)).all()
        for m in moments:
            video = session.get(Video, m.video_id)
            if not video:
                continue

            combined = f"{m.title} {m.description}".lower()
            if any(kw in combined for kw in keywords):
                candidates.append(
                    SearchResult(
                        video_id=video.video_id,
                        video_title=video.title,
                        timestamp=m.start_time,
                        content_type="moment",
                        content=f"{m.title}: {m.description}",
                        relevance_score=0.0,
                    )
                )

    # Deduplicate by video+timestamp (within 5 second window)
    seen = set()
    unique = []
    for c in candidates:
        key = (c.video_id, round(c.timestamp / 5))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique[:limit]


def rerank_results(query: str, candidates: list[SearchResult]) -> list[SearchResult]:
    """Re-rank search results using AI provider."""
    if not candidates:
        return []

    # Format candidates for prompt
    candidate_text = "\n".join(
        f"[{i}] ({c.content_type}) {c.video_title} @ {c.timestamp:.1f}s: {c.content[:200]}"
        for i, c in enumerate(candidates)
    )

    provider = get_provider()
    response = provider.generate(
        RERANK_PROMPT.format(query=query, candidates=candidate_text),
        system="You are a search relevance expert. Always return valid JSON.",
    )

    # Parse scores
    scores = _parse_rerank_response(response)

    # Apply scores and filter
    results = []
    for score_data in scores:
        idx = score_data.get("index", -1)
        score = score_data.get("score", 0)

        if 0 <= idx < len(candidates) and score >= 0.3:
            candidate = candidates[idx]
            candidate.relevance_score = score
            results.append(candidate)

    # Sort by relevance
    results.sort(key=lambda x: x.relevance_score, reverse=True)

    return results


def _parse_rerank_response(response: str) -> list[dict]:
    """Parse re-ranking response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    return []
