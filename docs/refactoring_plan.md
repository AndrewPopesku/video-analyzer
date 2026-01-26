# Refactoring Plan: Service Layer Pattern (Simpler)

## Goal
Refactor the codebase to separate business logic from the CLI and API, ensuring **DRY** (logic shared between CLI/API) and **KISS** (simple structure), while adhering to **SOLID** principles.

## User Review Required
> [!NOTE]
> We will avoid the complex `domain/application/infrastructure` structure. Instead, we will introduce a **Service Layer** to centralize business logic.

## Proposed Architecture

1.  **Service Layer (`VideoAnalyzerService`)**: The core "brain" of the application. It orchestrates the download, processing, analysis, and storage steps.
2.  **Interfaces**: Define contracts for external dependencies (Downloader, Processor) to allow easy swapping/testing.
3.  **Presentation (CLI & API)**: Thin wrappers that call the Service Layer.

## Proposed Changes

### 1. Interfaces & Core Abstractions
#### [NEW] `src/video_analyzer/interfaces.py`
```python
from typing import Protocol
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VideoInfo:
    video_id: str
    url: str
    title: str
    channel: str
    duration_seconds: int
    thumbnail_url: str | None = None
    description: str | None = None

class MediaDownloader(Protocol):
    def get_info(self, url: str) -> VideoInfo: ...
    def download(self, url: str) -> tuple[VideoInfo, Path]: ...

class MediaProcessor(Protocol):
    def extract_audio(self, video_id: str, video_path: Path) -> Path: ...
    def extract_frames(self, video_id: str, video_path: Path) -> list[Path]: ...
```

> **Note**: The existing `AIProvider` abstract base class in `analyzer/provider.py` is already well-designed and requires no changes. The service will use it via `get_provider()`.

### 2. Service Layer
#### [NEW] `src/video_analyzer/service.py`
```python
from typing import Callable
from .interfaces import MediaDownloader, MediaProcessor
from .analyzer.provider import AIProvider
from .storage.models import Video, SearchResult

class VideoAnalyzerService:
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
    ) -> Video:
        """Full analysis pipeline. Returns the completed Video record."""
        ...

    def get_video(self, video_id: str) -> Video | None:
        """Retrieve a video by its YouTube ID."""
        ...

    def list_videos(self) -> list[Video]:
        """List all analyzed videos."""
        ...

    def search_videos(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search across all analyzed videos."""
        ...
```

**Key design decisions**:
*   **Progress callback**: The `on_progress` parameter allows CLI to show progress via `rich.Progress` while API can ignore it or use SSE.
*   **Transaction boundaries**: The service will consolidate the current ~7 separate `get_session()` blocks into fewer, logical transactions (e.g., one for video creation, one for all analysis results).
*   **Dependencies injected**: Enables testing with mocks.

### 3. Refactoring Existing Modules

#### [MODIFY] `ingestion/downloader.py`
Add a class that delegates to existing functions (no rewrite needed):
```python
class YouTubeDownloader:
    """Implements MediaDownloader protocol."""

    def get_info(self, url: str) -> VideoInfo:
        return get_video_info(url)

    def download(self, url: str) -> tuple[VideoInfo, Path]:
        return download_video(url)
```

#### [MODIFY] `ingestion/processor.py`
Add a class that delegates to existing functions:
```python
class FFmpegProcessor:
    """Implements MediaProcessor protocol."""

    def extract_audio(self, video_id: str, video_path: Path) -> Path:
        return extract_audio(video_id, video_path)

    def extract_frames(self, video_id: str, video_path: Path) -> list[Path]:
        return extract_frames(video_id, video_path)
```

#### [NO CHANGE] `analyzer/provider.py`
The existing `AIProvider` ABC and `get_provider()` singleton are already suitable.

### 4. Entry Points

#### [MODIFY] `cli.py`
Massive reduction in size. The `analyze` command becomes:
```python
@app.command()
def analyze(url: str, provider: str | None = None):
    set_provider(provider)
    if not validate_api_key(provider):
        raise typer.Exit(1)

    init_db()

    service = VideoAnalyzerService(
        downloader=YouTubeDownloader(),
        processor=FFmpegProcessor(),
        provider=get_provider(),
    )

    with Progress(...) as progress:
        task = progress.add_task("Analyzing...", total=None)

        def on_progress(msg: str):
            progress.update(task, description=msg)

        video = service.analyze_video(url, on_progress=on_progress)

    console.print(f"[green]Analysis complete![/green]")
    console.print(f"View hooks: video-analyzer hooks {video.video_id}")
```

#### [MODIFY] `api/routes.py`
*   Use `VideoAnalyzerService` for `list_videos`, `get_video`, and `search`.
*   **Add new endpoint**: `POST /analyze` to trigger analysis via API (demonstrates DRY benefit).

```python
@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest):
    """Analyze a YouTube video."""
    service = VideoAnalyzerService(
        downloader=YouTubeDownloader(),
        processor=FFmpegProcessor(),
        provider=get_provider(),
    )
    video = service.analyze_video(request.url)
    return {"video_id": video.video_id, "status": video.status}
```

## Implementation Steps

| Step | File | Action |
|------|------|--------|
| 1 | `interfaces.py` | Create protocols for `MediaDownloader`, `MediaProcessor` |
| 2 | `downloader.py` | Add `YouTubeDownloader` class (delegates to existing functions) |
| 3 | `processor.py` | Add `FFmpegProcessor` class (delegates to existing functions) |
| 4 | `service.py` | Create `VideoAnalyzerService`, extract logic from `cli.py:64-368` |
| 5 | `cli.py` | Replace orchestration with service calls |
| 6 | `api/routes.py` | Refactor existing endpoints + add `POST /analyze` |
| 7 | Verify | Run manual tests |

## Verification Plan

1.  **Manual Test (CLI)**:
    *   `python -m video_analyzer.cli analyze <URL>`
    *   Verify DB is populated and status is `completed`.
2.  **Manual Test (API)**:
    *   `python -m video_analyzer.cli serve`
    *   `POST /analyze` with a URL and verify it works.
    *   `GET /videos` and verify data matches.
3.  **Regression Check**:
    *   Ensure `hooks`, `summary`, `search`, `list` commands still work.
