"""CLI interface for video analyzer."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlmodel import select
from tenacity import RetryError

from .config import settings
from .storage.database import init_db, get_session, get_today_quota
from .storage.models import Video, Hook, Scene, Entity, Topic, Keyword, OCRResult
from .storage.media import get_storage_stats
from .ingestion.downloader import YouTubeDownloader
from .ingestion.processor import FFmpegProcessor
from .analyzer.provider import get_provider, reset_provider
from .analyzer.gemini import QuotaExceeded, GeminiAPIError
from .analyzer.groq import GroqAPIError
from .service import VideoAnalyzerService, VideoAlreadyAnalyzedError

app = typer.Typer(help="YouTube Video Analyzer - Powered by Gemini/Groq AI")
console = Console()


def validate_api_key(provider: str | None = None) -> bool:
    """Validate that the required API key is configured."""
    provider = provider or settings.ai_provider

    if provider == "groq":
        if not settings.groq_api_key:
            console.print(
                "[red]Error: GROQ_API_KEY environment variable not set.[/red]"
            )
            console.print("\nTo fix this, run:")
            console.print("  export GROQ_API_KEY='your-api-key'")
            console.print("\nGet your API key at: https://console.groq.com/keys")
            return False
    else:
        if not settings.gemini_api_key:
            console.print(
                "[red]Error: GEMINI_API_KEY environment variable not set.[/red]"
            )
            console.print("\nTo fix this, run:")
            console.print("  export GEMINI_API_KEY='your-api-key'")
            console.print("\nGet your API key at: https://aistudio.google.com/apikey")
            return False
    return True


def set_provider(provider: str | None) -> None:
    """Set the AI provider if specified."""
    if provider:
        settings.ai_provider = provider
        reset_provider()


def _create_service() -> VideoAnalyzerService:
    """Create a VideoAnalyzerService with default dependencies."""
    return VideoAnalyzerService(
        downloader=YouTubeDownloader(),
        processor=FFmpegProcessor(),
        provider=get_provider(),
    )


@app.command()
def analyze(
    url: str = typer.Argument(..., help="YouTube video URL"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="AI provider: gemini or groq"
    ),
    skip_download: bool = typer.Option(
        False, "--skip-download", help="Skip download if video exists"
    ),
):
    """Analyze a YouTube video for hooks and key moments."""
    # Set provider if specified
    set_provider(provider)

    # Validate API key before doing any work
    if not validate_api_key(provider):
        raise typer.Exit(1)

    init_db()
    console.print(f"[dim]Using provider: {settings.ai_provider}[/dim]")

    service = _create_service()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting analysis...", total=None)

        def on_progress(msg: str) -> None:
            progress.update(task, description=msg)

        try:
            result = service.analyze_video(url, on_progress=on_progress)

            # Show video info
            console.print(f"\n[bold]{result.video.title}[/bold]")
            console.print(f"Channel: {result.video.channel}")
            console.print(
                f"Duration: {result.video.duration_seconds // 60}:{result.video.duration_seconds % 60:02d}"
            )

            # Show stats
            console.print(f"Transcribed {result.transcript_count} segments")

            if result.dedup_stats:
                stats = result.dedup_stats
                total = stats.get("total_frames", 0)
                unique = stats.get("unique_frames", 0)
                dups = stats.get("duplicates", 0)
                ratio = stats.get("dedup_ratio", 0)
                api_saved = dups // settings.max_frames_per_batch
                console.print(
                    f"Frames: {total} total, {unique} unique, {dups} duplicates "
                    f"([green]{ratio:.0%} savings[/green], ~{api_saved} API calls saved)"
                )
            else:
                console.print(f"Analyzed {result.frame_count} frames")

            console.print(f"Detected {result.hook_count} hooks")

            progress.update(task, description="Analysis complete!")

        except VideoAlreadyAnalyzedError as e:
            console.print(
                f"\n[yellow]Video already analyzed. Use 'hooks {e.video_id}' to view results.[/yellow]"
            )
            raise typer.Exit(0)
        except QuotaExceeded as e:
            console.print(f"\n[red]Quota exceeded: {e}[/red]")
            raise typer.Exit(1)
        except (GeminiAPIError, GroqAPIError) as e:
            console.print(f"\n[red]API error: {e}[/red]")
            raise typer.Exit(1)
        except RetryError as e:
            cause = e.last_attempt.exception() if e.last_attempt else e
            console.print(f"\n[red]Error during analysis (RetryError): {cause}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(
                f"\n[red]Error during analysis ({type(e).__name__}): {e}[/red]"
            )
            raise typer.Exit(1)

    console.print("\n[green]Analysis complete![/green]")
    console.print(f"View hooks: video-analyzer hooks {result.video.video_id}")


@app.command()
def hooks(
    video_id: str = typer.Argument(..., help="YouTube video ID"),
):
    """View detected hooks for a video."""
    init_db()

    with get_session() as session:
        video = session.exec(select(Video).where(Video.video_id == video_id)).first()

        if not video:
            console.print(f"[red]Video not found: {video_id}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]{video.title}[/bold]")
        console.print(f"Channel: {video.channel}\n")

        hooks_list = session.exec(
            select(Hook).where(Hook.video_id == video.id).order_by(Hook.timestamp)
        ).all()

        if not hooks_list:
            console.print("[yellow]No hooks detected for this video.[/yellow]")
            raise typer.Exit(0)

        table = Table(title="Detected Hooks")
        table.add_column("Time", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Confidence", style="yellow")

        for h in hooks_list:
            time_str = f"{int(h.timestamp // 60)}:{int(h.timestamp % 60):02d}"
            conf_str = f"{h.confidence:.0%}"
            table.add_row(time_str, h.hook_type, h.description[:60], conf_str)

        console.print(table)


@app.command()
def summary(
    video_id: str = typer.Argument(..., help="YouTube video ID"),
):
    """View a comprehensive summary of video insights."""
    init_db()

    with get_session() as session:
        video = session.exec(select(Video).where(Video.video_id == video_id)).first()

        if not video:
            console.print(f"[red]Video not found: {video_id}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold underline]{video.title}[/bold underline]")
        console.print(
            f"Channel: {video.channel} | Duration: {video.duration_seconds // 60}:{video.duration_seconds % 60:02d}"
        )
        console.print(f"Status: {video.status}\n")

        # 1. Topics and Keywords
        topics = session.exec(select(Topic).where(Topic.video_id == video.id)).all()
        keywords = session.exec(
            select(Keyword).where(Keyword.video_id == video.id)
        ).all()

        if topics:
            console.print(
                "[bold cyan]Topics:[/bold cyan] " + ", ".join(t.name for t in topics)
            )
        if keywords:
            console.print(
                "[bold cyan]Keywords:[/bold cyan] "
                + ", ".join(k.text for k in keywords)
            )
        console.print("")

        # 2. Entities
        entities = session.exec(select(Entity).where(Entity.video_id == video.id)).all()
        if entities:
            table = Table(
                title="Named Entities", show_header=True, header_style="bold magenta"
            )
            table.add_column("Name")
            table.add_column("Type")
            table.add_column("Mentions")
            table.add_column("Description")
            for e in entities:
                table.add_row(e.name, e.type, str(e.count), e.description or "")
            console.print(table)
            console.print("")

        # 3. Scenes
        scenes = session.exec(
            select(Scene).where(Scene.video_id == video.id).order_by(Scene.start_time)
        ).all()
        if scenes:
            table = Table(
                title="Scenes & Shots", show_header=True, header_style="bold green"
            )
            table.add_column("Time", width=15)
            table.add_column("Label", width=20)
            table.add_column("Description", width=50)
            for s in scenes:
                time_range = f"{int(s.start_time // 60)}:{int(s.start_time % 60):02d} - {int(s.end_time // 60)}:{int(s.end_time % 60):02d}"
                description = (s.description or "")[:50]
                if s.description and len(s.description) > 50:
                    description += "..."
                table.add_row(time_range, s.label or "Scene", description)
            console.print(table)
            console.print("")

        # 4. OCR Highlights (max 5)
        ocr = session.exec(
            select(OCRResult).where(OCRResult.video_id == video.id).limit(10)
        ).all()
        if ocr:
            console.print("[bold yellow]OCR Highlights:[/bold yellow]")
            unique_text = sorted(list(set(o.text for o in ocr)))[:10]
            for text in unique_text:
                console.print(f"  - {text}")
            console.print("")

        # 5. Hooks Summary
        hooks_count = len(
            session.exec(select(Hook).where(Hook.video_id == video.id)).all()
        )
        console.print(f"[bold blue]Hooks Detected:[/bold blue] {hooks_count}")
        console.print(f"Run 'video-analyzer hooks {video.video_id}' for full details.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="AI provider: gemini or groq"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Search across analyzed videos."""
    # Set provider if specified
    set_provider(provider)

    # Validate API key before doing any work (needed for re-ranking)
    if not validate_api_key(provider):
        raise typer.Exit(1)

    init_db()

    service = _create_service()

    with console.status("Searching..."):
        results = service.search_videos(query, limit)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Search Results: '{query}'")
    table.add_column("Video", style="cyan")
    table.add_column("Time", style="magenta")
    table.add_column("Type", style="blue")
    table.add_column("Content", style="green")
    table.add_column("Score", style="yellow")

    for r in results:
        time_str = f"{int(r.timestamp // 60)}:{int(r.timestamp % 60):02d}"
        table.add_row(
            r.video_title[:30],
            time_str,
            r.content_type,
            r.content[:50],
            f"{r.relevance_score:.0%}",
        )

    console.print(table)


@app.command("list")
def list_videos():
    """List all analyzed videos."""
    init_db()

    service = _create_service()
    videos = service.list_videos()

    if not videos:
        console.print("[yellow]No videos analyzed yet.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Analyzed Videos")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Channel", style="magenta")
    table.add_column("Duration", style="blue")
    table.add_column("Status", style="yellow")

    for v in videos:
        duration = f"{v.duration_seconds // 60}:{v.duration_seconds % 60:02d}"
        status_style = (
            "green"
            if v.status == "completed"
            else "yellow" if v.status == "processing" else "red"
        )
        table.add_row(
            v.video_id,
            v.title[:40],
            v.channel[:20],
            duration,
            f"[{status_style}]{v.status}[/{status_style}]",
        )

    console.print(table)


@app.command()
def quota():
    """Show API quota usage."""
    init_db()

    q = get_today_quota()

    # Show quota based on current provider
    if settings.ai_provider == "groq":
        daily_limit = settings.groq_daily_limit
        provider_name = "Groq"
    else:
        daily_limit = settings.gemini_daily_limit
        provider_name = "Gemini"

    remaining = daily_limit - q.request_count
    usage_pct = q.request_count / daily_limit * 100

    console.print(f"\n[bold]{provider_name} API Quota Usage[/bold]")
    console.print(f"Current provider: {settings.ai_provider}")
    console.print(f"Date: {q.date}")
    console.print(f"Requests: {q.request_count} / {daily_limit}")
    console.print(f"Remaining: {remaining}")

    if usage_pct >= 80:
        console.print(f"[red]Usage: {usage_pct:.1f}% - Running low![/red]")
    elif usage_pct >= 50:
        console.print(f"[yellow]Usage: {usage_pct:.1f}%[/yellow]")
    else:
        console.print(f"[green]Usage: {usage_pct:.1f}%[/green]")


@app.command()
def storage():
    """Show storage usage statistics."""
    stats = get_storage_stats()

    console.print("\n[bold]Storage Usage[/bold]")
    console.print(f"Total: {stats['total_size_mb']} MB")
    console.print(f"Videos: {stats['video_count']}")

    if stats["videos"]:
        table = Table(title="Videos by Size")
        table.add_column("Video ID", style="cyan")
        table.add_column("Size (MB)", style="green")

        for v in sorted(stats["videos"], key=lambda x: x["size_bytes"], reverse=True):
            size_mb = round(v["size_bytes"] / (1024 * 1024), 2)
            table.add_row(v["video_id"], str(size_mb))

        console.print(table)


@app.command()
def config():
    """Show current configuration."""
    console.print("\n[bold]Current Configuration[/bold]")
    console.print(f"AI Provider: {settings.ai_provider}")
    console.print(
        f"Gemini API Key: {'***' + settings.gemini_api_key[-4:] if settings.gemini_api_key else 'Not set'}"
    )
    console.print(
        f"Groq API Key: {'***' + settings.groq_api_key[-4:] if settings.groq_api_key else 'Not set'}"
    )
    console.print(f"Data Directory: {settings.data_directory}")
    console.print(f"Frame Rate: {settings.frame_rate} fps")

    console.print("\n[bold]Frame Deduplication[/bold]")
    console.print(f"Enabled: {settings.enable_deduplication}")
    console.print(f"Method: {settings.dedup_method}")
    if settings.dedup_method == "embedding":
        console.print(f"Threshold: {settings.dedup_threshold} (cosine similarity, 0-1)")
    else:
        console.print(f"Algorithm: {settings.dedup_algorithm}")
        console.print(f"Hash size: {settings.dedup_hash_size}")
        console.print(f"Threshold: {settings.dedup_threshold} (Hamming distance)")
        console.print(f"Sequential only: {settings.dedup_sequential_only}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
):
    """Start the API server."""
    import uvicorn

    init_db()
    console.print(f"Starting server at http://{host}:{port}")
    uvicorn.run("video_analyzer.api.routes:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
