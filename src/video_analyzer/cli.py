"""CLI interface for video analyzer."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlmodel import select
from tenacity import RetryError

from .config import settings
from .storage.database import init_db, get_session, get_today_quota, get_all_videos, increment_quota
from .storage.models import (
    Video, Transcript, Frame, Hook, Moment, Scene, Shot,
    Entity, Topic, Keyword, OCRResult
)
from .storage.media import get_storage_stats
from .ingestion.downloader import download_video, get_video_info
from .ingestion.processor import extract_audio, extract_frames
from .analyzer.provider import get_provider, reset_provider
from .analyzer.gemini import QuotaExceeded, GeminiAPIError
from .analyzer.groq import GroqAPIError
from .analyzer.transcribe import transcribe_audio
from .analyzer.vision import analyze_frames
from .analyzer.hooks import detect_hooks, analyze_intro
from .search.reranker import search_videos

app = typer.Typer(help="YouTube Video Analyzer - Powered by Gemini/Groq AI")
console = Console()


def validate_api_key(provider: str | None = None) -> bool:
    """Validate that the required API key is configured."""
    provider = provider or settings.ai_provider

    if provider == "groq":
        if not settings.groq_api_key:
            console.print("[red]Error: GROQ_API_KEY environment variable not set.[/red]")
            console.print("\nTo fix this, run:")
            console.print("  export GROQ_API_KEY='your-api-key'")
            console.print("\nGet your API key at: https://console.groq.com/keys")
            return False
    else:
        if not settings.gemini_api_key:
            console.print("[red]Error: GEMINI_API_KEY environment variable not set.[/red]")
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


@app.command()
def analyze(
    url: str = typer.Argument(..., help="YouTube video URL"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="AI provider: gemini or groq"),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip download if video exists"),
):
    """Analyze a YouTube video for hooks and key moments."""
    # Set provider if specified
    set_provider(provider)

    # Validate API key before doing any work
    if not validate_api_key(provider):
        raise typer.Exit(1)

    init_db()
    console.print(f"[dim]Using provider: {settings.ai_provider}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Get video info
        task = progress.add_task("Getting video info...", total=None)
        try:
            info = get_video_info(url)
        except Exception as e:
            console.print(f"[red]Error getting video info: {e}[/red]")
            raise typer.Exit(1)

        progress.update(task, description=f"Video: {info.title}")
        console.print(f"\n[bold]{info.title}[/bold]")
        console.print(f"Channel: {info.channel}")
        console.print(f"Duration: {info.duration_seconds // 60}:{info.duration_seconds % 60:02d}")

        # Check if already analyzed
        with get_session() as session:
            existing = session.exec(
                select(Video).where(Video.video_id == info.video_id)
            ).first()
            if existing and existing.status == "completed":
                console.print(f"\n[yellow]Video already analyzed. Use 'hooks {info.video_id}' to view results.[/yellow]")
                raise typer.Exit(0)

        # Create/update video record
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
            video_db_id = video.id

        try:
            # Initialize provider
            provider = get_provider()

            # Download video
            progress.update(task, description="Downloading video...")
            _, video_path = download_video(url)

            # Extract audio
            progress.update(task, description="Extracting audio...")
            audio_path = extract_audio(info.video_id, video_path)

            # Extract frames
            progress.update(task, description="Extracting frames...")
            frame_paths = extract_frames(info.video_id, video_path)
            console.print(f"Extracted {len(frame_paths)} frames")

            # Transcribe audio
            progress.update(task, description="Transcribing audio...")
            transcript = transcribe_audio(info.video_id)
            console.print(f"Transcribed {len(transcript)} segments")

            # Save transcript to database
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

            # Analyze frames with deduplication
            progress.update(task, description="Analyzing frames...")
            frame_analysis, dedup_result = analyze_frames(info.video_id)

            # Show deduplication stats
            if dedup_result and dedup_result.stats:
                stats = dedup_result.stats
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
                console.print(f"Analyzed {len(frame_analysis)} frames")

            # Save frame analysis to database
            with get_session() as session:
                # First pass: save all frames and build path->id map
                frame_id_map: dict[str, int] = {}
                for fa in frame_analysis:
                    frame_path = fa.get("frame_path", "")
                    # Get hash from dedup_result if available
                    phash = None
                    if dedup_result and dedup_result.hashes:
                        from pathlib import Path
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

            # Detect hooks
            progress.update(task, description="Detecting hooks...")
            hooks = detect_hooks(transcript, frame_analysis)
            console.print(f"Detected {len(hooks)} hooks")

            # Save hooks to database
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

            # Extract Topics and Keywords
            progress.update(task, description="Extracting topics and keywords...")
            topics_data = provider.extract_topics_and_keywords(transcript)
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

            # Extract Entities
            progress.update(task, description="Extracting entities...")
            keyframe_paths = [Path(f["frame_path"]) for f in frame_analysis if f.get("frame_path")][:10]
            entities = provider.extract_entities(transcript, keyframe_paths)
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

            # Perform OCR and Scene Detection
            progress.update(task, description="Performing OCR and scene detection...")
            with get_session() as session:
                # Map timestamps to frame IDs
                stmt = select(Frame).where(Frame.video_id == video_db_id)
                db_frames = session.exec(stmt).all()
                frame_map = {f.timestamp: f.id for f in db_frames}

                for fa in frame_analysis:
                    if fa.get("text_visible") or fa.get("frame_index", 0) % 10 == 0:
                        image_path = Path(fa["frame_path"])
                        ocr_data = provider.perform_ocr(image_path)
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

            # Scene Detection - use all frames with proper timestamps
            progress.update(task, description="Detecting scenes...")
            all_frame_data = [
                (Path(f["frame_path"]), f.get("timestamp", 0.0))
                for f in frame_analysis
                if f.get("frame_path")
            ]
            scene_frame_paths = [f[0] for f in all_frame_data]
            scene_timestamps = [f[1] for f in all_frame_data]

            scenes_data = provider.detect_scenes_and_shots(scene_frame_paths, scene_timestamps)
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

            # Mark as completed
            with get_session() as session:
                video = session.get(Video, video_db_id)
                video.status = "completed"
                video.analyzed_at = datetime.utcnow()
                session.add(video)
                session.commit()

            progress.update(task, description="Analysis complete!")

        except QuotaExceeded as e:
            console.print(f"\n[red]Quota exceeded: {e}[/red]")
            with get_session() as session:
                video = session.get(Video, video_db_id)
                video.status = "failed"
                session.add(video)
                session.commit()
            raise typer.Exit(1)
        except (GeminiAPIError, GroqAPIError) as e:
            console.print(f"\n[red]API error: {e}[/red]")
            with get_session() as session:
                video = session.get(Video, video_db_id)
                video.status = "failed"
                session.add(video)
                session.commit()
            raise typer.Exit(1)
        except RetryError as e:
            # Extract the actual exception from RetryError
            cause = e.last_attempt.exception() if e.last_attempt else e
            console.print(f"\n[red]Error during analysis (RetryError): {cause}[/red]")
            with get_session() as session:
                video = session.get(Video, video_db_id)
                video.status = "failed"
                session.add(video)
                session.commit()
            raise typer.Exit(1)
        except Exception as e:
            # Show the actual error type for debugging
            console.print(f"\n[red]Error during analysis ({type(e).__name__}): {e}[/red]")
            with get_session() as session:
                video = session.get(Video, video_db_id)
                video.status = "failed"
                session.add(video)
                session.commit()
            raise typer.Exit(1)

    console.print(f"\n[green]Analysis complete![/green]")
    console.print(f"View hooks: video-analyzer hooks {info.video_id}")


@app.command()
def hooks(
    video_id: str = typer.Argument(..., help="YouTube video ID"),
):
    """View detected hooks for a video."""
    init_db()

    with get_session() as session:
        video = session.exec(
            select(Video).where(Video.video_id == video_id)
        ).first()

        if not video:
            console.print(f"[red]Video not found: {video_id}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]{video.title}[/bold]")
        console.print(f"Channel: {video.channel}\n")

        hooks = session.exec(
            select(Hook).where(Hook.video_id == video.id).order_by(Hook.timestamp)
        ).all()

        if not hooks:
            console.print("[yellow]No hooks detected for this video.[/yellow]")
            raise typer.Exit(0)

        table = Table(title="Detected Hooks")
        table.add_column("Time", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Confidence", style="yellow")

        for h in hooks:
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
        video = session.exec(
            select(Video).where(Video.video_id == video_id)
        ).first()

        if not video:
            console.print(f"[red]Video not found: {video_id}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold underline]{video.title}[/bold underline]")
        console.print(f"Channel: {video.channel} | Duration: {video.duration_seconds // 60}:{video.duration_seconds % 60:02d}")
        console.print(f"Status: {video.status}\n")

        # 1. Topics and Keywords
        topics = session.exec(select(Topic).where(Topic.video_id == video.id)).all()
        keywords = session.exec(select(Keyword).where(Keyword.video_id == video.id)).all()

        if topics:
            console.print("[bold cyan]Topics:[/bold cyan] " + ", ".join(t.name for t in topics))
        if keywords:
            console.print("[bold cyan]Keywords:[/bold cyan] " + ", ".join(k.text for k in keywords))
        console.print("")

        # 2. Entities
        entities = session.exec(select(Entity).where(Entity.video_id == video.id)).all()
        if entities:
            table = Table(title="Named Entities", show_header=True, header_style="bold magenta")
            table.add_column("Name")
            table.add_column("Type")
            table.add_column("Mentions")
            table.add_column("Description")
            for e in entities:
                table.add_row(e.name, e.type, str(e.count), e.description or "")
            console.print(table)
            console.print("")

        # 3. Scenes
        scenes = session.exec(select(Scene).where(Scene.video_id == video.id).order_by(Scene.start_time)).all()
        if scenes:
            table = Table(title="Scenes & Shots", show_header=True, header_style="bold green")
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
        ocr = session.exec(select(OCRResult).where(OCRResult.video_id == video.id).limit(10)).all()
        if ocr:
            console.print("[bold yellow]OCR Highlights:[/bold yellow]")
            unique_text = sorted(list(set(o.text for o in ocr)))[:10]
            for text in unique_text:
                console.print(f"  • {text}")
            console.print("")

        # 5. Hooks Summary
        hooks_count = len(session.exec(select(Hook).where(Hook.video_id == video.id)).all())
        console.print(f"[bold blue]Hooks Detected:[/bold blue] {hooks_count}")
        console.print(f"Run 'video-analyzer hooks {video.video_id}' for full details.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="AI provider: gemini or groq"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Search across analyzed videos."""
    # Set provider if specified
    set_provider(provider)

    # Validate API key before doing any work (needed for re-ranking)
    if not validate_api_key(provider):
        raise typer.Exit(1)

    init_db()

    with console.status("Searching..."):
        results = search_videos(query, limit)

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

    videos = get_all_videos()

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
        status_style = "green" if v.status == "completed" else "yellow" if v.status == "processing" else "red"
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

    console.print(f"\n[bold]Storage Usage[/bold]")
    console.print(f"Total: {stats['total_size_mb']} MB")
    console.print(f"Videos: {stats['video_count']}")

    if stats['videos']:
        table = Table(title="Videos by Size")
        table.add_column("Video ID", style="cyan")
        table.add_column("Size (MB)", style="green")

        for v in sorted(stats['videos'], key=lambda x: x['size_bytes'], reverse=True):
            size_mb = round(v['size_bytes'] / (1024 * 1024), 2)
            table.add_row(v['video_id'], str(size_mb))

        console.print(table)


@app.command()
def config():
    """Show current configuration."""
    console.print(f"\n[bold]Current Configuration[/bold]")
    console.print(f"AI Provider: {settings.ai_provider}")
    console.print(f"Gemini API Key: {'***' + settings.gemini_api_key[-4:] if settings.gemini_api_key else 'Not set'}")
    console.print(f"Groq API Key: {'***' + settings.groq_api_key[-4:] if settings.groq_api_key else 'Not set'}")
    console.print(f"Data Directory: {settings.data_directory}")
    console.print(f"Frame Rate: {settings.frame_rate} fps")

    console.print(f"\n[bold]Frame Deduplication[/bold]")
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
