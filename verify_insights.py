import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime
from sqlmodel import Session, select
from video_analyzer.storage.database import get_engine, init_db, get_session
from video_analyzer.storage.models import Video, Transcript, Frame, Hook, Topic, Keyword, Entity, OCRResult, Scene
from video_analyzer.cli import app
from typer.testing import CliRunner

runner = CliRunner()

class TestAdvancedInsights(unittest.TestCase):
    def setUp(self):
        # Use in-memory SQLite for testing
        self.engine = get_engine()
        init_db()
        
    def test_database_models(self):
        """Test that new models can be saved and retrieved."""
        with get_session() as session:
            # Create a test video
            v = Video(
                video_id="test_vid_123",
                url="https://youtube.com/watch?v=test_vid_123",
                title="Test Video",
                channel="Test Channel",
                duration_seconds=120,
                status="completed"
            )
            session.add(v)
            session.commit()
            session.refresh(v)
            
            # Add Topic
            topic = Topic(video_id=v.id, name="Test Topic", confidence=0.9)
            session.add(topic)
            
            # Add Keyword
            kw = Keyword(video_id=v.id, text="testkeyword", confidence=0.8)
            session.add(kw)
            
            # Add Entity
            entity = Entity(video_id=v.id, name="Test Brand", type="Brand", count=2, description="A test brand")
            session.add(entity)
            
            # Add Scene
            scene = Scene(video_id=v.id, start_time=0.0, end_time=10.0, label="Intro")
            session.add(scene)
            
            # Add Frame and OCR
            frame = Frame(video_id=v.id, timestamp=1.0, frame_path="path/to/frame.jpg")
            session.add(frame)
            session.commit()
            session.refresh(frame)
            
            ocr = OCRResult(video_id=v.id, frame_id=frame.id, text="Hello World", left=0.1, top=0.1, width=0.2, height=0.05)
            session.add(ocr)
            
            session.commit()
            
            # Verify retrieval
            video = session.exec(select(Video).where(Video.video_id == "test_vid_123")).first()
            self.assertEqual(len(video.topics), 1)
            self.assertEqual(video.topics[0].name, "Test Topic")
            self.assertEqual(len(video.keywords), 1)
            self.assertEqual(video.entities[0].name, "Test Brand")
            self.assertEqual(len(video.scenes), 1)
            self.assertEqual(len(video.ocr_results), 1)
            self.assertEqual(video.ocr_results[0].text, "Hello World")

    @patch("video_analyzer.cli.get_provider")
    @patch("video_analyzer.cli.get_video_info")
    @patch("video_analyzer.cli.download_video")
    @patch("video_analyzer.cli.extract_audio")
    @patch("video_analyzer.cli.extract_frames")
    @patch("video_analyzer.cli.transcribe_audio")
    @patch("video_analyzer.cli.analyze_frames")
    @patch("video_analyzer.cli.detect_hooks")
    def test_analyze_flow_with_insights(self, mock_hooks, mock_frames, mock_transcribe, mock_ext_frames, mock_ext_audio, mock_download, mock_info, mock_provider_factory):
        """Test the full analysis flow including advanced insights."""
        # Setup mocks
        mock_info.return_value = MagicMock(video_id="vid1", title="Title", channel="Channel", duration=100, thumbnail_url="")
        mock_ext_audio.return_value = Path("audio.mp3")
        mock_ext_frames.return_value = [Path("f1.jpg")]
        mock_transcribe.return_value = "This is a test transcript."
        mock_frames.return_value = [{"timestamp": 0.0, "frame_path": "f1.jpg", "description": "desc", "objects": [], "scene_type": "talking_head", "text_visible": True}]
        mock_hooks.return_value = []
        
        mock_provider = MagicMock()
        mock_provider_factory.return_value = mock_provider
        mock_provider.extract_topics_and_keywords.return_value = {
            "topics": [{"name": "Topic A", "confidence": 0.9}],
            "keywords": [{"text": "Keyword 1", "confidence": 0.8}]
        }
        mock_provider.extract_entities.return_value = [{"name": "Brand X", "type": "Brand", "count": 1, "description": "X"}]
        mock_provider.perform_ocr.return_value = [{"text": "OCR Text", "left": 0.1, "top": 0.1, "width": 0.2, "height": 0.1}]
        mock_provider.detect_scenes_and_shots.return_value = [{"start_frame_index": 0, "end_frame_index": 5, "label": "Intro"}]
        
        # Run analyze command
        result = runner.invoke(app, ["analyze", "https://youtube.com/watch?v=vid1"])
        
        # Verify result
        self.assertEqual(result.exit_code, 0)
        
        # Check database
        with get_session() as session:
            video = session.exec(select(Video).where(Video.video_id == "vid1")).first()
            self.assertIsNotNone(video)
            self.assertEqual(len(video.topics), 1)
            self.assertEqual(video.topics[0].name, "Topic A")
            self.assertEqual(len(video.entities), 1)
            self.assertEqual(video.entities[0].name, "Brand X")
            self.assertEqual(len(video.scenes), 1)
            self.assertEqual(len(video.ocr_results), 1)
            self.assertEqual(video.ocr_results[0].text, "OCR Text")

    def test_summary_command(self):
        """Test the summary command display."""
        with get_session() as session:
            v = Video(video_id="v_summ", url="url", title="Summary Vid", channel="Chan", duration_seconds=60, status="completed")
            session.add(v)
            session.commit()
            session.refresh(v)
            
            session.add(Topic(video_id=v.id, name="AI News"))
            session.add(Keyword(video_id=v.id, text="LLM"))
            session.add(Entity(video_id=v.id, name="Google", type="Brand"))
            session.add(Scene(video_id=v.id, start_time=0, end_time=30, label="Section 1"))
            session.add(OCRResult(video_id=v.id, frame_id=1, text="Headline", left=0, top=0, width=1, height=1))
            session.commit()
            
        result = runner.invoke(app, ["summary", "v_summ"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Summary Vid", result.output)
        self.assertIn("AI News", result.output)
        self.assertIn("Google", result.output)
        self.assertIn("Section 1", result.output)
        self.assertIn("Headline", result.output)

if __name__ == "__main__":
    unittest.main()
