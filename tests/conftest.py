"""Shared pytest fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ASSETS_DIR = Path(__file__).parent.parent / "assets"
TMP_TEST_DIR = Path(__file__).parent.parent / "tmp" / "test"


@pytest.fixture(scope="session")
def sample_wav_path(tmp_path_factory) -> Path:
    """Generate a short synthetic bass-like WAV for testing (no real audio needed)."""
    out_dir = tmp_path_factory.mktemp("audio")
    wav_path = out_dir / "test_bass.wav"

    sr = 44100
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Simulate bass note pattern: A1 (55 Hz) with harmonics
    y = (
        0.5 * np.sin(2 * np.pi * 55 * t)
        + 0.25 * np.sin(2 * np.pi * 110 * t)
        + 0.1 * np.sin(2 * np.pi * 165 * t)
    )
    y = (y * 0.8).astype(np.float32)

    sf.write(str(wav_path), y, sr)
    return wav_path


@pytest.fixture(scope="session")
def sample_note_events():
    """Minimal list of NoteEvent for score building tests."""
    from music_to_bass_score.transcriber import NoteEvent
    return [
        NoteEvent(pitch=40, start_sec=0.0, end_sec=0.5, velocity=80),
        NoteEvent(pitch=45, start_sec=0.5, end_sec=1.0, velocity=80),
        NoteEvent(pitch=38, start_sec=1.0, end_sec=1.5, velocity=80),
        NoteEvent(pitch=40, start_sec=1.5, end_sec=2.0, velocity=80),
    ]


@pytest.fixture(scope="session")
def sample_metadata(tmp_path_factory):
    """Minimal SongMetadata for tests."""
    from pathlib import Path
    from music_to_bass_score.downloader import SongMetadata
    return SongMetadata(
        title="Test Song",
        artist="Test Artist",
        duration_sec=120.0,
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        audio_path=Path("/tmp/test.wav"),
    )


@pytest.fixture(scope="session")
def sample_analysis():
    """Minimal AudioAnalysis for tests."""
    from music_to_bass_score.analyzer import AudioAnalysis
    return AudioAnalysis(
        bpm=120.0,
        bpm_rounded=120,
        key="A minor",
        time_signature_num=4,
        time_signature_den=4,
        duration_sec=120.0,
        sample_rate=44100,
    )
