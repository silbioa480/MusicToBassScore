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
    bpm = 120.0
    duration = 8.0  # longer for reliable BPM detection
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Bass tones: A1(55Hz), C2(65Hz), E2(82Hz), G2(98Hz) per beat
    y = np.zeros(n_samples, dtype=np.float32)
    beat_period = int(sr * 60.0 / bpm)
    notes = [55.0, 65.4, 82.4, 98.0]
    for i, freq in enumerate(notes * int(duration)):
        start = i * beat_period
        end = min(start + beat_period, n_samples)
        if start >= n_samples:
            break
        seg_t = t[start:end]
        tone = (
            0.5 * np.sin(2 * np.pi * freq * seg_t)
            + 0.2 * np.sin(2 * np.pi * freq * 2 * seg_t)
        )
        # Sharp attack, quick decay — makes beat onsets detectable
        decay = np.exp(-np.linspace(0, 6, end - start))
        y[start:end] += (tone * decay).astype(np.float32)

    y = np.clip(y / (np.abs(y).max() + 1e-8) * 0.8, -1.0, 1.0).astype(np.float32)

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
