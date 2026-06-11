"""Per-measure chord detection using librosa chroma features.

Detects the bass ROOT per half-measure (2-beat resolution) from the separated bass
stem. Bass is essentially monophonic, so chord *quality* (maj/min/7) cannot be
reliably inferred from it — the root is the reliable, useful signal for a bassist.
Returns two labels per measure (first half, second half) to capture fast harmonic
rhythm that full-measure averaging would blur together.
"""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .config import HOP_LENGTH, SAMPLE_RATE
from .logger import get_logger

logger = get_logger(__name__)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_chords_per_measure(
    audio_path: Path,
    bpm: float,
    time_sig_num: int,
    beat_times: Optional[list[float]] = None,
    measure_grid: Optional[list[float]] = None,
    sample_rate: int = SAMPLE_RATE,
) -> list[list[str]]:
    """Detect chord-root labels for each measure (two per measure: halves).

    Returns a list (one entry per measure) of label lists. With a 4/4 measure the
    inner list has 2 entries — the root over beats 1-2 and over beats 3-4.

    When measure_grid (constant-tempo measure-start times) is provided, segmentation
    uses it for stable, jitter-free boundaries. Otherwise falls back to fixed-BPM frames.
    """
    logger.info(
        "Detecting chords (2/measure): %s (bpm=%.1f time_sig=%d/4 grid=%s)",
        audio_path, bpm, time_sig_num,
        f"{len(measure_grid)} measures" if measure_grid else "none",
    )
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)

    if measure_grid and len(measure_grid) >= 1:
        labels = _chords_from_grid(chroma, measure_grid, time_sig_num, bpm, sr)
    else:
        logger.warning("measure_grid unavailable, falling back to fixed-BPM segmentation")
        labels = _chords_from_fixed_bpm(chroma, bpm, time_sig_num, sr)

    flat_preview = [f"{m[0]}-{m[1]}" if len(m) == 2 else "/".join(m) for m in labels[:6]]
    logger.info("Chord detection complete: %d measures — %s", len(labels), flat_preview)
    return labels


def _root_of_segment(chroma: np.ndarray, start_frame: int, end_frame: int) -> str:
    """Return the dominant pitch-class name (bass root) over a frame range."""
    end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))
    if start_frame >= chroma.shape[1]:
        return "N.C."
    seg = chroma[:, start_frame:end_frame].mean(axis=1)
    if np.linalg.norm(seg) < 1e-6:
        return "N.C."
    return _NOTE_NAMES[int(np.argmax(seg))]


def _chords_from_grid(
    chroma: np.ndarray,
    measure_grid: list[float],
    time_sig_num: int,
    bpm: float,
    sr: int,
) -> list[list[str]]:
    """Detect two roots per measure using constant-tempo measure boundaries."""
    seconds_per_measure = (60.0 / bpm) * time_sig_num
    half = seconds_per_measure / 2.0
    labels: list[list[str]] = []

    for m_start in measure_grid:
        mid = m_start + half
        m_end = m_start + seconds_per_measure

        sf1 = librosa.time_to_frames(m_start, sr=sr, hop_length=HOP_LENGTH)
        mf = librosa.time_to_frames(mid, sr=sr, hop_length=HOP_LENGTH)
        ef = librosa.time_to_frames(m_end, sr=sr, hop_length=HOP_LENGTH)

        first = _root_of_segment(chroma, sf1, mf)
        second = _root_of_segment(chroma, mf, ef)
        labels.append([first, second])

    return labels


def _chords_from_fixed_bpm(
    chroma: np.ndarray,
    bpm: float,
    time_sig_num: int,
    sr: int,
) -> list[list[str]]:
    """Fallback: fixed-BPM half-measure segmentation starting at frame 0."""
    beat_period_frames = (60.0 / bpm) * (sr / HOP_LENGTH)
    half_frames = beat_period_frames * (time_sig_num / 2.0)

    n_frames = chroma.shape[1]
    labels: list[list[str]] = []
    pos = 0.0

    while pos < n_frames:
        s1 = int(pos)
        s2 = int(pos + half_frames)
        s3 = int(pos + 2 * half_frames)
        first = _root_of_segment(chroma, s1, s2)
        second = _root_of_segment(chroma, s2, s3)
        labels.append([first, second])
        pos += 2 * half_frames

    return labels
