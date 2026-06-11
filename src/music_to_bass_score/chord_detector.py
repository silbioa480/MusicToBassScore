"""Per-measure chord detection using librosa chroma features (full-mix based).

Detects chord SYMBOLS (root + quality) per half-measure (2-beat resolution) from the
FULL mix — the full mix contains the harmony instruments needed to determine chord
quality (major/minor/7th), which a monophonic bass line cannot provide. Returns two
chord symbols per measure to capture fast harmonic rhythm.
"""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .config import HOP_LENGTH, SAMPLE_RATE
from .logger import get_logger

logger = get_logger(__name__)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# (intervals, symbol-suffix) — symbol suffix is appended to the root name
_CHORD_QUALITIES = [
    ([0, 4, 7], ""),        # major triad     → C
    ([0, 3, 7], "m"),       # minor triad     → Cm
    ([0, 4, 7, 10], "7"),   # dominant 7th    → C7
    ([0, 3, 7, 10], "m7"),  # minor 7th       → Cm7
    ([0, 4, 7, 11], "maj7"),# major 7th       → Cmaj7
    ([0, 3, 6], "dim"),     # diminished      → Cdim
]


def _build_templates() -> dict[str, np.ndarray]:
    templates = {}
    for root in range(12):
        name = _NOTE_NAMES[root]
        for intervals, suffix in _CHORD_QUALITIES:
            vec = np.zeros(12)
            for interval in intervals:
                vec[(root + interval) % 12] = 1.0
            vec /= np.linalg.norm(vec)
            templates[f"{name}{suffix}"] = vec
    return templates


_CHORD_TEMPLATES = _build_templates()


def detect_chords_per_measure(
    audio_path: Path,
    bpm: float,
    time_sig_num: int,
    measure_grid: Optional[list[float]] = None,
    sample_rate: int = SAMPLE_RATE,
) -> list[list[str]]:
    """Detect chord symbols for each measure (two per measure: halves).

    Returns a list (one per measure) of label lists. In 4/4 each inner list has 2
    entries — the chord over beats 1-2 and over beats 3-4. `audio_path` should be the
    FULL mix (harmony needed for chord quality).
    """
    logger.info(
        "Detecting chords (full-mix, 2/measure): %s (bpm=%.1f time_sig=%d/4 grid=%s)",
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

    preview = [f"{m[0]}-{m[1]}" if len(m) == 2 else "/".join(m) for m in labels[:6]]
    logger.info("Chord detection complete: %d measures — %s", len(labels), preview)
    return labels


def _match_chord(chroma: np.ndarray, start_frame: int, end_frame: int) -> str:
    """Template-match the mean chroma over a frame range to the nearest chord symbol."""
    end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))
    if start_frame >= chroma.shape[1]:
        return "N.C."
    seg = chroma[:, start_frame:end_frame].mean(axis=1)
    norm = np.linalg.norm(seg)
    if norm < 1e-6:
        return "N.C."
    seg = seg / norm

    best_name, best_score = "N.C.", -np.inf
    for name, template in _CHORD_TEMPLATES.items():
        score = float(np.dot(seg, template))
        if score > best_score:
            best_score, best_name = score, name
    return best_name


def _chords_from_grid(
    chroma: np.ndarray,
    measure_grid: list[float],
    time_sig_num: int,
    bpm: float,
    sr: int,
) -> list[list[str]]:
    """Detect two chords per measure using constant-tempo measure boundaries."""
    seconds_per_measure = (60.0 / bpm) * time_sig_num
    half = seconds_per_measure / 2.0
    labels: list[list[str]] = []

    for m_start in measure_grid:
        mid = m_start + half
        m_end = m_start + seconds_per_measure

        sf1 = librosa.time_to_frames(m_start, sr=sr, hop_length=HOP_LENGTH)
        mf = librosa.time_to_frames(mid, sr=sr, hop_length=HOP_LENGTH)
        ef = librosa.time_to_frames(m_end, sr=sr, hop_length=HOP_LENGTH)

        labels.append([_match_chord(chroma, sf1, mf), _match_chord(chroma, mf, ef)])

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
        labels.append([_match_chord(chroma, s1, s2), _match_chord(chroma, s2, s3)])
        pos += 2 * half_frames

    return labels
