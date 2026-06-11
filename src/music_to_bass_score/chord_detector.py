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
) -> list[list[tuple[float, str]]]:
    """Detect chord changes per measure at beat resolution.

    Returns a list (one entry per measure) of (beat_offset, chord_symbol) tuples.
    Each measure is analyzed beat-by-beat, then consecutive identical chords are
    collapsed — so a measure with one chord yields one entry, and a measure where the
    chord changes N times yields N entries at their respective beat offsets.
    `audio_path` should be the FULL mix (harmony needed for chord quality).
    """
    logger.info(
        "Detecting chords (full-mix, per-beat+collapse): %s (bpm=%.1f time_sig=%d/4 grid=%s)",
        audio_path, bpm, time_sig_num,
        f"{len(measure_grid)} measures" if measure_grid else "none",
    )
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)

    if measure_grid and len(measure_grid) >= 1:
        measures = _chords_from_grid(chroma, measure_grid, time_sig_num, bpm, sr)
    else:
        logger.warning("measure_grid unavailable, falling back to fixed-BPM segmentation")
        measures = _chords_from_fixed_bpm(chroma, bpm, time_sig_num, sr)

    preview = ["|".join(s for _, s in m) for m in measures[:6]]
    logger.info("Chord detection complete: %d measures — %s", len(measures), preview)
    return measures


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


# Half-measure (2-segment) resolution: smoother/more reliable than per-beat template
# matching, which jitters every beat. Each measure yields its two half-measure chords,
# then identical halves collapse to one — so a steady measure shows one chord and a
# changing measure shows two, at the beat offsets where the change occurs.
_SEGMENTS_PER_MEASURE = 2


def _measure_segments(time_sig_num: int) -> list[tuple[float, float]]:
    """Return (start_fraction, end_fraction) of each detection segment within a measure."""
    n = _SEGMENTS_PER_MEASURE
    return [(i / n, (i + 1) / n) for i in range(n)]


def _collapse_segments(seg_chords: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Collapse consecutive identical (offset, chord) segments, keeping first offset."""
    result: list[tuple[float, str]] = []
    prev = None
    for off, chord in seg_chords:
        if chord != prev:
            result.append((off, chord))
            prev = chord
    return result or [(0.0, "N.C.")]


def _chords_from_grid(
    chroma: np.ndarray,
    measure_grid: list[float],
    time_sig_num: int,
    bpm: float,
    sr: int,
) -> list[list[tuple[float, str]]]:
    """Half-measure detection on the constant-tempo grid, identical segments collapsed."""
    seconds_per_measure = (60.0 / bpm) * time_sig_num
    segs = _measure_segments(time_sig_num)
    measures: list[list[tuple[float, str]]] = []

    for m_start in measure_grid:
        seg_chords = []
        for s_frac, e_frac in segs:
            bs = m_start + s_frac * seconds_per_measure
            be = m_start + e_frac * seconds_per_measure
            sfb = librosa.time_to_frames(bs, sr=sr, hop_length=HOP_LENGTH)
            efb = librosa.time_to_frames(be, sr=sr, hop_length=HOP_LENGTH)
            beat_offset = s_frac * time_sig_num
            seg_chords.append((beat_offset, _match_chord(chroma, sfb, efb)))
        measures.append(_collapse_segments(seg_chords))

    return measures


def _chords_from_fixed_bpm(
    chroma: np.ndarray,
    bpm: float,
    time_sig_num: int,
    sr: int,
) -> list[list[tuple[float, str]]]:
    """Fallback: fixed-BPM half-measure segmentation from frame 0, identical collapsed."""
    beat_frames = (60.0 / bpm) * (sr / HOP_LENGTH)
    measure_frames = beat_frames * time_sig_num
    segs = _measure_segments(time_sig_num)

    n_frames = chroma.shape[1]
    measures: list[list[tuple[float, str]]] = []
    pos = 0.0

    while pos < n_frames:
        seg_chords = []
        for s_frac, e_frac in segs:
            s = int(pos + s_frac * measure_frames)
            e = int(pos + e_frac * measure_frames)
            seg_chords.append((s_frac * time_sig_num, _match_chord(chroma, s, e)))
        measures.append(_collapse_segments(seg_chords))
        pos += measure_frames

    return measures
