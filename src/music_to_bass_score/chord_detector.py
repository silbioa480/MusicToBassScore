"""Per-measure chord detection using librosa chroma features."""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .config import HOP_LENGTH, SAMPLE_RATE
from .logger import get_logger

logger = get_logger(__name__)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_MAJOR_INTERVALS  = [0, 4, 7]
_MINOR_INTERVALS  = [0, 3, 7]
_DOM7_INTERVALS   = [0, 4, 7, 10]
_MIN7_INTERVALS   = [0, 3, 7, 10]
_MAJ7_INTERVALS   = [0, 4, 7, 11]


def _build_templates() -> dict[str, np.ndarray]:
    templates = {}
    for root in range(12):
        name = _NOTE_NAMES[root]
        for intervals, suffix in [
            (_MAJOR_INTERVALS,  ""),
            (_MINOR_INTERVALS,  "m"),
            (_DOM7_INTERVALS,   "7"),
            (_MIN7_INTERVALS,   "m7"),
            (_MAJ7_INTERVALS,   "maj7"),
        ]:
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
    beat_times: Optional[list[float]] = None,
    sample_rate: int = SAMPLE_RATE,
) -> list[str]:
    """Detect chord labels for each measure of the audio.

    When beat_times is provided (actual beat positions from librosa beat tracker),
    chroma is segmented by real beat boundaries instead of fixed BPM frames —
    this eliminates drift from pickup measures and BPM estimation error.
    """
    logger.info(
        "Detecting chords: %s (bpm=%.1f time_sig=%d/4 beat_times=%s)",
        audio_path, bpm, time_sig_num,
        f"{len(beat_times)} beats, first={beat_times[0]:.3f}s" if beat_times else "none",
    )
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)

    if beat_times and len(beat_times) >= time_sig_num:
        chord_labels = _chords_from_beat_times(chroma, beat_times, time_sig_num, bpm, sr)
    else:
        logger.warning("beat_times unavailable, falling back to fixed-BPM segmentation")
        chord_labels = _chords_from_fixed_bpm(chroma, bpm, time_sig_num, sr)

    logger.info("Chord detection complete: %d measures — %s", len(chord_labels), chord_labels[:8])
    return chord_labels


def _chords_from_beat_times(
    chroma: np.ndarray,
    beat_times: list[float],
    time_sig_num: int,
    bpm: float,
    sr: int,
) -> list[str]:
    """Segment chroma by actual detected beat positions for accurate alignment."""
    beat_dur = 60.0 / bpm
    chord_labels = []

    for i in range(0, len(beat_times), time_sig_num):
        start_t = beat_times[i]
        end_t = (beat_times[i + time_sig_num]
                 if i + time_sig_num < len(beat_times)
                 else beat_times[-1] + beat_dur)

        sf = librosa.time_to_frames(start_t, sr=sr, hop_length=HOP_LENGTH)
        ef = librosa.time_to_frames(end_t,   sr=sr, hop_length=HOP_LENGTH)
        ef = max(sf + 1, min(ef, chroma.shape[1]))

        segment = chroma[:, sf:ef].mean(axis=1)
        chord_labels.append(_chroma_to_chord(segment))

    return chord_labels


def _chords_from_fixed_bpm(
    chroma: np.ndarray,
    bpm: float,
    time_sig_num: int,
    sr: int,
) -> list[str]:
    """Fallback: fixed BPM-based measure segmentation starting at frame 0."""
    beat_period_frames  = (60.0 / bpm) * (sr / HOP_LENGTH)
    measure_period_frames = beat_period_frames * time_sig_num

    n_frames = chroma.shape[1]
    chord_labels = []
    measure_start = 0.0

    while measure_start < n_frames:
        measure_end = min(measure_start + measure_period_frames, n_frames)
        s = int(measure_start)
        e = max(s + 1, int(measure_end))
        segment = chroma[:, s:e].mean(axis=1)
        chord_labels.append(_chroma_to_chord(segment))
        measure_start += measure_period_frames

    return chord_labels


def _chroma_to_chord(chroma_vector: np.ndarray) -> str:
    """Template-match a 12-dimensional chroma vector to the nearest chord."""
    norm = np.linalg.norm(chroma_vector)
    if norm < 1e-6:
        return "N.C."

    normalized = chroma_vector / norm
    best_chord = "C"
    best_score = -np.inf

    for chord_name, template in _CHORD_TEMPLATES.items():
        score = float(np.dot(normalized, template))
        if score > best_score:
            best_score = score
            best_chord = chord_name

    return best_chord
