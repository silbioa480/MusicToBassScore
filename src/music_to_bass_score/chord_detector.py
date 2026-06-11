"""Per-measure chord detection using librosa chroma features."""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .config import HOP_LENGTH, SAMPLE_RATE
from .logger import get_logger

logger = get_logger(__name__)

CHORD_TEMPLATES: dict[str, np.ndarray] = {}

_MAJOR_INTERVALS = [0, 4, 7]
_MINOR_INTERVALS = [0, 3, 7]
_DOM7_INTERVALS = [0, 4, 7, 10]
_MIN7_INTERVALS = [0, 3, 7, 10]

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _build_templates() -> dict[str, np.ndarray]:
    templates = {}
    for root in range(12):
        name = _NOTE_NAMES[root]
        for intervals, suffix in [
            (_MAJOR_INTERVALS, ""),
            (_MINOR_INTERVALS, "m"),
            (_DOM7_INTERVALS, "7"),
            (_MIN7_INTERVALS, "m7"),
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
    sample_rate: int = SAMPLE_RATE,
) -> list[str]:
    """Detect chord labels for each measure of the audio."""
    logger.info("Detecting chords: %s (bpm=%.1f, time_sig=%d/4)", audio_path, bpm, time_sig_num)
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)

    beat_period_frames = (60.0 / bpm) * (sr / HOP_LENGTH)
    measure_period_frames = beat_period_frames * time_sig_num

    n_frames = chroma.shape[1]
    chord_labels = []

    measure_start = 0.0
    while measure_start < n_frames:
        measure_end = min(measure_start + measure_period_frames, n_frames)
        start_idx = int(measure_start)
        end_idx = max(start_idx + 1, int(measure_end))

        segment_chroma = chroma[:, start_idx:end_idx].mean(axis=1)
        chord = _chroma_to_chord(segment_chroma)
        chord_labels.append(chord)

        measure_start += measure_period_frames

    logger.info("Chord detection complete: %d measures — %s", len(chord_labels), chord_labels[:8])
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
