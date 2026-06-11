"""Audio analysis: BPM, key, time signature using librosa."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .config import HOP_LENGTH, N_FFT, SAMPLE_RATE
from .logger import get_logger

logger = get_logger(__name__)

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class AudioAnalysis:
    bpm: float
    bpm_rounded: int
    key: str
    time_signature_num: int
    time_signature_den: int
    duration_sec: float
    sample_rate: int


def analyze_audio(audio_path: Path) -> AudioAnalysis:
    """Load audio and extract BPM, key, and time signature."""
    logger.info("Analyzing audio: %s", audio_path)
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        bpm = _estimate_bpm(y, sr)
        key = _estimate_key(y, sr)
        time_sig_num, time_sig_den = _estimate_time_signature(y, sr, bpm)

        logger.info(
            "Analysis done: bpm=%.1f key=%r time_sig=%d/%d duration=%.1fs",
            bpm, key, time_sig_num, time_sig_den, duration,
        )
        return AudioAnalysis(
            bpm=bpm,
            bpm_rounded=round(bpm),
            key=key,
            time_signature_num=time_sig_num,
            time_signature_den=time_sig_den,
            duration_sec=duration,
            sample_rate=sr,
        )
    except Exception as exc:
        logger.error("Audio analysis failed: %s", exc, exc_info=True)
        raise


def _estimate_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    return float(tempo)


def _estimate_key(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_chroma=12
    )
    mean_chroma = chroma.mean(axis=1)

    best_score = -np.inf
    best_key = "C major"

    for root_idx in range(12):
        rotated = np.roll(mean_chroma, -root_idx)

        major_score = float(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
        if major_score > best_score:
            best_score = major_score
            best_key = f"{KEY_NAMES[root_idx]} major"

        minor_score = float(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{KEY_NAMES[root_idx]} minor"

    return best_key


def _estimate_time_signature(
    y: np.ndarray, sr: int, tempo: float
) -> tuple[int, int]:
    """Estimate time signature by checking 3/4 vs 4/4 beat grouping."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    beats_per_bar_3 = _grouping_score(onset_env, sr, tempo, grouping=3)
    beats_per_bar_4 = _grouping_score(onset_env, sr, tempo, grouping=4)

    if beats_per_bar_3 > beats_per_bar_4 * 1.1:
        return (3, 4)
    return (4, 4)


def _grouping_score(
    onset_env: np.ndarray, sr: int, tempo: float, grouping: int
) -> float:
    """Score how well the onset envelope aligns with a given beat grouping."""
    beat_period_frames = (60.0 / tempo) * (sr / HOP_LENGTH)
    bar_period_frames = beat_period_frames * grouping

    if bar_period_frames < 1:
        return 0.0

    n_frames = len(onset_env)
    bar_indices = np.arange(0, n_frames, bar_period_frames).astype(int)
    bar_indices = bar_indices[bar_indices < n_frames]

    if len(bar_indices) == 0:
        return 0.0

    return float(onset_env[bar_indices].mean())
