"""Audio analysis: BPM, key, time signature using librosa."""

from dataclasses import dataclass, field
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
    beat_times: list[float] = field(default_factory=list)


def analyze_audio(audio_path: Path) -> AudioAnalysis:
    """Load audio and extract BPM, key, and time signature."""
    logger.info("Analyzing audio: %s", audio_path)
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        bpm, beat_frames = _estimate_bpm_and_beats(y, sr)
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=HOP_LENGTH
        ).tolist()
        key = _estimate_key(y, sr)
        time_sig_num, time_sig_den = _estimate_time_signature(y, sr, bpm)

        logger.info(
            "Analysis done: bpm=%.1f key=%r time_sig=%d/%d duration=%.1fs beats=%d first_beat=%.3fs",
            bpm, key, time_sig_num, time_sig_den, duration,
            len(beat_times), beat_times[0] if beat_times else 0.0,
        )
        return AudioAnalysis(
            bpm=bpm,
            bpm_rounded=round(bpm),
            key=key,
            time_signature_num=time_sig_num,
            time_signature_den=time_sig_den,
            duration_sec=duration,
            sample_rate=sr,
            beat_times=beat_times,
        )
    except Exception as exc:
        logger.error("Audio analysis failed: %s", exc, exc_info=True)
        raise


def detect_first_onset(audio_path: Path, sr: int = SAMPLE_RATE) -> float:
    """Return the time (seconds) of the first onset in the audio.

    Used as a downbeat anchor for the constant-tempo measure grid. Far more stable
    than librosa beat_track's jittery beat positions for building measure boundaries.
    """
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, backtrack=True)
    if len(onset_frames) == 0:
        return 0.0
    return float(librosa.frames_to_time(onset_frames[:1], sr=sr, hop_length=HOP_LENGTH)[0])


def build_measure_grid(
    bpm: float,
    beats_per_measure: int,
    anchor: float,
    duration_sec: float,
) -> list[float]:
    """Build constant-tempo measure-start times anchored at `anchor`.

    measure_start[i] = anchor + i * (60/bpm) * beats_per_measure

    A constant grid avoids the ±15% jitter of librosa beat_track, keeping notes and
    chords aligned to stable measure boundaries.
    """
    seconds_per_measure = (60.0 / bpm) * beats_per_measure
    if seconds_per_measure <= 0:
        return [0.0]
    grid = []
    t = anchor
    # Include a measure that starts slightly before the end so trailing notes land somewhere
    while t < duration_sec + seconds_per_measure:
        grid.append(round(t, 4))
        t += seconds_per_measure
    return grid


def _estimate_bpm_and_beats(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    tempo = _correct_tempo_octave(float(tempo), y, sr)
    return tempo, beats


# Plausible tempo band for octave candidates (J-pop / pop / rock covers).
_TEMPO_MIN = 55.0
_TEMPO_MAX = 210.0


def _correct_tempo_octave(tempo_bt: float, y: np.ndarray, sr: int) -> float:
    """Fix beat_track's frequent half/double-tempo (octave) errors.

    beat_track is built around a log-normal tempo prior centred at 120 BPM, so it
    routinely locks onto *half* the true tempo of a fast song (e.g. reports 86 for a
    172 BPM track, 75 for 150, 99 for 198) — the halved value sits closer to 120 in
    log space than the true tempo does, and the prior wins.

    An autocorrelation reference carries the same 120-centred bias, so instead we read
    the actual periodicity strength straight from the tempogram. Among the octave
    candidates (×0.5, ×1, ×2 of the beat_track tempo) that fall in a plausible band,
    we pick the one whose tempogram energy is highest — i.e. the periodicity the signal
    itself expresses most strongly. This snaps genuinely fast songs up to their true
    tempo without relying on any prior, while a song that really is slow keeps its
    lower octave because that is where its periodicity peaks.
    """
    if tempo_bt <= 0:
        return tempo_bt

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH
    )
    tempo_freqs = librosa.tempo_frequencies(
        tempogram.shape[0], hop_length=HOP_LENGTH, sr=sr
    )
    mean_tempogram = tempogram.mean(axis=1)

    def energy_at(bpm: float) -> float:
        idx = int(np.argmin(np.abs(tempo_freqs - bpm)))
        return float(mean_tempogram[idx])

    candidates = [
        bpm
        for bpm in (tempo_bt * 0.5, tempo_bt, tempo_bt * 2.0)
        if _TEMPO_MIN <= bpm <= _TEMPO_MAX
    ]
    if not candidates:  # beat_track tempo itself out of band — keep it as-is
        return tempo_bt

    corrected = max(candidates, key=energy_at)
    if abs(corrected - tempo_bt) > 1e-3:
        logger.info(
            "Tempo octave corrected: beat_track=%.1f → %.1f "
            "(tempogram energy %.3f vs %.3f)",
            tempo_bt, corrected, energy_at(corrected), energy_at(tempo_bt),
        )
    return corrected


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
