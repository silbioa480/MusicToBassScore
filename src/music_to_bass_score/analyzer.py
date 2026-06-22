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


def _score_key(chroma_vec: np.ndarray) -> str:
    """Return best-fit key string for a 12-element mean chroma vector."""
    best_score = -np.inf
    best_key = "C major"
    for root_idx in range(12):
        rotated = np.roll(chroma_vec, -root_idx)
        major_score = float(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
        if major_score > best_score:
            best_score = major_score
            best_key = f"{KEY_NAMES[root_idx]} major"
        minor_score = float(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{KEY_NAMES[root_idx]} minor"
    return best_key


def _estimate_key(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_chroma=12
    )
    return _score_key(chroma.mean(axis=1))


def detect_key_per_section(
    audio_path: Path,
    measure_grid: list[float],
    section_measures: int = 8,
    stride_measures: int = 2,
    min_stable: int = 4,
    initial_key: Optional[str] = None,
) -> list[str]:
    """Detect key modulations by sliding-window chroma analysis.

    Returns one key string per measure (same length as measure_grid).
    A new key is only accepted after min_stable consecutive measures
    agree on it, preventing noisy short-region flips.

    initial_key seeds the stability filter with the known global key so that
    a short ambiguous intro does not get mislabelled before the song settles.
    """
    from collections import Counter

    n = len(measure_grid)
    if n == 0:
        return []

    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH, n_chroma=12)
    total_frames = chroma.shape[1]

    measure_frames = [min(int(t * sr / HOP_LENGTH), total_frames) for t in measure_grid]

    # Each measure collects candidate key votes from all windows covering it.
    votes: list[list[str]] = [[] for _ in range(n)]
    for w_start in range(0, n, stride_measures):
        w_end = min(w_start + section_measures, n)
        f_start = measure_frames[w_start]
        f_end = measure_frames[w_end] if w_end < n else total_frames
        if f_end <= f_start:
            continue
        candidate = _score_key(chroma[:, f_start:f_end].mean(axis=1))
        for m_idx in range(w_start, w_end):
            votes[m_idx].append(candidate)

    # Majority vote per measure.
    raw_keys = [
        Counter(v).most_common(1)[0][0] if v else "C major"
        for v in votes
    ]

    # Stability filter: require min_stable consecutive measures before switching key.
    # Seed from the global key when available — prevents short ambiguous intros from
    # being mislabelled before the song's harmonic context is fully established.
    current = initial_key if initial_key else raw_keys[0]
    stable: list[str] = [current]
    pending: Optional[str] = None
    pending_count = 0
    for k in raw_keys[1:]:
        # Relative keys (G major ↔ E minor) share a key signature and cannot be told
        # apart reliably from chroma alone — treat them as the current key, never a modulation.
        if k == current or _is_relative_key(current, k):
            pending = None
            pending_count = 0
            stable.append(current)
        elif k == pending:
            pending_count += 1
            if pending_count >= min_stable:
                current = k
                pending = None
                pending_count = 0
            stable.append(current)
        else:
            pending = k
            pending_count = 1
            stable.append(current)

    # Log detected key changes.
    transitions = []
    prev = stable[0]
    for i, k in enumerate(stable[1:], 1):
        if k != prev:
            transitions.append(f"m{i}: {prev} → {k}")
            prev = k
    if transitions:
        logger.info("Key modulations detected (%d): %s", len(transitions), ", ".join(transitions))
    else:
        logger.info("No key modulation detected: %s (all %d measures)", stable[0], n)

    return stable


_MAJOR_INTERVALS = frozenset([0, 2, 4, 5, 7, 9, 11])
_MINOR_INTERVALS = frozenset([0, 2, 3, 5, 7, 8, 10])

_NOTE_SEMITONES_A: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def _is_relative_key(k1: str, k2: str) -> bool:
    """True iff k1 and k2 are relative major/minor (same key signature, different tonic).

    G major ↔ E minor share the same 7 pitches (1 sharp). The relative minor of a
    major key lies 9 semitones above (or 3 below) the major tonic.
    """
    parts1, parts2 = k1.split(), k2.split()
    if len(parts1) < 2 or len(parts2) < 2:
        return False
    tonic1, mode1 = parts1[0], parts1[1]
    tonic2, mode2 = parts2[0], parts2[1]
    if mode1 == mode2:
        return False
    sem1 = _NOTE_SEMITONES_A.get(tonic1, -1)
    sem2 = _NOTE_SEMITONES_A.get(tonic2, -1)
    if sem1 < 0 or sem2 < 0:
        return False
    diff = (sem2 - sem1) % 12
    if mode1 == "major" and mode2 == "minor":
        return diff == 9
    if mode1 == "minor" and mode2 == "major":
        return diff == 3
    return False


def _diatonic_count(chord_labels_window: list, key_str: str) -> int:
    """Count chords in the window whose root is diatonic to key_str."""
    parts = key_str.strip().split()
    tonic = parts[0] if parts else "C"
    mode = parts[1].lower() if len(parts) > 1 else "major"
    intervals = _MAJOR_INTERVALS if mode == "major" else _MINOR_INTERVALS
    tonic_semi = _NOTE_SEMITONES_A.get(tonic, 0)
    count = 0
    for measure in chord_labels_window:
        for _, sym in measure:
            if sym in ("N.C.", "NC", ""):
                continue
            sym_clean = sym.split("/")[0]
            if sym_clean.endswith("?"):
                sym_clean = sym_clean[:-1]
            root = sym_clean[:2] if len(sym_clean) > 1 and sym_clean[1] in ("#", "b") else sym_clean[:1]
            note_semi = _NOTE_SEMITONES_A.get(root)
            if note_semi is None:
                continue
            if (note_semi - tonic_semi) % 12 in intervals:
                count += 1
    return count


def refine_key_with_chords(
    key_labels: list[str],
    chord_labels: list,
    window: int = 8,
) -> list[str]:
    """Re-score key labels using diatonic chord membership to resolve parallel-key ambiguity.

    For each sliding window of `window` measures, counts how many chord roots are diatonic
    to the current key vs the parallel key (same tonic, opposite mode). If the parallel key
    achieves a significantly higher diatonic count (> 1.3×), the window is re-assigned.

    This resolves A major vs A minor confusion that pure chroma matching cannot distinguish.
    Relative-key pairs (G major ↔ E minor) are NOT switched here — they share a key signature
    and are suppressed upstream in detect_key_per_section().
    """
    n = len(key_labels)
    refined = list(key_labels)
    stride = max(1, window // 2)

    for i in range(0, n, stride):
        w_end = min(i + window, n)
        current_key = key_labels[i]
        parts = current_key.split()
        tonic = parts[0] if parts else "C"
        mode = parts[1].lower() if len(parts) > 1 else "major"
        parallel_mode = "minor" if mode == "major" else "major"
        parallel_key = f"{tonic} {parallel_mode}"

        window_chords = chord_labels[i:w_end]
        current_score = _diatonic_count(window_chords, current_key)
        parallel_score = _diatonic_count(window_chords, parallel_key)

        if parallel_score > current_score * 1.3:
            for j in range(i, w_end):
                refined[j] = parallel_key

    changes = [i for i in range(n) if key_labels[i] != refined[i]]
    if changes:
        logger.info(
            "Key refinement via chord analysis: %d measure(s) changed, first at m%d (%s → %s)",
            len(changes), changes[0], key_labels[changes[0]], refined[changes[0]],
        )
    return refined


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
