"""Per-measure chord detection: BTC Transformer + bass-slash annotation.

Primary engine: BTC large-voca Transformer on the full mix.

Post-processing — slash bass:
  Independently detect the dominant bass note from a C2-C4 chroma (<262 Hz).
  When the bass note differs from the chord root, append it as a slash:
  "G/B", "Am7/E", "C/G", etc.  This makes walking-bass and inversion voicings
  explicit on the chart (e.g. C-G/B-Am-G descending bass line).

Note on treble-based root correction:
  Experiments showed that bass-guitar harmonics reach into the C5+ range
  (D2's 8th harmonic = D5 at 588 Hz > C5 = 523 Hz), so frequency filtering
  cannot reliably separate bass from upper-voice chroma.  Treble-root override
  was therefore removed — it introduced false corrections on correctly-identified
  root-position chords (C → G) without fixing bass-contaminated labels (D → A).
  Proper source separation (Demucs) would solve this but is too slow on CPU.

Falls back to the librosa chroma matcher if BTC is unavailable or errors.
"""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .config import HOP_LENGTH, SAMPLE_RATE
from .logger import get_logger

logger = get_logger(__name__)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# (intervals, symbol-suffix, weights) — weights emphasize root/3rd/5th/7th for robust
# template matching. Symbol suffix is appended to the root name.
_CHORD_QUALITIES = [
    ([0, 4, 7],      "",     [1.0, 0.8, 0.9]),         # major triad   → C
    ([0, 3, 7],      "m",    [1.0, 0.8, 0.9]),         # minor triad   → Cm
    ([0, 4, 7, 10],  "7",    [1.0, 0.8, 0.9, 0.7]),    # dominant 7th  → C7
    ([0, 3, 7, 10],  "m7",   [1.0, 0.8, 0.9, 0.7]),    # minor 7th     → Cm7
    ([0, 4, 7, 11],  "maj7", [1.0, 0.8, 0.9, 0.7]),    # major 7th     → Cmaj7
    ([0, 3, 6],      "dim",  [1.0, 0.8, 0.9]),         # diminished    → Cdim
]


def _build_templates() -> dict[str, np.ndarray]:
    templates = {}
    for root in range(12):
        name = _NOTE_NAMES[root]
        for intervals, suffix, weights in _CHORD_QUALITIES:
            vec = np.zeros(12)
            for interval, w in zip(intervals, weights):
                vec[(root + interval) % 12] = w
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
    """Detect chord changes per measure.

    Returns a list (one entry per measure) of (beat_offset, chord_symbol) tuples,
    with consecutive identical chords collapsed to change points.

    Primary engine: the pretrained BTC Transformer (large vocabulary). Falls back to
    the librosa chroma matcher if BTC is unavailable or errors. `audio_path` should be
    the FULL mix (harmony needed for chord quality).
    """
    if measure_grid and len(measure_grid) >= 1:
        try:
            from . import btc_chord
            if btc_chord.is_available():
                return _chords_from_btc(audio_path, measure_grid, time_sig_num, bpm)
            logger.warning("BTC model not found; using chroma fallback")
        except Exception as exc:
            logger.warning("BTC chord recognition failed (%s); using chroma fallback", exc)

    return _detect_chords_chroma(audio_path, bpm, time_sig_num, measure_grid, sample_rate)


# ---------------------------------------------------------------------------
# BTC path
# ---------------------------------------------------------------------------

def _chords_from_btc(
    audio_path: Path,
    measure_grid: list[float],
    time_sig_num: int,
    bpm: float,
) -> list[list[tuple[float, str]]]:
    """Map the BTC chord timeline onto half-measure windows, then add bass-slash."""
    from . import btc_chord

    logger.info(
        "Detecting chords (BTC large-voca): %s (bpm=%.1f time_sig=%d/4 grid=%d measures)",
        audio_path, bpm, time_sig_num, len(measure_grid),
    )
    timeline = btc_chord.recognize_chords(audio_path)
    seconds_per_measure = (60.0 / bpm) * time_sig_num
    segs = _measure_segments(time_sig_num)

    # Build per-segment raw chords from BTC overlap voting
    raw_segs: list[tuple[float, float, str]] = []
    for m_start in measure_grid:
        for s_frac, e_frac in segs:
            ws = m_start + s_frac * seconds_per_measure
            we = m_start + e_frac * seconds_per_measure
            chord = btc_chord.chord_at_window(timeline, ws, we)
            raw_segs.append((ws, we, chord))

    # Load audio once for bass detection
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    y_h = librosa.effects.harmonic(y, margin=4.0)

    # Bass chroma: C2-C4 (<262 Hz) — fundamental register of bass guitar
    bass_ch = librosa.feature.chroma_cqt(
        y=y_h, sr=sr, hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz("C2"), n_octaves=2,
    )
    times = librosa.frames_to_time(
        np.arange(bass_ch.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )

    corrected: list[tuple[float, float, str]] = []
    for ws, we, chord in raw_segs:
        chord = _apply_slash_bass(chord, bass_ch, times, ws, we)
        corrected.append((ws, we, chord))

    # Re-group into per-measure lists
    n_segs = len(segs)
    measures: list[list[tuple[float, str]]] = []
    for m_idx, m_start in enumerate(measure_grid):
        seg_list = []
        for s_idx, (s_frac, _) in enumerate(segs):
            flat_idx = m_idx * n_segs + s_idx
            if flat_idx >= len(corrected):
                break
            _ws, _we, chord = corrected[flat_idx]
            beat_offset = s_frac * time_sig_num
            seg_list.append((beat_offset, chord))
        measures.append(seg_list)

    measures = _smooth_chord_sequence(measures)
    preview = ["|".join(s for _, s in m) for m in measures[:6]]
    logger.info("BTC chord mapping complete: %d measures — %s", len(measures), preview)
    return measures


def _chord_root(symbol: str) -> str:
    """Extract root note name from a chord symbol (handles '#' accidentals)."""
    if not symbol or symbol in ("N.C.", "NC"):
        return ""
    if len(symbol) > 1 and symbol[1] in ("#", "b"):
        return symbol[:2]
    return symbol[:1]


def _chroma_window(ch: np.ndarray, times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Mean chroma vector over [t0, t1), normalised to [0,1]."""
    mask = (times >= t0) & (times < t1)
    if mask.sum() == 0:
        return np.zeros(12)
    vec = ch[:, mask].mean(axis=1)
    mx = vec.max()
    if mx < 1e-9:
        return vec
    return vec / mx


# Minimum confidence for bass note (normalised 0-1) to trigger slash notation.
# Set conservatively to avoid noise-induced slashes on ambiguous segments.
_BASS_MIN_CONFIDENCE = 0.72


def _apply_slash_bass(
    chord: str, bass_ch: np.ndarray, times: np.ndarray, t0: float, t1: float
) -> str:
    """Append a slash-bass note when the dominant bass note differs from the chord root.

    Uses C2-C4 chroma (fundamental register of bass guitar). Only appends when the
    bass note confidence exceeds _BASS_MIN_CONFIDENCE to suppress noisy segments.
    """
    if chord in ("N.C.", "NC"):
        return chord

    chord_root = _chord_root(chord.split("/")[0])
    if not chord_root:
        return chord

    bass_vec = _chroma_window(bass_ch, times, t0, t1)
    bass_idx = int(np.argmax(bass_vec))
    bass_conf = float(bass_vec[bass_idx])
    bass_note = _NOTE_NAMES[bass_idx]

    if bass_conf < _BASS_MIN_CONFIDENCE or bass_note == chord_root:
        return chord

    return f"{chord}/{bass_note}"


# ---------------------------------------------------------------------------
# Chroma fallback path
# ---------------------------------------------------------------------------

def _detect_chords_chroma(
    audio_path: Path,
    bpm: float,
    time_sig_num: int,
    measure_grid: Optional[list[float]],
    sample_rate: int,
) -> list[list[tuple[float, str]]]:
    """Fallback: librosa chroma template matching (half-measure resolution)."""
    logger.info(
        "Detecting chords (chroma fallback, half-measure): %s (bpm=%.1f time_sig=%d/4 grid=%s)",
        audio_path, bpm, time_sig_num,
        f"{len(measure_grid)} measures" if measure_grid else "none",
    )
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    chroma = _harmonic_chroma(y, sr)

    if measure_grid and len(measure_grid) >= 1:
        measures = _chords_from_grid(chroma, measure_grid, time_sig_num, bpm, sr)
    else:
        logger.warning("measure_grid unavailable, falling back to fixed-BPM segmentation")
        measures = _chords_from_fixed_bpm(chroma, bpm, time_sig_num, sr)

    measures = _smooth_chord_sequence(measures)

    preview = ["|".join(s for _, s in m) for m in measures[:6]]
    logger.info("Chord detection complete: %d measures — %s", len(measures), preview)
    return measures


def _harmonic_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    y_h = librosa.effects.harmonic(y, margin=4.0)
    tuning = librosa.estimate_tuning(y=y_h, sr=sr)
    chroma = librosa.feature.chroma_cens(
        y=y_h, sr=sr, hop_length=HOP_LENGTH, tuning=tuning, win_len_smooth=9
    )
    logger.debug("CENS chroma: tuning=%.3f semitones, shape=%s", tuning, chroma.shape)
    return chroma


def _smooth_chord_sequence(
    measures: list[list[tuple[float, str]]],
) -> list[list[tuple[float, str]]]:
    """Remove isolated single-segment outliers, then collapse identical neighbors."""
    flat: list[str] = []
    index: list[tuple[int, float]] = []
    for m_idx, segs in enumerate(measures):
        for off, chord in segs:
            flat.append(chord)
            index.append((m_idx, off))

    fixed = flat[:]
    for i in range(1, len(flat) - 1):
        if flat[i] != flat[i - 1] and flat[i - 1] == flat[i + 1]:
            fixed[i] = flat[i - 1]

    rebuilt: list[list[tuple[float, str]]] = [[] for _ in measures]
    for (m_idx, off), chord in zip(index, fixed):
        rebuilt[m_idx].append((off, chord))

    return [_collapse_segments(segs) if segs else [(0.0, "N.C.")] for segs in rebuilt]


def _match_chord(chroma: np.ndarray, start_frame: int, end_frame: int) -> str:
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

    root = best_name[:2] if len(best_name) > 1 and best_name[1] in ("#", "b") else best_name[:1]
    quality = best_name[len(root):]
    if quality in ("7", "m7", "maj7"):
        triad = root + ("m" if quality == "m7" else "")
        if triad in _CHORD_TEMPLATES:
            triad_score = float(np.dot(seg, _CHORD_TEMPLATES[triad]))
            if best_score - triad_score < 0.025:
                return triad
    return best_name


_SEGMENTS_PER_MEASURE = 2


def _measure_segments(time_sig_num: int) -> list[tuple[float, float]]:
    n = _SEGMENTS_PER_MEASURE
    return [(i / n, (i + 1) / n) for i in range(n)]


def _collapse_segments(seg_chords: list[tuple[float, str]]) -> list[tuple[float, str]]:
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
        measures.append(seg_chords)

    return measures


def _chords_from_fixed_bpm(
    chroma: np.ndarray,
    bpm: float,
    time_sig_num: int,
    sr: int,
) -> list[list[tuple[float, str]]]:
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
        measures.append(seg_chords)
        pos += measure_frames

    return measures
