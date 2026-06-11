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


def _chords_from_btc(
    audio_path: Path,
    measure_grid: list[float],
    time_sig_num: int,
    bpm: float,
) -> list[list[tuple[float, str]]]:
    """Map the BTC chord timeline onto half-measure windows of the constant-tempo grid."""
    from . import btc_chord

    logger.info(
        "Detecting chords (BTC large-voca): %s (bpm=%.1f time_sig=%d/4 grid=%d measures)",
        audio_path, bpm, time_sig_num, len(measure_grid),
    )
    timeline = btc_chord.recognize_chords(audio_path)
    seconds_per_measure = (60.0 / bpm) * time_sig_num
    segs = _measure_segments(time_sig_num)

    measures: list[list[tuple[float, str]]] = []
    for m_start in measure_grid:
        seg_chords = []
        for s_frac, e_frac in segs:
            ws = m_start + s_frac * seconds_per_measure
            we = m_start + e_frac * seconds_per_measure
            beat_offset = s_frac * time_sig_num
            seg_chords.append((beat_offset, btc_chord.chord_at_window(timeline, ws, we)))
        measures.append(seg_chords)

    measures = _smooth_chord_sequence(measures)
    preview = ["|".join(s for _, s in m) for m in measures[:6]]
    logger.info("BTC chord mapping complete: %d measures — %s", len(measures), preview)
    return measures


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
    """Compute a clean chroma: HPSS harmonic component + tuning correction + CENS.

    - HPSS removes percussive/transient energy (drums) that pollutes pitch content.
    - Tuning estimation corrects for songs not at exact A440.
    - chroma_cens applies L1-norm → amplitude quantisation (6 levels) → window average,
      making it more robust to dynamics and timbre than raw chroma_cqt.
    - win_len_smooth=9 (~0.21 s at 512/22050) preserves beat-level resolution while
      still suppressing frame-level jitter.
    """
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
    """Remove isolated single-segment outliers, then collapse identical neighbors.

    Input is the raw uncollapsed per-segment sequence (offsets preserved). A segment
    whose chord differs from BOTH neighbors, where those neighbors agree, is treated as
    a one-off blip and replaced. Sustained changes (2+ consecutive segments) are kept.
    Finally each measure's consecutive identical segments collapse to change points.
    """
    # Flatten to a single chord stream, remembering each segment's (measure, offset)
    flat: list[str] = []
    index: list[tuple[int, float]] = []
    for m_idx, segs in enumerate(measures):
        for off, chord in segs:
            flat.append(chord)
            index.append((m_idx, off))

    # Mode filter: fix a lone outlier surrounded by an agreeing pair
    fixed = flat[:]
    for i in range(1, len(flat) - 1):
        if flat[i] != flat[i - 1] and flat[i - 1] == flat[i + 1]:
            fixed[i] = flat[i - 1]

    # Scatter back into per-measure segment lists, then collapse identical neighbors
    rebuilt: list[list[tuple[float, str]]] = [[] for _ in measures]
    for (m_idx, off), chord in zip(index, fixed):
        rebuilt[m_idx].append((off, chord))

    return [_collapse_segments(segs) if segs else [(0.0, "N.C.")] for segs in rebuilt]


def _match_chord(chroma: np.ndarray, start_frame: int, end_frame: int) -> str:
    """Template-match the mean chroma over a frame range to the nearest chord symbol.

    After finding the best match, applies a simplicity bias: if the same-root triad
    scores within 0.025 of the best 7th/maj7 chord, prefer the simpler triad.
    7th tones often appear from ringing strings or adjacent voices rather than the
    chord itself, so this reduces systematic over-detection of seventh chords.
    """
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

    # Simplicity bias: prefer triad over 7th when margin is within threshold
    root = best_name[:2] if len(best_name) > 1 and best_name[1] in ("#", "b") else best_name[:1]
    quality = best_name[len(root):]
    if quality in ("7", "m7", "maj7"):
        triad = root + ("m" if quality == "m7" else "")
        if triad in _CHORD_TEMPLATES:
            triad_score = float(np.dot(seg, _CHORD_TEMPLATES[triad]))
            if best_score - triad_score < 0.025:
                return triad
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
    """Half-measure detection on the constant-tempo grid (raw, uncollapsed segments)."""
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
    """Fallback: fixed-BPM half-measure segmentation from frame 0 (raw, uncollapsed)."""
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
