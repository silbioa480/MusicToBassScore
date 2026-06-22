"""Per-measure chord detection: BTC Transformer + inversion-slash annotation.

Primary engine: BTC large-voca Transformer on the full mix.

Harmonic rhythm: adaptive — 1 chord per measure by default; 2 chords only when
  BTC shows a genuine mid-measure change (second chord covers ≥40% of the
  measure AND represents >50% of the underlying BTC segment's duration).

Post-processing — inversion slash:
  Independently detect the dominant bass note from a C2-C4 chroma (<262 Hz).
  Slash notation is appended ONLY when ALL of the following hold:
  (a) bass confidence > 0.88 (suppresses noise/blips),
  (b) bass note differs from the chord root,
  (c) bass note IS a chord tone (3rd, 5th, 7th …) — e.g. G/B (B∈G major ✓)
      but NOT Am/G (G∉Am ✗). This prevents spurious "wrong root" slashes.

BTC limitation: bass-guitar fundamentals (e.g. D2) contaminate the full-mix
  CQT, causing BTC to occasionally label A/Am as D in heavily bass-driven
  sections. This cannot be resolved without source separation (Demucs times
  out on CPU).

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
    bass_stem_path: Optional[Path] = None,
    btc_input_path: Optional[Path] = None,
) -> list[list[tuple[float, str]]]:
    """Detect chord changes per measure.

    Returns a list (one entry per measure) of (beat_offset, chord_symbol) tuples,
    with consecutive identical chords collapsed to change points.

    Primary engine: the pretrained BTC Transformer (large vocabulary). By default
    runs on the FULL mix (harmony needed for chord quality). When `btc_input_path`
    is provided (e.g. a harmonic submix without drums), BTC runs on that file instead
    — this reduces bass-fundamental contamination that causes D/Dm mis-labels.

    Falls back to the librosa chroma matcher if BTC is unavailable or errors.

    `bass_stem_path`, when provided (a Demucs-separated bass stem), supplies a clean
    bass line for accurate inversion-slash notation. Without it, a (noisier) full-mix
    low-register chroma is used.
    """
    if measure_grid and len(measure_grid) >= 1:
        try:
            from . import btc_chord
            if btc_chord.is_available():
                return _chords_from_btc(
                    audio_path, measure_grid, time_sig_num, bpm, bass_stem_path,
                    btc_input_path=btc_input_path,
                )
            logger.warning("BTC model not found; using chroma fallback")
        except Exception as exc:
            logger.warning("BTC chord recognition failed (%s); using chroma fallback", exc)

    return _detect_chords_chroma(audio_path, bpm, time_sig_num, measure_grid, sample_rate)


# ---------------------------------------------------------------------------
# BTC path
# ---------------------------------------------------------------------------

# BTC boundary must land in the inner 50% of the measure (25%–75%) to be
# treated as a genuine mid-measure chord change. Replaces the old magic fractions.
_MID_MEASURE_BAND = 0.25

# Softmax confidence below this threshold → append "?" to the chord symbol so
# users know to verify by ear. 170-class softmax: uniform ≈ 0.006; confident ≥ 0.35.
_LOW_CONF_THRESHOLD = 0.35

# Snap measure boundaries to the nearest BTC segment boundary within ±N beats.
_BOUNDARY_SNAP_BEATS = 0.5


def _snap_to_btc_boundary(timeline: list, t: float, beat_dur: float) -> float:
    """Return `t` snapped to the nearest BTC segment-start within ±_BOUNDARY_SNAP_BEATS beats."""
    tol = beat_dur * _BOUNDARY_SNAP_BEATS
    best_t, best_dist = t, float("inf")
    for seg in timeline:
        boundary = seg[0]
        dist = abs(boundary - t)
        if dist < tol and dist < best_dist:
            best_dist, best_t = dist, boundary
    return best_t


def _chords_from_btc(
    audio_path: Path,
    measure_grid: list[float],
    time_sig_num: int,
    bpm: float,
    bass_stem_path: Optional[Path] = None,
    btc_input_path: Optional[Path] = None,
) -> list[list[tuple[float, str]]]:
    """Map the BTC chord timeline adaptively, then add inversion-slash notation.

    BTC runs on `btc_input_path` when provided (harmonic submix without drums reduces
    bass-fundamental contamination), otherwise on the full mix `audio_path`.

    Beat-snap: each measure boundary is snapped to the nearest BTC segment boundary
    within ±0.5 beats — this compensates for constant-tempo grid drift.

    Mid-measure detection: 2 chords per measure only when a BTC boundary falls in the
    inner 50% of the measure (replacing fragile overlap-fraction heuristics).

    Confidence: chord symbols get a "?" suffix when the model's max softmax probability
    is below _LOW_CONF_THRESHOLD — signals uncertain regions for human review.
    """
    from . import btc_chord

    run_path = btc_input_path if (btc_input_path and Path(btc_input_path).is_file()) else audio_path
    logger.info(
        "Detecting chords (BTC large-voca): input=%s bpm=%.1f time_sig=%d/4 grid=%d measures bass=%s",
        run_path.name, bpm, time_sig_num, len(measure_grid),
        "clean" if bass_stem_path else "full-mix",
    )
    timeline = btc_chord.recognize_chords(run_path)
    beat_dur = 60.0 / bpm
    spm = beat_dur * time_sig_num  # seconds per measure

    # Bass chroma source: prefer clean bass stem; else full-mix low-register band.
    if bass_stem_path is not None and Path(bass_stem_path).is_file():
        y, sr = librosa.load(str(bass_stem_path), sr=SAMPLE_RATE, mono=True)
        bass_ch = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("C1"), n_octaves=3,
        )
    else:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        y = librosa.effects.harmonic(y, margin=4.0)
        bass_ch = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("C2"), n_octaves=2,
        )
    times = librosa.frames_to_time(
        np.arange(bass_ch.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )

    measures: list[list[tuple[float, str]]] = []
    for m_start_grid in measure_grid:
        # Snap measure start to nearest BTC boundary to correct for tempo-grid drift.
        m_start = _snap_to_btc_boundary(timeline, m_start_grid, beat_dur)
        m_end = m_start + spm
        mid = m_start + spm / 2.0

        dom, dom_conf = btc_chord.chord_at_window(timeline, m_start, m_end)
        c1, c1_conf = btc_chord.chord_at_window(timeline, m_start, mid)
        c2, c2_conf = btc_chord.chord_at_window(timeline, mid, m_end)

        if c1 != c2:
            # Genuine mid-measure change: a BTC segment boundary must fall in the
            # inner [25%, 75%] of the measure — replaces fragile overlap fractions.
            inner_lo = m_start + _MID_MEASURE_BAND * spm
            inner_hi = m_end - _MID_MEASURE_BAND * spm
            genuine = any(
                inner_lo <= seg[0] <= inner_hi or inner_lo <= seg[1] <= inner_hi
                for seg in timeline
                if max(m_start, seg[0]) < min(m_end, seg[1])
            )
        else:
            genuine = False

        if genuine:
            beat_half = float(time_sig_num) / 2.0
            raw: list[tuple[float, str, float]] = [(0.0, c1, c1_conf), (beat_half, c2, c2_conf)]
        else:
            raw = [(0.0, dom, dom_conf)]

        # Apply inversion-slash annotation + low-confidence "?" marker
        seg_list = []
        for beat_off, chord, conf in raw:
            t0 = m_start + (beat_off / time_sig_num) * spm
            t1 = m_start + ((beat_off / time_sig_num) + 0.5) * spm
            chord = _apply_slash_bass(chord, bass_ch, times, t0, t1)
            if conf < _LOW_CONF_THRESHOLD and chord not in ("N.C.", "NC"):
                chord = chord + "?"
            seg_list.append((beat_off, chord))
        measures.append(seg_list)

    measures = _smooth_chord_sequence(measures)
    preview = ["|".join(s for _, s in m) for m in measures[:6]]
    logger.info("BTC chord mapping complete: %d measures — %s", len(measures), preview)
    return measures


def _segment_coverage_in_window(
    timeline: list, sym: str, t0: float, t1: float
) -> float:
    """Total seconds that BTC segments with label `sym` overlap [t0, t1)."""
    total = 0.0
    for seg in timeline:
        s, e, c = seg[0], seg[1], seg[2]
        if c == sym:
            total += max(0.0, min(t1, e) - max(t0, s))
    return total


def _segment_total_duration(
    timeline: list, sym: str, t0: float, t1: float
) -> float:
    """Duration of the BTC segment for `sym` that overlaps [t0, t1) the most."""
    best_overlap, best_dur = 0.0, 0.0
    for seg in timeline:
        s, e, c = seg[0], seg[1], seg[2]
        if c == sym:
            overlap = max(0.0, min(t1, e) - max(t0, s))
            if overlap > best_overlap:
                best_overlap, best_dur = overlap, e - s
    return best_dur


def _chord_root(symbol: str) -> str:
    """Extract root note name from a chord symbol (handles '#'/'b' accidentals)."""
    if not symbol or symbol in ("N.C.", "NC"):
        return ""
    if len(symbol) > 1 and symbol[1] in ("#", "b"):
        return symbol[:2]
    return symbol[:1]


def _chord_quality(symbol: str) -> str:
    """Extract the quality suffix after the root (e.g. 'Cm7' → 'm7', 'G' → '')."""
    root = _chord_root(symbol)
    return symbol[len(root):]


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


# Chord tone intervals (semitones above root) for each quality suffix.
_CHORD_TONE_INTERVALS: dict[str, set[int]] = {
    "":      {0, 4, 7},
    "m":     {0, 3, 7},
    "7":     {0, 4, 7, 10},
    "m7":    {0, 3, 7, 10},
    "maj7":  {0, 4, 7, 11},
    "dim":   {0, 3, 6},
    "dim7":  {0, 3, 6, 9},
    "m7b5":  {0, 3, 6, 10},
    "+":     {0, 4, 8},
    "sus4":  {0, 5, 7},
    "sus2":  {0, 2, 7},
    "6":     {0, 4, 7, 9},
    "m6":    {0, 3, 7, 9},
    "mM7":   {0, 3, 7, 11},
}

# Bass confidence must exceed this threshold AND bass must be a chord tone.
_BASS_CHORD_TONE_CONFIDENCE = 0.88


def _apply_slash_bass(
    chord: str, bass_ch: np.ndarray, times: np.ndarray, t0: float, t1: float
) -> str:
    """Append slash-bass only for clear inversions (bass is a chord tone, high confidence).

    Conditions (all must hold):
      1. bass confidence > _BASS_CHORD_TONE_CONFIDENCE  (blip suppression)
      2. bass note ≠ chord root
      3. bass note IS in the chord's tone set (3rd, 5th, 7th …)
    """
    if chord in ("N.C.", "NC"):
        return chord

    upper = chord.split("/")[0]
    chord_root = _chord_root(upper)
    if not chord_root:
        return chord

    bass_vec = _chroma_window(bass_ch, times, t0, t1)
    bass_idx = int(np.argmax(bass_vec))
    bass_conf = float(bass_vec[bass_idx])
    bass_note = _NOTE_NAMES[bass_idx]

    if bass_conf < _BASS_CHORD_TONE_CONFIDENCE:
        return chord
    if bass_note == chord_root:
        return chord

    # Check bass note is a chord tone
    quality = _chord_quality(upper)
    intervals = _CHORD_TONE_INTERVALS.get(quality)
    if intervals is None:
        return chord  # unknown quality — skip slash

    root_semi = _NOTE_SEMITONES.get(chord_root)
    bass_semi = _NOTE_SEMITONES.get(bass_note)
    if root_semi is None or bass_semi is None:
        return chord

    if (bass_semi - root_semi) % 12 not in intervals:
        return chord  # not a chord tone

    return f"{upper}/{bass_note}"


_NOTE_SEMITONES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


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
