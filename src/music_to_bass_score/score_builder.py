"""Build music21 Score from MIDI, metadata, analysis, and chord labels."""

from pathlib import Path
from typing import Optional

import music21
from music21 import (
    chord as m21chord,
    clef,
    expressions,
    harmony,
    key,
    metadata,
    meter,
    note as m21note,
    stream,
    tempo,
)

from .analyzer import AudioAnalysis
from .config import BASS_MIDI_MAX, BASS_MIDI_MIN, BASS_STRING_TUNINGS
from .downloader import SongMetadata
from .logger import get_logger
from .transcriber import NoteEvent

logger = get_logger(__name__)


def build_chord_chart(
    song_metadata: SongMetadata,
    analysis: AudioAnalysis,
    chord_labels: list,
    roman_labels: list,
    include_tab: bool = False,  # accepted for API symmetry; ignored (chart has no TAB)
) -> stream.Score:
    """Build a single-staff chord chart: chord symbols above, roman numerals inside.

    chord_labels and roman_labels are list[list[str]] (per-measure label lists, e.g.
    two entries per measure for 2-beat resolution). The score carries the labels as
    TextExpressions distinguished by placement: 'above' = chord symbol, 'below' = roman
    degree. pdf_exporter renders the actual single-staff chart layout.
    """
    logger.info(
        "Building chord chart: %r by %r, %d measures",
        song_metadata.title, song_metadata.artist, len(chord_labels),
    )
    score = stream.Score()

    md = metadata.Metadata()
    md.title = song_metadata.title
    md.composer = song_metadata.artist
    score.insert(0, md)

    beats_per_measure = analysis.time_signature_num

    part = stream.Part(id="chords")
    part.partName = "Chord Chart"
    part.append(clef.TrebleClef())

    root_name, mode = _parse_key_string(analysis.key)
    part.append(key.Key(root_name, mode))
    part.append(meter.TimeSignature(
        f"{analysis.time_signature_num}/{analysis.time_signature_den}"
    ))
    part.insert(0, tempo.MetronomeMark(number=analysis.bpm_rounded))

    n_measures = max(len(chord_labels), len(roman_labels), 1)
    for m_idx in range(n_measures):
        measure = stream.Measure(number=m_idx + 1)

        chords = _normalize_measure_labels(
            chord_labels[m_idx] if m_idx < len(chord_labels) else []
        )
        romans = _normalize_measure_labels(
            roman_labels[m_idx] if m_idx < len(roman_labels) else []
        )

        n = max(len(chords), 1)
        for j in range(n):
            off = beats_per_measure * j / n
            if j < len(chords):
                te = expressions.TextExpression(chords[j])
                te.placement = 'above'
                measure.insert(off, te)
            if j < len(romans):
                tr = expressions.TextExpression(romans[j])
                tr.placement = 'below'
                measure.insert(off, tr)

        # Invisible whole-measure rest keeps the measure rhythmically valid
        rest = m21note.Rest(quarterLength=beats_per_measure)
        rest.style.hideObjectOnPrint = True
        measure.append(rest)

        part.append(measure)

    score.append(part)
    return score


def build_score(
    song_metadata: SongMetadata,
    analysis: AudioAnalysis,
    note_events: list[NoteEvent],
    chord_labels: list,
    include_tab: bool = True,
    measure_grid: Optional[list[float]] = None,
) -> stream.Score:
    """Construct a music21 Score with bass clef staff and optional TAB.

    chord_labels may be either list[str] (one label per measure) or
    list[list[str]] (multiple labels per measure, placed at evenly spaced offsets).
    measure_grid, when provided, gives constant-tempo measure-start times used to
    map notes to measures (more stable than jittery beat tracking).
    """
    logger.info(
        "Building score: %r by %r, %d notes, %d chords, tab=%s",
        song_metadata.title, song_metadata.artist,
        len(note_events), len(chord_labels), include_tab,
    )
    score = stream.Score()

    md = metadata.Metadata()
    md.title = song_metadata.title
    md.composer = song_metadata.artist
    score.insert(0, md)

    beats_per_measure = analysis.time_signature_num
    seconds_per_beat = 60.0 / analysis.bpm
    grid = measure_grid or analysis.beat_times or []

    bass_part = _build_bass_part(
        note_events=note_events,
        chord_labels=chord_labels,
        analysis=analysis,
        beats_per_measure=beats_per_measure,
        seconds_per_beat=seconds_per_beat,
        measure_grid=grid,
    )
    score.append(bass_part)

    if include_tab:
        tab_part = _build_tab_part(
            note_events=note_events,
            beats_per_measure=beats_per_measure,
            seconds_per_beat=seconds_per_beat,
            analysis=analysis,
            measure_grid=grid,
        )
        score.append(tab_part)

    return score


def _normalize_measure_labels(entry) -> list[str]:
    """Coerce a per-measure chord entry into a list of label strings."""
    if entry is None:
        return []
    if isinstance(entry, str):
        return [entry]
    return [str(x) for x in entry]


def _build_bass_part(
    note_events: list[NoteEvent],
    chord_labels: list,
    analysis: AudioAnalysis,
    beats_per_measure: int,
    seconds_per_beat: float,
    measure_grid: list[float],
) -> stream.Part:
    part = stream.Part(id="bass")
    part.partName = "Bass Guitar"

    part.append(clef.BassClef())

    root_name, mode = _parse_key_string(analysis.key)
    part.append(key.Key(root_name, mode))

    time_sig = meter.TimeSignature(
        f"{analysis.time_signature_num}/{analysis.time_signature_den}"
    )
    part.append(time_sig)

    mm = tempo.MetronomeMark(number=analysis.bpm_rounded)
    part.insert(0, mm)

    notes_by_measure = _group_notes_by_measure(
        note_events, seconds_per_beat, beats_per_measure, measure_grid
    )

    n_measures = max(
        max(notes_by_measure.keys()) + 1 if notes_by_measure else 1,
        len(chord_labels),
    )

    for m_idx in range(n_measures):
        measure = stream.Measure(number=m_idx + 1)

        if m_idx < len(chord_labels):
            labels = _normalize_measure_labels(chord_labels[m_idx])
            n_lbl = len(labels)
            for j, label in enumerate(labels):
                try:
                    te = expressions.TextExpression(label)
                    te.style.fontStyle = 'bold'
                    te.placement = 'above'
                    # Evenly space labels across the measure (offset in quarter beats)
                    off = (beats_per_measure * j / n_lbl) if n_lbl > 0 else 0.0
                    measure.insert(off, te)
                except Exception:
                    pass

        measure_notes = notes_by_measure.get(m_idx, [])
        if measure_notes:
            for n in measure_notes:
                measure.append(n)
        else:
            rest = m21note.Rest(quarterLength=beats_per_measure)
            measure.append(rest)

        part.append(measure)

    return part


def _build_tab_part(
    note_events: list[NoteEvent],
    beats_per_measure: int,
    seconds_per_beat: float,
    analysis: AudioAnalysis,
    measure_grid: list[float],
) -> stream.Part:
    """Build a simplified TAB representation as a second part."""
    from music21 import tablature

    part = stream.Part(id="tab")
    part.partName = "Bass TAB"

    try:
        tab_clef = tablature.TabClef()
        part.append(tab_clef)
    except AttributeError:
        part.append(clef.TabClef())

    time_sig = meter.TimeSignature(
        f"{analysis.time_signature_num}/{analysis.time_signature_den}"
    )
    part.append(time_sig)

    notes_by_measure = _group_notes_by_measure(
        note_events, seconds_per_beat, beats_per_measure, measure_grid
    )

    n_measures = max(notes_by_measure.keys()) + 1 if notes_by_measure else 1

    for m_idx in range(n_measures):
        measure = stream.Measure(number=m_idx + 1)
        measure_notes = notes_by_measure.get(m_idx, [])

        if measure_notes:
            for n in measure_notes:
                string_num, fret_num = _midi_to_tab(n.pitch.midi)
                try:
                    tab_note = tablature.TabNote(
                        pitch=n.pitch,
                        fretNumber=fret_num,
                        stringNumber=string_num,
                        quarterLength=n.quarterLength,
                    )
                    measure.append(tab_note)
                except Exception:
                    measure.append(n)
        else:
            rest = m21note.Rest(quarterLength=beats_per_measure)
            measure.append(rest)

        part.append(measure)

    return part


def _group_notes_by_measure(
    note_events: list[NoteEvent],
    seconds_per_beat: float,
    beats_per_measure: int,
    measure_grid: Optional[list[float]] = None,
) -> dict[int, list[m21note.Note]]:
    """Convert NoteEvents to music21 Notes grouped by measure index.

    When measure_grid (constant-tempo measure-start times) is provided, notes are
    mapped to measures by searchsorted on those boundaries — stable and jitter-free.
    Falls back to fixed BPM arithmetic when no grid is available.
    """
    if measure_grid and len(measure_grid) >= 2:
        return _group_by_grid(note_events, measure_grid, seconds_per_beat)
    return _group_by_fixed_bpm(note_events, seconds_per_beat, beats_per_measure)


def _group_by_grid(
    note_events: list[NoteEvent],
    measure_grid: list[float],
    seconds_per_beat: float,
) -> dict[int, list[m21note.Note]]:
    import numpy as np
    from music21 import pitch as m21pitch

    measure_starts = np.array(measure_grid)
    groups: dict[int, list[m21note.Note]] = {}

    for event in note_events:
        midi_pitch = event.pitch
        if not (BASS_MIDI_MIN <= midi_pitch <= BASS_MIDI_MAX):
            continue

        m_idx = max(0, int(np.searchsorted(measure_starts, event.start_sec, side='right')) - 1)
        measure_start_t = float(measure_starts[m_idx])

        beat_in_measure = _quantize((event.start_sec - measure_start_t) / seconds_per_beat)
        beat_in_measure = max(0.0, beat_in_measure)
        dur_beats = _quantize(max(0.125, (event.end_sec - event.start_sec) / seconds_per_beat))

        n = m21note.Note()
        n.pitch = m21pitch.Pitch(midi=midi_pitch)
        n.quarterLength = dur_beats
        n.offset = beat_in_measure

        groups.setdefault(m_idx, []).append(n)

    return groups


def _group_by_fixed_bpm(
    note_events: list[NoteEvent],
    seconds_per_beat: float,
    beats_per_measure: int,
) -> dict[int, list[m21note.Note]]:
    from music21 import pitch as m21pitch

    groups: dict[int, list[m21note.Note]] = {}

    for event in note_events:
        midi_pitch = event.pitch
        if not (BASS_MIDI_MIN <= midi_pitch <= BASS_MIDI_MAX):
            continue

        start_beat = _quantize(event.start_sec / seconds_per_beat)
        dur_beats = _quantize(max(0.125, (event.end_sec - event.start_sec) / seconds_per_beat))

        measure_idx = int(start_beat // beats_per_measure)
        beat_in_measure = start_beat % beats_per_measure

        n = m21note.Note()
        n.pitch = m21pitch.Pitch(midi=midi_pitch)
        n.quarterLength = dur_beats
        n.offset = beat_in_measure

        groups.setdefault(measure_idx, []).append(n)

    return groups


def _quantize(value: float, grid: float = 0.25) -> float:
    """Snap value to the nearest grid position (16th note = 0.25 beats)."""
    return round(value / grid) * grid


def _parse_key_string(key_str: str) -> tuple[str, str]:
    """'A minor' -> ('A', 'minor'), 'C major' -> ('C', 'major')."""
    parts = key_str.strip().split()
    if len(parts) >= 2:
        return parts[0], parts[1].lower()
    return parts[0], "major"


def _midi_to_tab(midi_pitch: int) -> tuple[int, int]:
    """Convert MIDI pitch to (string_number, fret_number) for 4-string bass.

    String tunings (MIDI): E1=28, A1=33, D2=38, G2=43
    String numbers: 1=G (highest), 2=D, 3=A, 4=E (lowest)
    """
    string_midi = list(reversed(BASS_STRING_TUNINGS))
    string_numbers = [1, 2, 3, 4]

    best_string = 4
    best_fret = midi_pitch - BASS_STRING_TUNINGS[0]
    best_score = float("inf")

    for s_num, open_midi in zip(string_numbers, string_midi):
        fret = midi_pitch - open_midi
        if 0 <= fret <= 24:
            score = fret
            if score < best_score:
                best_score = score
                best_string = s_num
                best_fret = fret

    return best_string, max(0, best_fret)
