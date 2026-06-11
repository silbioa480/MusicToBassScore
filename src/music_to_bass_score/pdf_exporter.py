"""PDF export — writes LilyPond source directly for proper bass-score layout."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from music21 import stream

from .config import LILYPOND_BIN, LILYPOND_TIMEOUT_SEC, SCORES_DIR
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExportResult:
    pdf_path: Path
    lily_path: Optional[Path] = None


def check_lilypond_available() -> bool:
    """Return True if LilyPond binary is accessible."""
    try:
        result = subprocess.run(
            [LILYPOND_BIN, "--version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def export_to_pdf(
    score: stream.Score,
    output_dir: Path = SCORES_DIR,
    filename_stem: str = "bass_score",
    method: Literal["lilypond", "musicxml"] = "lilypond",
) -> ExportResult:
    """Render a music21 Score to PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lily_available = check_lilypond_available()
    logger.info(
        "Exporting PDF: method=%s lilypond_available=%s stem=%s",
        method, lily_available, filename_stem,
    )

    if method == "lilypond" and lily_available:
        return _export_via_lilypond(score, output_dir, filename_stem)
    else:
        if method == "lilypond" and not lily_available:
            logger.warning("LilyPond not found — falling back to MusicXML")
        return _export_via_musicxml(score, output_dir, filename_stem)


# ── Pitch / duration helpers ──────────────────────────────────────────────────

# LilyPond English-language pitch names (chromatic from C)
_PITCH_NAMES = ['c', 'cs', 'd', 'ef', 'e', 'f', 'fs', 'g', 'af', 'a', 'bf', 'b']

# Quarter-length → LilyPond duration string
_QL_MAP: dict[float, str] = {
    4.0: '1', 3.0: '2.', 2.0: '2', 1.5: '4.', 1.0: '4',
    0.75: '8.', 0.5: '8', 0.375: '16.', 0.25: '16', 0.125: '32',
}
_QL_SORTED = sorted(_QL_MAP.keys(), reverse=True)


def _midi_to_lily(midi: int) -> str:
    """Convert a MIDI note number to a LilyPond pitch string (English notation)."""
    name = _PITCH_NAMES[midi % 12]
    oct_num = midi // 12 - 1      # MIDI octave (C4=60 → oct 4)
    lily_oct = oct_num - 3        # LilyPond 'c' (no marker) = MIDI 48 = oct 3
    if lily_oct > 0:
        return name + "'" * lily_oct
    if lily_oct < 0:
        return name + "," * (-lily_oct)
    return name


def _split_ql(ql: float) -> list[str]:
    """Decompose a quarter-length into a list of LilyPond duration strings."""
    result: list[str] = []
    rem = round(ql * 32) / 32
    for _ in range(8):
        if rem < 0.09:
            break
        for d in _QL_SORTED:
            if d <= rem + 0.02:
                result.append(_QL_MAP[d])
                rem = round((rem - d) * 32) / 32
                break
    return result or ['4']


def _note_ly(midi: int, ql: float, markup: str = '') -> str:
    """LilyPond note token(s) — ties multiple durations if needed."""
    pitch = _midi_to_lily(midi)
    durs = _split_ql(ql)
    parts = []
    for i, dur in enumerate(durs):
        s = f"{pitch}{dur}"
        if i == 0 and markup:
            s += markup
        if i < len(durs) - 1:
            s += '~'
        parts.append(s)
    return ' '.join(parts)


def _rest_ly(ql: float, markup: str = '') -> str:
    """LilyPond rest token(s)."""
    durs = _split_ql(ql)
    parts = []
    for i, dur in enumerate(durs):
        s = f"r{dur}"
        if i == 0 and markup:
            s += markup
        parts.append(s)
    return ' '.join(parts)


def _esc(s: str) -> str:
    return s.replace('\\', '\\\\').replace('"', "'")


# ── Measure renderer ──────────────────────────────────────────────────────────

# Round to nearest 32nd note (0.125 quarter beats)
def _q(x: float) -> float:
    return round(x * 8) / 8


def _markup_for(label: str) -> str:
    return f'^\\markup{{\\bold "{_esc(label)}"}}'


def _measure_to_ly(measure, beats: int, chords: Optional[list]) -> str:
    """Render a music21 Measure as a LilyPond measure string of EXACTLY `beats` quarter beats.

    `chords` is a list of (offset_in_quarter_beats, label) tuples; each label's markup is
    attached to the first token at or after its offset, so multiple chords per measure land
    at their correct beat positions. Uses 32nd-note resolution so the bar check | never overflows.
    """
    from music21 import note as m21note

    # Pending chords sorted by offset; placed greedily as the cursor advances.
    pending = sorted(((_q(o), lbl) for o, lbl in (chords or []) if lbl), key=lambda x: x[0])

    def _take_markup(cur_pos: float) -> str:
        """Return markup for any chord whose offset has been reached, else ''."""
        mk = ''
        while pending and pending[0][0] <= cur_pos + 0.01:
            _, lbl = pending.pop(0)
            if not mk:  # one markup per token; collapse same-position duplicates
                mk = _markup_for(lbl)
        return mk

    tokens: list[str] = []
    cur = 0.0          # current time cursor in quarter beats (32nd-note grid)
    beats_f = float(beats)

    elems = sorted(measure.notesAndRests, key=lambda e: float(e.offset))

    for elem in elems:
        off = _q(float(elem.offset))

        # Note starts at or past the end of the measure — stop
        if off >= beats_f - 0.01:
            break

        # Note starts before cursor (overlap) — skip
        if off < cur - 0.01:
            continue

        # Fill gap before this element with a rest
        gap = _q(off - cur)
        if gap >= 0.125:
            tokens.append(_rest_ly(gap, _take_markup(cur)))
            cur = _q(cur + gap)

        # Snap cursor to the note's quantised start
        cur = max(cur, off)

        # Available space from cursor to end of measure
        avail = _q(beats_f - cur)
        if avail < 0.125:
            break  # Less than a 32nd note left — measure is full

        # Clip note duration to available space (NO max(0.125, ...) — that causes overflow)
        ql = _q(min(float(elem.quarterLength), avail))
        if ql < 0.125:
            break  # Duration rounds to zero — stop

        mk = _take_markup(cur)
        if isinstance(elem, m21note.Note):
            tokens.append(_note_ly(elem.pitch.midi, ql, mk))
        else:
            tokens.append(_rest_ly(ql, mk))

        cur = _q(cur + ql)

        if cur >= beats_f - 0.01:
            break  # Measure exactly full

    # Fill remaining space to reach EXACTLY `beats` quarter beats
    tail = _q(beats_f - cur)
    if tail >= 0.125:
        tokens.append(_rest_ly(tail, _take_markup(cur)))

    if not tokens:
        tokens = [_rest_ly(float(beats), _take_markup(0.0))]
    elif pending:
        # Chord offsets past all tokens — attach the next one to the last token if free.
        if '^\\markup' not in tokens[-1]:
            tokens[-1] = tokens[-1] + _markup_for(pending[0][1])

    return ' '.join(tokens) + ' |'


# ── LilyPond source builder ───────────────────────────────────────────────────

def _score_to_ly(score: stream.Score, stem: str) -> str:
    """Build a complete LilyPond source string from a music21 Score."""
    bass_part = next((p for p in score.parts if p.id == "bass"), score.parts[0])
    has_tab = any(p.id == "tab" for p in score.parts)

    # ── Header ──────────────────────────────────────────────────────────────
    md = score.metadata
    title    = _esc((md.title    if md and md.title    else stem) or stem)
    composer = _esc((md.composer if md and md.composer else "")   or "")

    # ── Key / time / tempo ───────────────────────────────────────────────────
    flat = bass_part.flatten()

    from music21 import key as m21key, tempo as m21tempo
    key_objs = list(flat.getElementsByClass(m21key.Key))
    if not key_objs:
        key_objs = list(flat.getElementsByClass(m21key.KeySignature))
    if key_objs:
        ks = key_objs[0]
        tonic = ks.tonic.name.replace('#', 's').replace('-', 'f').lower()
        mode  = getattr(ks, 'mode', 'major') or 'major'
        key_lily = f"\\key {tonic} \\{mode}"
    else:
        key_lily = "\\key c \\major"

    time_sigs = list(flat.getElementsByClass('TimeSignature'))
    if time_sigs:
        ts = time_sigs[0]
        beats = ts.numerator
        time_lily = f"\\time {ts.numerator}/{ts.denominator}"
    else:
        beats, time_lily = 4, "\\time 4/4"

    mm_marks = list(flat.getElementsByClass(m21tempo.MetronomeMark))
    bpm = int(round(mm_marks[0].number)) if mm_marks else 120
    tempo_lily = f"\\tempo 4 = {bpm}"

    # ── Notes: bass (with chord markup) & tab (bare) ─────────────────────────
    MEASURES_PER_LINE = 4

    bass_lines: list[str] = []
    tab_lines:  list[str] = []

    for i, measure in enumerate(bass_part.getElementsByClass('Measure')):
        chords = [
            (_q(float(te.offset)), te.content)
            for te in measure.getElementsByClass('TextExpression')
        ]

        bass_lines.append(f"  {_measure_to_ly(measure, beats, chords)}  % m{i+1}")
        tab_lines.append(f"  {_measure_to_ly(measure, beats, None)}")

        # Force a line break every MEASURES_PER_LINE measures
        if (i + 1) % MEASURES_PER_LINE == 0:
            bass_lines.append("  \\break")
            tab_lines.append("  \\break")

    bass_block = "\n".join(bass_lines)
    tab_block  = "\n".join(tab_lines)

    # ── TAB staff section ─────────────────────────────────────────────────────
    tab_staff_section = ""
    if has_tab:
        tab_staff_section = """
    \\new TabStaff \\with {
      \\remove "Key_engraver"
      stringTunings = \\stringTuning <e,, a,, d, g,>
    } {
      \\global
      \\tabNotes
    }"""

    return f'''\\version "2.24.0"
\\language "english"

\\header {{
  title = "{title}"
  composer = "{composer}"
  tagline = ##f
}}

\\paper {{
  #(set-paper-size "a4")
  indent = 15\\mm
  short-indent = 0\\mm
  system-system-distance = #\'((basic-distance . 14) (minimum-distance . 10) (stretchability . 6))
  ragged-last-bottom = ##f
  ragged-bottom = ##f
  ragged-last = ##t
}}

global = {{
  {key_lily}
  {time_lily}
}}

bassNotes = {{
  \\clef bass
  {tempo_lily}
{bass_block}
}}

tabNotes = {{
{tab_block}
}}

\\score {{
  \\new StaffGroup <<
    \\new Staff {{
      \\global
      \\bassNotes
    }}{tab_staff_section}
  >>
  \\layout {{
    \\context {{
      \\Score
      proportionalNotationDuration = #(ly:make-moment 1/8)
      \\override SpacingSpanner.uniform-stretching = ##t
    }}
  }}
}}
'''


# ── Chord-chart builder (single staff: chord above, roman degree large inside) ──

def _spacer_ly(ql: float) -> str:
    """LilyPond invisible spacer token(s) for a duration of `ql` quarter beats."""
    return ' '.join(f"s{d}" for d in _split_ql(ql))


def _chart_measure_ly(measure, beats: int) -> str:
    """Render one chord-chart measure: invisible spacers carrying chord/roman markups."""
    # Collect markups by offset: 'above' = chord symbol, 'below' = roman degree
    above: dict[float, str] = {}
    below: dict[float, str] = {}
    for te in measure.getElementsByClass('TextExpression'):
        off = _q(float(te.offset))
        place = getattr(te, 'placement', 'above') or 'above'
        (below if place == 'below' else above)[off] = te.content

    # Segment boundaries = sorted unique offsets where a chord starts
    starts = sorted(set(above.keys()) | {0.0})
    beats_f = float(beats)
    tokens: list[str] = []

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else beats_f
        dur = _q(end - start)
        if dur < 0.125:
            continue
        spacers = _split_ql(dur)
        # Attach markups to the first spacer of the segment
        chord = above.get(start)
        roman = below.get(start)
        for j, d in enumerate(spacers):
            tok = f"s{d}"
            if j == 0:
                if chord:
                    tok += f'^\\markup{{\\bold "{_esc(chord)}"}}'
                if roman:
                    tok += f'_\\markup{{\\large \\bold "{_esc(roman)}"}}'
            tokens.append(tok)

    if not tokens:
        tokens = [_spacer_ly(beats_f)]
    return ' '.join(tokens) + ' |'


def _chart_to_ly(score: stream.Score, stem: str) -> str:
    """Build a single-staff chord-chart LilyPond source from a music21 Score."""
    part = next((p for p in score.parts if p.id == "chords"), score.parts[0])

    md = score.metadata
    title    = _esc((md.title    if md and md.title    else stem) or stem)
    composer = _esc((md.composer if md and md.composer else "")   or "")

    flat = part.flatten()
    from music21 import key as m21key, tempo as m21tempo

    key_objs = list(flat.getElementsByClass(m21key.Key)) or list(flat.getElementsByClass(m21key.KeySignature))
    if key_objs:
        ks = key_objs[0]
        tonic = ks.tonic.name.replace('#', 's').replace('-', 'f').lower()
        mode  = getattr(ks, 'mode', 'major') or 'major'
        key_lily = f"\\key {tonic} \\{mode}"
        key_name = f"{ks.tonic.name} {mode}"
    else:
        key_lily, key_name = "\\key c \\major", "C major"

    time_sigs = list(flat.getElementsByClass('TimeSignature'))
    if time_sigs:
        ts = time_sigs[0]
        beats = ts.numerator
        time_lily = f"\\time {ts.numerator}/{ts.denominator}"
    else:
        beats, time_lily = 4, "\\time 4/4"

    mm = list(flat.getElementsByClass(m21tempo.MetronomeMark))
    bpm = int(round(mm[0].number)) if mm else 120

    MEASURES_PER_LINE = 4
    lines: list[str] = []
    for i, measure in enumerate(part.getElementsByClass('Measure')):
        lines.append(f"  {_chart_measure_ly(measure, beats)}  % m{i+1}")
        if (i + 1) % MEASURES_PER_LINE == 0:
            lines.append("  \\break")
    chart_block = "\n".join(lines)

    subtitle = _esc(f"Key: {key_name}   |   {beats}/4   |   ♩ = {bpm}")

    return f'''\\version "2.24.0"
\\language "english"

\\header {{
  title = "{title}"
  composer = "{composer}"
  subtitle = "{subtitle}"
  tagline = ##f
}}

\\paper {{
  #(set-paper-size "a4")
  indent = 0\\mm
  ragged-last = ##t
  ragged-last-bottom = ##f
  system-system-distance = #'((basic-distance . 18) (minimum-distance . 14) (stretchability . 8))
}}

global = {{
  {key_lily}
  {time_lily}
}}

chartNotes = {{
  \\clef treble
  \\tempo 4 = {bpm}
{chart_block}
}}

\\score {{
  \\new Staff \\with {{
    \\override TimeSignature.stencil = ##f
  }} {{
    \\global
    \\override Staff.Clef.stencil = ##f
    \\chartNotes
  }}
  \\layout {{
    \\context {{
      \\Score
      \\override SpacingSpanner.uniform-stretching = ##t
      proportionalNotationDuration = #(ly:make-moment 1/8)
    }}
  }}
}}
'''


# ── Export functions ──────────────────────────────────────────────────────────

def _export_via_lilypond(
    score: stream.Score,
    output_dir: Path,
    filename_stem: str,
) -> ExportResult:
    ly_path  = output_dir / f"{filename_stem}.ly"
    pdf_path = output_dir / f"{filename_stem}.pdf"

    is_chart = any(p.id == "chords" for p in score.parts)
    ly_content = _chart_to_ly(score, filename_stem) if is_chart else _score_to_ly(score, filename_stem)
    ly_path.write_text(ly_content, encoding='utf-8')
    logger.debug("LilyPond source written: %s (%d bytes)", ly_path, len(ly_content))

    result = subprocess.run(
        [LILYPOND_BIN, "-o", str(output_dir / filename_stem), str(ly_path)],
        capture_output=True,
        text=True,
        timeout=LILYPOND_TIMEOUT_SEC,
    )

    if result.returncode != 0:
        logger.error("LilyPond failed (exit %d):\n%s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(
            f"LilyPond failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )

    if not pdf_path.exists():
        matches = list(output_dir.glob(f"{filename_stem}*.pdf"))
        if matches:
            pdf_path = matches[0]
        else:
            raise FileNotFoundError(
                f"LilyPond ran successfully but no PDF found in {output_dir}"
            )

    logger.info(
        "PDF exported via LilyPond: %s (%dKB)", pdf_path, pdf_path.stat().st_size // 1024
    )
    return ExportResult(pdf_path=pdf_path, lily_path=ly_path)


def _export_via_musicxml(
    score: stream.Score,
    output_dir: Path,
    filename_stem: str,
) -> ExportResult:
    """Fallback: export as MusicXML."""
    xml_path = output_dir / f"{filename_stem}.musicxml"
    score.write("musicxml", fp=str(xml_path))

    pdf_path = output_dir / f"{filename_stem}.pdf"
    try:
        score.write("musicxml.pdf", fp=str(pdf_path))
        if pdf_path.exists():
            return ExportResult(pdf_path=pdf_path, lily_path=None)
    except Exception:
        pass

    return ExportResult(pdf_path=xml_path, lily_path=None)
