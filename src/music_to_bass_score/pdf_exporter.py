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


# ── Chord-chart builder (banded row box: one box per row, measures split by bars) ──

_MEASURES_PER_LINE = 4
# 24 staff-spaces per cell. Combined with set-global-staff-size 18 and 8 mm side
# margins, 4 cells + inner dividers fill the page width with minimal side whitespace
# while still fitting long two-degree cells.
_CELL_WIDTH = 24
# Built-in whitespace margin (staff-spaces) baked around each row so adjacent rows
# always have a wide, uniform vertical gap regardless of LilyPond system spacing.
_ROW_PAD = 2.2


def _combine_markups(parts: list[str]) -> str:
    """Overlay several markups at a shared origin via nested binary \\combine."""
    if len(parts) == 1:
        return parts[0]
    return '\\combine ' + parts[0] + ' ' + _combine_markups(parts[1:])


def _positioned_cell(items: list[tuple[float, str]], beats: int) -> str:
    """Lay chords/degrees inside a fixed-width cell at their beat positions.

    A single chord is centred. With multiple chords each is left-anchored at its beat
    fraction (offset/beats × cell width) — so the first chord starts at the cell's left
    edge and a second chord at beat (beats/2) begins at the cell's centre. `items` are
    already-wrapped markup strings (e.g. '"Em7"' or '\\bold "vi7"').
    """
    w = _CELL_WIDTH
    if not items:
        return f'\\hcenter-in #{w} \\transparent "x"'
    if len(items) == 1:
        return f'\\hcenter-in #{w} {items[0][1]}'
    # Strut sets the cell width; each item is translated to its beat-fraction x.
    parts = [f'\\hspace #{w}']
    for off, markup in items:
        x = max(0.0, min(w, (off / beats) * w))
        parts.append(f"\\translate #'({x:.2f} . 0) {markup}")
    return _combine_markups(parts)


def _measure_cell_parts(measure, beats: int) -> tuple[str, str]:
    """Return (chord_markup, degree_markup) for one measure, each a fixed-width cell.

    Repeated chords are already collapsed upstream, so each measure carries only its
    actual chord changes (1..N), placed at their beat positions within the cell.
    """
    above: list[tuple[float, str]] = []
    below: dict[float, str] = {}
    for te in measure.getElementsByClass('TextExpression'):
        off = _q(float(te.offset))
        place = getattr(te, 'placement', 'above') or 'above'
        if place == 'below':
            below[off] = te.content
        else:
            above.append((off, te.content))
    above.sort(key=lambda x: x[0])

    chord_items = [(off, f'"{_esc(sym)}"') for off, sym in above]
    # \large \bold per item (NOT around the whole cell, which would scale the width strut
    # and break alignment with the chord row above).
    deg_items = [(off, f'\\large \\bold "{_esc(below.get(off, ""))}"') for off, _ in above]

    chord_cell = _positioned_cell(chord_items, beats)
    deg_cell = _positioned_cell(deg_items, beats)
    return chord_cell, deg_cell


def _chart_to_ly(score: stream.Score, stem: str) -> str:
    """Build a box-grid chord-chart LilyPond source (markup only — no staff)."""
    part = next((p for p in score.parts if p.id == "chords"), score.parts[0])

    md = score.metadata
    title    = _esc((md.title    if md and md.title    else stem) or stem)
    composer = _esc((md.composer if md and md.composer else "")   or "")

    flat = part.flatten()
    from music21 import key as m21key, tempo as m21tempo

    key_objs = list(flat.getElementsByClass(m21key.Key)) or list(flat.getElementsByClass(m21key.KeySignature))
    if key_objs:
        ks = key_objs[0]
        mode = getattr(ks, 'mode', 'major') or 'major'
        key_name = f"{ks.tonic.name} {mode}"
    else:
        key_name = "C major"

    time_sigs = list(flat.getElementsByClass('TimeSignature'))
    beats = time_sigs[0].numerator if time_sigs else 4
    denom = time_sigs[0].denominator if time_sigs else 4

    mm = list(flat.getElementsByClass(m21tempo.MetronomeMark))
    bpm = int(round(mm[0].number)) if mm else 120

    measures = list(part.getElementsByClass('Measure'))
    cells = [_measure_cell_parts(m, beats) for m in measures]

    # Internal measure-divider: \filled-box reports its extent to LilyPond so
    # \rounded-box can compute the correct box height (unlike \draw-line which has
    # zero extent and causes top/bottom border clipping).
    # Y range -0.3..1.8 covers the full ascent/descent of \large bold text.
    vbar_w = 0.88  # horizontal width of one divider: \hspace 0.4 + box 0.08 + \hspace 0.4
    vbar = " \\hspace #0.4 \\filled-box #'(-0.04 . 0.04) #'(-0.3 . 1.8) #0 \\hspace #0.4 "
    # barspace must equal vbar horizontal width so chord names stay aligned above cells.
    barspace = ' \\hspace #0.8 '
    # Advance per measure cell (cell width + one divider) — used to pad short final
    # rows out to a full row's width so they left-align under the rows above.
    cell_advance = _CELL_WIDTH + vbar_w
    filler_chord = f'\\hcenter-in #{_CELL_WIDTH} \\transparent "x"'

    # One \markup block per row. Every row is padded to _MEASURES_PER_LINE cells wide
    # so all center-columns share an identical width → \fill-line centers them to the
    # SAME left edge. A short final row keeps its real content on the left (the padding
    # is invisible filler on the right), so it appears left-aligned with the rows above.
    rows: list[str] = []
    for i in range(0, len(cells), _MEASURES_PER_LINE):
        row = cells[i:i + _MEASURES_PER_LINE]
        n = len(row)
        missing = _MEASURES_PER_LINE - n

        # Chord-name line: real cells + invisible filler cells out to full width.
        chord_cells = [c for c, _ in row] + [filler_chord] * missing
        chord_line = ' \\hspace #0.4 ' + barspace.join(chord_cells) + ' \\hspace #0.4 '

        # Degree strip: only real cells with INNER dividers (no outer bars). A trailing
        # \hspace pads the box line out to a full row's width, keeping the box on the left.
        deg_strip = vbar.join(d for _, d in row)
        # Box geometry:
        #  - box-padding #0.7 gives generous clearance between the text and the border
        #    on all sides (an earlier \pad-to-box approach mis-sized the box and clipped
        #    the text tops — this overrides the border padding directly instead).
        #  - a transparent strut (\with-dimensions … \null) fixes the line's vertical
        #    extent so every box is the same height regardless of its glyphs.
        strut = "\\with-dimensions #'(0 . 0) #'(-0.55 . 2.35) \\null"
        box = (
            "\\override #'(box-padding . 0.7) \\rounded-box "
            f"\\line {{ {strut} {deg_strip} }}"
        )
        if missing:
            box_line = f'\\line {{ {box} \\hspace #{missing * cell_advance:.2f} }}'
        else:
            box_line = box

        # \pad-markup #_ROW_PAD bakes a fixed whitespace margin around the WHOLE row
        # (chord line + box) directly into the row's bounding box. This is the only
        # reliable way to space top-level markups: inter-markup spacing variables
        # (markup-system-spacing) are not honoured between consecutive markups, which
        # left rows cramped and made box borders collide with the next row's text.
        # A per-row built-in margin guarantees a uniform, generous gap and clearance.
        rows.append(
            '\\markup \\fill-line { '
            f'\\pad-markup #{_ROW_PAD} \\center-column {{ '
            f'\\line {{ {chord_line} }} '
            f'{box_line} '
            '} }'
        )
    body = "\n".join(rows)

    subtitle = _esc(f"Key: {key_name}   |   {beats}/{denom}   |   tempo {bpm}")

    return f'''\\version "2.24.0"
\\language "english"

#(set-global-staff-size 18)

\\header {{
  title = "{title}"
  composer = "{composer}"
  subtitle = "{subtitle}"
  tagline = ##f
}}

\\paper {{
  #(set-paper-size "a4")
  top-margin = 15\\mm
  left-margin = 8\\mm
  right-margin = 8\\mm
  % ragged-bottom disables vertical justification, so rows keep their natural spacing
  % and are NEVER compressed (which was clipping the tallest box outlines). Content
  % simply flows onto another page when it doesn't fit.
  ragged-bottom = ##t
  ragged-last-bottom = ##t
  % Uniform, generous gap between rows.
  markup-system-spacing.basic-distance = #14
  markup-system-spacing.padding = #1
}}

{body}
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
