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

def _measure_to_ly(measure, beats: int, chord_label: Optional[str]) -> str:
    """Render a music21 Measure as a LilyPond measure string (ending with |)."""
    from music21 import note as m21note

    markup = ''
    if chord_label:
        markup = f'^\\markup{{\\bold "{_esc(chord_label)}"}}'

    tokens: list[str] = []
    cur = 0.0
    chord_placed = False

    elems = sorted(measure.notesAndRests, key=lambda e: float(e.offset))

    for elem in elems:
        off = round(float(elem.offset) * 32) / 32

        if off < cur - 0.02:
            continue  # overlapping note — skip

        # Fill gap before this element
        gap = round((off - cur) * 32) / 32
        if gap > 0.1:
            mk = ''
            if not chord_placed and markup:
                mk = markup
                chord_placed = True
            tokens.append(_rest_ly(gap, mk))

        cur = off

        # Truncate note at measure boundary
        avail = round((beats - off) * 32) / 32
        ql = round(min(float(elem.quarterLength), max(0.125, avail)) * 32) / 32
        ql = max(0.125, ql)

        mk = ''
        if not chord_placed and markup:
            mk = markup
            chord_placed = True

        if isinstance(elem, m21note.Note):
            tokens.append(_note_ly(elem.pitch.midi, ql, mk))
        else:
            tokens.append(_rest_ly(ql, mk))

        cur = round((off + ql) * 32) / 32

    # Fill remaining tail
    tail = round((beats - cur) * 32) / 32
    if tail > 0.1:
        mk = ''
        if not chord_placed and markup:
            mk = markup
            chord_placed = True
        tokens.append(_rest_ly(tail, mk))

    if not tokens:
        mk = markup if markup else ''
        tokens = [_rest_ly(float(beats), mk)]

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
    bass_lines: list[str] = []
    tab_lines:  list[str] = []

    for i, measure in enumerate(bass_part.getElementsByClass('Measure')):
        chord_label = None
        for te in measure.getElementsByClass('TextExpression'):
            chord_label = te.content
            break

        bass_lines.append(f"  {_measure_to_ly(measure, beats, chord_label)}  % m{i+1}")
        tab_lines.append(f"  {_measure_to_ly(measure, beats, None)}")

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
      \\override SpacingSpanner.common-shortest-duration = #(ly:make-moment 1/8)
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

    ly_content = _score_to_ly(score, filename_stem)
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
