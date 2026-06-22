"""Convert chord symbols to key-relative roman-numeral scale degrees.

Uses music21's roman-numeral analysis for the degree base (I, ii, V, ...) and appends
a clean quality suffix derived from the chord symbol, so output is readable:
  key C major:  G7 → V7,  Dm7 → ii7,  F → IV,  Cmaj7 → Imaj7,  Bdim → vii°
"""

from .logger import get_logger

logger = get_logger(__name__)

# Chord-symbol suffix → roman-numeral suffix. Case (major/minor) is already encoded
# in music21's romanNumeralAlone, so triads need no extra mark.
_SUFFIX_MAP = {
    "": "",
    "m": "",
    "7": "7",
    "m7": "7",
    "maj7": "maj7",
    "dim": "°",   # ° degree sign
}


def _quality_suffix(symbol: str) -> str:
    """Extract the chord-quality portion (after the root) from a chord symbol."""
    if not symbol or symbol in ("N.C.", "NC"):
        return ""
    # Strip root: letter + optional accidental
    i = 1
    if len(symbol) > 1 and symbol[1] in ("#", "b"):
        i = 2
    return symbol[i:]


def chord_to_roman(symbol: str, key_str: str) -> str:
    """Return the roman-numeral degree of `symbol` within key `key_str` (e.g. 'C major').

    Handles slash chords (e.g. 'G/B'): converts the upper chord part to a roman
    numeral and appends the bass degree as a slash (e.g. 'I/III').
    A trailing "?" confidence marker is stripped before analysis and not propagated
    to the roman numeral (the chord label already carries the uncertainty signal).
    Falls back to the raw chord symbol if analysis fails.
    """
    if not symbol or symbol in ("N.C.", "NC"):
        return symbol or ""

    # Strip low-confidence marker before music21 analysis
    if symbol.endswith("?"):
        symbol = symbol[:-1]

    # Split slash chord: upper chord + optional bass note
    if "/" in symbol:
        upper, bass_note = symbol.split("/", 1)
        upper_rn = chord_to_roman(upper, key_str)
        bass_rn = _note_to_roman_degree(bass_note, key_str)
        if bass_rn:
            return f"{upper_rn}/{bass_rn}"
        return upper_rn

    try:
        from music21 import harmony, key as m21key, roman

        parts = key_str.strip().split()
        tonic = parts[0] if parts else "C"
        mode = parts[1].lower() if len(parts) > 1 else "major"
        k = m21key.Key(tonic, mode)

        cs = harmony.ChordSymbol(symbol)
        rn = roman.romanNumeralFromChord(cs, k)
        base = rn.romanNumeralAlone  # 'V', 'ii', 'I', ... (case = chord major/minor)

        # romanNumeralAlone drops chromatic accidentals; recover from the scale degree
        prefix = ""
        try:
            _, acc = rn.scaleDegreeWithAlteration
            if acc is not None and acc.alter:
                prefix = "#" if acc.alter > 0 else "b"
        except Exception:
            pass

        suffix = _SUFFIX_MAP.get(_quality_suffix(symbol), "")
        return f"{prefix}{base}{suffix}"
    except Exception as exc:
        logger.debug("Roman conversion failed for %r in %r: %s", symbol, key_str, exc)
        return symbol


def _note_to_roman_degree(note: str, key_str: str) -> str:
    """Convert a single note name to its scale-degree numeral (e.g. 'B' in G major → 'III').

    Returns an empty string if conversion fails.
    """
    _ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII"]
    _NOTE_SEMITONES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                       "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
                       "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    try:
        parts = key_str.strip().split()
        tonic = parts[0] if parts else "C"
        tonic_semi = _NOTE_SEMITONES.get(tonic, 0)
        note_semi = _NOTE_SEMITONES.get(note)
        if note_semi is None:
            return ""
        interval = (note_semi - tonic_semi) % 12
        # Map semitone interval to diatonic scale degree (major scale intervals)
        _MAJOR_DEGREES = {0: "I", 2: "II", 4: "III", 5: "IV", 7: "V", 9: "VI", 11: "VII"}
        # Chromatic degrees get the nearest with accidental
        degree = _MAJOR_DEGREES.get(interval)
        if degree:
            return degree
        # Use flat/sharp for chromatic notes
        if interval - 1 in _MAJOR_DEGREES:
            return f"#{_MAJOR_DEGREES[interval - 1]}"
        if interval + 1 in _MAJOR_DEGREES:
            return f"b{_MAJOR_DEGREES[interval + 1]}"
        return ""
    except Exception:
        return ""


def measures_to_roman(chord_measures: list, key: "str | list[str]") -> list:
    """Convert per-measure (offset, chord) tuples to (offset, roman) tuples.

    key may be a single string (applied to all measures) or a list of strings
    (one per measure, for songs with key modulations).

    Input/output shape: list[list[tuple[float, str]]].
    """
    result = []
    for i, measure in enumerate(chord_measures):
        key_str = key[i] if isinstance(key, list) else key
        result.append([(off, chord_to_roman(sym, key_str)) for off, sym in measure])
    return result
