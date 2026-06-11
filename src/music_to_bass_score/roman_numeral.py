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

    Falls back to the raw chord symbol if analysis fails.
    """
    if not symbol or symbol in ("N.C.", "NC"):
        return symbol or ""

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


def measures_to_roman(chord_measures: list, key_str: str) -> list:
    """Convert per-measure (offset, chord) tuples to (offset, roman) tuples.

    Input/output shape: list[list[tuple[float, str]]].
    """
    result = []
    for measure in chord_measures:
        result.append([(off, chord_to_roman(sym, key_str)) for off, sym in measure])
    return result
