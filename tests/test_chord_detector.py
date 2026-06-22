"""Tests for the chord detector module."""

import numpy as np
import pytest

from music_to_bass_score.chord_detector import (
    _chord_quality,
    _chord_root,
    _match_chord,
    _snap_to_btc_boundary,
)


def _chroma_matrix(active_pitches, n_frames=4):
    """Build a (12, n_frames) chroma matrix with the given pitch classes active."""
    vec = np.zeros(12)
    for p in active_pitches:
        vec[p] = 1.0
    return np.tile(vec[:, None], (1, n_frames))


class TestMatchChord:
    def test_returns_string(self):
        chroma = _chroma_matrix([0, 4, 7])
        result = _match_chord(chroma, 0, 4)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_zero_matrix_returns_nc(self):
        chroma = np.zeros((12, 4))
        assert _match_chord(chroma, 0, 4) == "N.C."

    def test_start_frame_past_end_returns_nc(self):
        chroma = _chroma_matrix([0, 4, 7])
        assert _match_chord(chroma, 99, 100) == "N.C."

    def test_c_major_template(self):
        # C major: C(0), E(4), G(7)
        chroma = _chroma_matrix([0, 4, 7])
        result = _match_chord(chroma, 0, 4)
        assert result.startswith("C")
        # major triad → no minor 'm' immediately after the root
        assert not result.startswith("Cm")

    def test_a_minor_template(self):
        # A minor: A(9), C(0), E(4)
        chroma = _chroma_matrix([9, 0, 4])
        result = _match_chord(chroma, 0, 4)
        assert result.startswith("A")


class TestChordRoot:
    def test_natural_root(self):
        assert _chord_root("Am7") == "A"

    def test_sharp_root(self):
        assert _chord_root("F#m") == "F#"

    def test_flat_root(self):
        assert _chord_root("Bb7") == "Bb"

    def test_nc_returns_empty(self):
        assert _chord_root("N.C.") == ""


class TestChordQuality:
    def test_major_is_empty(self):
        assert _chord_quality("G") == ""

    def test_minor_seventh(self):
        assert _chord_quality("Cm7") == "m7"

    def test_quality_after_sharp_root(self):
        assert _chord_quality("F#maj7") == "maj7"


class TestSnapToBtcBoundary:
    def _timeline(self):
        # (start, end, symbol, confidence)
        return [
            (0.0, 2.05, "C", 0.9),
            (2.05, 4.0, "G", 0.8),
            (4.0, 6.0, "Am", 0.7),
        ]

    def test_snaps_to_nearby_boundary(self):
        # grid says 2.0; a BTC boundary sits at 2.05 within ±0.5 beats (beat_dur=0.5)
        snapped = _snap_to_btc_boundary(self._timeline(), 2.0, beat_dur=0.5)
        assert snapped == pytest.approx(2.05)

    def test_no_snap_when_out_of_tolerance(self):
        # grid says 3.0; nearest boundary is 2.05/4.0, both > 0.25s away (0.5 beat × 0.5 dur)
        snapped = _snap_to_btc_boundary(self._timeline(), 3.0, beat_dur=0.5)
        assert snapped == pytest.approx(3.0)

    def test_returns_original_for_empty_timeline(self):
        assert _snap_to_btc_boundary([], 1.234, beat_dur=0.5) == pytest.approx(1.234)
