"""Tests for the chord detector module."""

import numpy as np
import pytest

from music_to_bass_score.chord_detector import _chroma_to_chord


class TestChromaToChord:
    def test_returns_string(self):
        chroma = np.random.rand(12)
        result = _chroma_to_chord(chroma)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_zero_vector_returns_nc(self):
        result = _chroma_to_chord(np.zeros(12))
        assert result == "N.C."

    def test_c_major_template(self):
        # C major: C(0), E(4), G(7)
        chroma = np.zeros(12)
        chroma[0] = 1.0
        chroma[4] = 1.0
        chroma[7] = 1.0
        result = _chroma_to_chord(chroma)
        assert "C" in result
        assert "m" not in result or result == "Cm7"

    def test_a_minor_template(self):
        # A minor: A(9), C(0), E(4)
        chroma = np.zeros(12)
        chroma[9] = 1.0
        chroma[0] = 1.0
        chroma[4] = 1.0
        result = _chroma_to_chord(chroma)
        assert "A" in result
