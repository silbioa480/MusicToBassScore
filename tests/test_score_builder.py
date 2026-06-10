"""Tests for the score builder module."""

import pytest
from music21 import stream

from music_to_bass_score.score_builder import (
    _midi_to_tab,
    _parse_key_string,
    _quantize,
    build_score,
)


class TestParseKeyString:
    def test_major(self):
        assert _parse_key_string("C major") == ("C", "major")

    def test_minor(self):
        assert _parse_key_string("A minor") == ("A", "minor")

    def test_sharp(self):
        assert _parse_key_string("F# major") == ("F#", "major")


class TestQuantize:
    def test_snap_to_grid(self):
        assert _quantize(0.13) == pytest.approx(0.25)
        assert _quantize(0.37) == pytest.approx(0.25)
        assert _quantize(0.63) == pytest.approx(0.75)

    def test_exact_grid_unchanged(self):
        assert _quantize(1.0) == pytest.approx(1.0)
        assert _quantize(0.5) == pytest.approx(0.5)


class TestMidiToTab:
    def test_open_e_string(self):
        string_num, fret = _midi_to_tab(28)
        assert fret == 0

    def test_returns_valid_fret(self):
        for midi_pitch in range(28, 68):
            string_num, fret = _midi_to_tab(midi_pitch)
            assert 0 <= fret <= 24
            assert 1 <= string_num <= 4


class TestBuildScore:
    def test_returns_score(self, sample_metadata, sample_analysis, sample_note_events):
        score = build_score(
            song_metadata=sample_metadata,
            analysis=sample_analysis,
            note_events=sample_note_events,
            chord_labels=["Am", "F", "C", "G"],
            include_tab=False,
        )
        assert isinstance(score, stream.Score)

    def test_score_has_parts(self, sample_metadata, sample_analysis, sample_note_events):
        score = build_score(
            song_metadata=sample_metadata,
            analysis=sample_analysis,
            note_events=sample_note_events,
            chord_labels=["Am", "F"],
            include_tab=True,
        )
        parts = list(score.parts)
        assert len(parts) >= 1

    def test_score_metadata(self, sample_metadata, sample_analysis, sample_note_events):
        score = build_score(
            song_metadata=sample_metadata,
            analysis=sample_analysis,
            note_events=sample_note_events,
            chord_labels=["Am"],
            include_tab=False,
        )
        assert score.metadata.title == "Test Song"
        assert score.metadata.composer == "Test Artist"
