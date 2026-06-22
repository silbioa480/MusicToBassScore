"""Tests for the audio analyzer module."""

import numpy as np
import pytest

from music_to_bass_score.analyzer import (
    _diatonic_count,
    _estimate_bpm_and_beats,
    _estimate_key,
    _estimate_time_signature,
    _is_relative_key,
    _score_key,
    build_measure_grid,
    refine_key_with_chords,
)


class TestEstimateBpm:
    def test_returns_positive_float(self, sample_wav_path):
        import librosa
        y, sr = librosa.load(str(sample_wav_path), sr=44100, mono=True)
        bpm, beats = _estimate_bpm_and_beats(y, sr)
        assert isinstance(bpm, float)
        assert 40.0 <= bpm <= 300.0
        assert hasattr(beats, "__len__")

    def test_synthetic_120bpm(self):
        sr = 44100
        bpm_expected = 120.0
        beat_period = sr * 60.0 / bpm_expected
        duration_samples = sr * 8
        y = np.zeros(duration_samples)
        beat_positions = np.arange(0, duration_samples, beat_period).astype(int)
        beat_positions = beat_positions[beat_positions < duration_samples]
        for pos in beat_positions:
            end = min(pos + 2205, duration_samples)
            y[pos:end] = 1.0
        bpm, _ = _estimate_bpm_and_beats(y, sr)
        # octave-corrected tempo should land near 120 (allow half/double tolerance band)
        assert abs(bpm - bpm_expected) < 30.0


class TestEstimateKey:
    def test_returns_string(self, sample_wav_path):
        import librosa
        y, sr = librosa.load(str(sample_wav_path), sr=44100, mono=True)
        key_str = _estimate_key(y, sr)
        assert isinstance(key_str, str)
        assert "major" in key_str or "minor" in key_str

    def test_contains_note_name(self, sample_wav_path):
        import librosa
        y, sr = librosa.load(str(sample_wav_path), sr=44100, mono=True)
        key_str = _estimate_key(y, sr)
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        assert any(key_str.startswith(n) for n in notes)


class TestScoreKey:
    def test_returns_key_string(self):
        # Strong C-major-ish chroma (C, E, G emphasised)
        vec = np.array([1.0, 0.1, 0.3, 0.1, 0.8, 0.4, 0.1, 0.9, 0.1, 0.3, 0.1, 0.2])
        key_str = _score_key(vec)
        assert isinstance(key_str, str)
        assert "major" in key_str or "minor" in key_str


class TestEstimateTimeSignature:
    def test_returns_tuple(self, sample_wav_path):
        import librosa
        y, sr = librosa.load(str(sample_wav_path), sr=44100, mono=True)
        result = _estimate_time_signature(y, sr, 120.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] in (3, 4)
        assert result[1] == 4


class TestBuildMeasureGrid:
    def test_constant_spacing(self):
        grid = build_measure_grid(bpm=120.0, beats_per_measure=4, anchor=0.0, duration_sec=10.0)
        assert grid[0] == 0.0
        # 120 bpm, 4 beats/measure → 2.0 s per measure
        assert grid[1] == pytest.approx(2.0, abs=0.01)
        assert all(b > a for a, b in zip(grid, grid[1:]))


class TestDiatonicCount:
    def test_counts_in_key_chords(self):
        # C major diatonic: C, Dm, Em, F, G, Am, Bdim
        measures = [[(0.0, "C")], [(0.0, "F")], [(0.0, "G")], [(0.0, "Am")]]
        assert _diatonic_count(measures, "C major") == 4

    def test_skips_nc_and_unknown(self):
        measures = [[(0.0, "N.C.")], [(0.0, "C")]]
        assert _diatonic_count(measures, "C major") == 1

    def test_strips_slash_and_question_mark(self):
        measures = [[(0.0, "G/B")], [(0.0, "C?")]]
        assert _diatonic_count(measures, "C major") == 2


class TestIsRelativeKey:
    def test_g_major_e_minor(self):
        assert _is_relative_key("G major", "E minor")
        assert _is_relative_key("E minor", "G major")

    def test_c_major_a_minor(self):
        assert _is_relative_key("C major", "A minor")
        assert _is_relative_key("A minor", "C major")

    def test_parallel_keys_not_relative(self):
        assert not _is_relative_key("A major", "A minor")

    def test_same_mode_not_relative(self):
        assert not _is_relative_key("G major", "D major")

    def test_unrelated_keys(self):
        assert not _is_relative_key("C major", "D minor")


class TestRefineKeyWithChords:
    def test_keeps_consistent_major_key(self):
        key_labels = ["C major"] * 8
        chords = [[(0.0, "C")], [(0.0, "F")], [(0.0, "G")], [(0.0, "Am")]] * 2
        refined = refine_key_with_chords(key_labels, chords, window=8)
        assert refined == key_labels

    def test_flips_to_parallel_minor_when_progression_supports_it(self):
        # Chroma mistakenly called it C major, but the chords are clearly C minor
        # (C minor diatonic: Cm, Ddim, Eb, Fm, Gm, Ab, Bb).
        key_labels = ["C major"] * 8
        chords = [[(0.0, "Cm")], [(0.0, "Fm")], [(0.0, "Gm")], [(0.0, "Eb")]] * 2
        refined = refine_key_with_chords(key_labels, chords, window=8)
        assert all(k == "C minor" for k in refined)

    def test_never_flips_to_relative_key(self):
        # Relative-key modulation is suppressed — G major stays G major even when
        # Em-rooted chords heavily dominate the window.
        key_labels = ["G major"] * 8
        chords = [[(0.0, "Em")], [(0.0, "Em")], [(0.0, "Em")], [(0.0, "G")]] * 2
        refined = refine_key_with_chords(key_labels, chords, window=8)
        assert all(k == "G major" for k in refined)

    def test_length_preserved(self):
        key_labels = ["A minor"] * 5
        chords = [[(0.0, "Am")]] * 5
        refined = refine_key_with_chords(key_labels, chords, window=8)
        assert len(refined) == len(key_labels)
