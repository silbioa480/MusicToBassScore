"""Tests for the audio analyzer module."""

import pytest

from music_to_bass_score.analyzer import _estimate_bpm, _estimate_key, _estimate_time_signature


class TestEstimateBpm:
    def test_returns_positive_float(self, sample_wav_path):
        import librosa
        y, sr = librosa.load(str(sample_wav_path), sr=44100, mono=True)
        bpm = _estimate_bpm(y, sr)
        assert isinstance(bpm, float)
        assert 40.0 <= bpm <= 300.0

    def test_synthetic_120bpm(self):
        import numpy as np
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
        bpm = _estimate_bpm(y, sr)
        assert abs(bpm - bpm_expected) < 20.0


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


class TestEstimateTimeSignature:
    def test_returns_tuple(self, sample_wav_path):
        import librosa
        y, sr = librosa.load(str(sample_wav_path), sr=44100, mono=True)
        result = _estimate_time_signature(y, sr, 120.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] in (3, 4)
        assert result[1] == 4
