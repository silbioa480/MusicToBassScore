"""Tests for the separator module (Demucs bass separation)."""

import numpy as np
import pytest
import soundfile as sf

from music_to_bass_score.separator import (
    SeparationResult,
    _resolve_device,
    check_model_cached,
    separate_bass,
)


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert _resolve_device("cuda") == "cuda"

    def test_auto_returns_string(self):
        result = _resolve_device("auto")
        assert result in ("cpu", "cuda")


class TestCheckModelCached:
    def test_returns_bool(self):
        result = check_model_cached()
        assert isinstance(result, bool)


class TestSeparateBass:
    def test_returns_separation_result(self, sample_wav_path, tmp_path):
        result = separate_bass(audio_path=sample_wav_path, output_dir=tmp_path)
        assert isinstance(result, SeparationResult)

    def test_bass_file_exists(self, sample_wav_path, tmp_path):
        result = separate_bass(audio_path=sample_wav_path, output_dir=tmp_path)
        assert result.bass_path.exists()

    def test_bass_file_is_wav(self, sample_wav_path, tmp_path):
        result = separate_bass(audio_path=sample_wav_path, output_dir=tmp_path)
        assert result.bass_path.suffix == ".wav"

    def test_bass_file_is_nonzero(self, sample_wav_path, tmp_path):
        result = separate_bass(audio_path=sample_wav_path, output_dir=tmp_path)
        assert result.bass_path.stat().st_size > 0

    def test_bass_audio_is_valid(self, sample_wav_path, tmp_path):
        result = separate_bass(audio_path=sample_wav_path, output_dir=tmp_path)
        data, sr = sf.read(str(result.bass_path))
        assert data.ndim >= 1
        assert sr > 0

    def test_progress_callback_called(self, sample_wav_path, tmp_path):
        calls = []
        separate_bass(
            audio_path=sample_wav_path,
            output_dir=tmp_path,
            progress_cb=lambda f: calls.append(f),
        )
        assert len(calls) > 0
        assert calls[-1] == pytest.approx(1.0)

    def test_missing_file_raises(self, tmp_path):
        from pathlib import Path
        with pytest.raises(Exception):
            separate_bass(
                audio_path=Path("/nonexistent/file.wav"),
                output_dir=tmp_path,
            )
