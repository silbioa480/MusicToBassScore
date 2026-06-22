"""Tests for the pipeline module.

Integration tests are marked with @pytest.mark.integration and require
a real YouTube URL + network access. Unit tests mock the individual steps.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from music_to_bass_score.pipeline import PipelineResult, _safe_filename


class TestSafeFilename:
    def test_basic(self):
        assert _safe_filename("Hello World") == "Hello_World"

    def test_special_chars_removed(self):
        result = _safe_filename("曲: テスト / Song!")
        assert "/" not in result
        assert ":" not in result

    def test_empty_fallback(self):
        assert _safe_filename("!!!") == "bass_score"

    def test_long_title_truncated(self):
        long_title = "A" * 200
        result = _safe_filename(long_title)
        assert len(result) <= 80

    def test_japanese_title(self):
        result = _safe_filename("テスト曲")
        assert isinstance(result, str)
        assert len(result) > 0


class TestPipelineMocked:
    """Unit tests that mock every sub-step to test orchestration logic only."""

    def _make_mocks(self, tmp_path):
        from music_to_bass_score.downloader import SongMetadata
        from music_to_bass_score.analyzer import AudioAnalysis
        from music_to_bass_score.separator import SeparationResult
        from music_to_bass_score.pdf_exporter import ExportResult
        from music21 import stream

        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"RIFF")
        bass = tmp_path / "bass.wav"
        bass.write_bytes(b"RIFF")
        pdf = tmp_path / "score.pdf"
        pdf.write_bytes(b"%PDF")

        meta = SongMetadata(
            title="Mock Song", artist="Mock Artist",
            duration_sec=60.0, youtube_url="https://youtu.be/test",
            audio_path=wav,
        )
        analysis = AudioAnalysis(
            bpm=120.0, bpm_rounded=120, key="A minor",
            time_signature_num=4, time_signature_den=4,
            duration_sec=60.0, sample_rate=44100,
        )
        # vocals/other left None → harmonic-mix step is skipped (BTC uses full mix)
        sep = SeparationResult(bass_path=bass, stems_dir=tmp_path)
        export = ExportResult(pdf_path=pdf, lily_path=None)
        score = stream.Score()

        # Current pipeline shapes: per-measure list of (offset, label) tuples
        chord_labels = [[(0.0, "Am")], [(0.0, "F")]]
        key_labels = ["A minor", "A minor"]
        measure_grid = [0.0, 2.0]

        return meta, analysis, sep, export, score, chord_labels, key_labels, measure_grid

    def _patches(self, meta, analysis, sep, export, score,
                 chord_labels, key_labels, measure_grid):
        """Patch every orchestrated sub-step of the current chord-chart pipeline."""
        return [
            patch("music_to_bass_score.pipeline.download_audio", return_value=meta),
            patch("music_to_bass_score.pipeline.analyze_audio", return_value=analysis),
            patch("music_to_bass_score.pipeline.detect_first_onset", return_value=0.0),
            patch("music_to_bass_score.pipeline.build_measure_grid", return_value=measure_grid),
            patch("music_to_bass_score.pipeline.detect_key_per_section", return_value=key_labels),
            patch("music_to_bass_score.pipeline.separate_bass_cached", return_value=sep),
            patch("music_to_bass_score.pipeline.detect_chords_per_measure", return_value=chord_labels),
            patch("music_to_bass_score.pipeline.refine_key_with_chords", return_value=key_labels),
            patch("music_to_bass_score.pipeline.measures_to_roman",
                  return_value=[[(0.0, "i")], [(0.0, "VI")]]),
            patch("music_to_bass_score.pipeline.build_chord_chart", return_value=score),
            patch("music_to_bass_score.pipeline.export_to_pdf", return_value=export),
        ]

    def test_pipeline_returns_result(self, tmp_path):
        mocks = self._make_mocks(tmp_path)
        chord_labels = mocks[5]

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self._patches(*mocks):
                stack.enter_context(p)
            from music_to_bass_score.pipeline import run_pipeline
            result = run_pipeline(
                youtube_url="https://youtu.be/test",
                output_dir=tmp_path,
            )

        assert isinstance(result, PipelineResult)
        assert result.metadata.title == "Mock Song"
        assert result.analysis.bpm_rounded == 120
        assert result.chord_labels == chord_labels
        assert result.export.pdf_path.exists()

    def test_progress_callback_receives_messages(self, tmp_path):
        mocks = self._make_mocks(tmp_path)
        messages = []

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self._patches(*mocks):
                stack.enter_context(p)
            from music_to_bass_score.pipeline import run_pipeline
            run_pipeline(
                youtube_url="https://youtu.be/test",
                output_dir=tmp_path,
                progress_cb=lambda msg, frac: messages.append((msg, frac)),
            )

        assert len(messages) > 0
        fracs = [f for _, f in messages]
        assert fracs[-1] == pytest.approx(1.0)


class TestRunPipelineFromFile:
    """Tests for the file-upload pipeline entry point."""

    def test_returns_result(self, sample_wav_path, tmp_path):
        from music_to_bass_score.pipeline import run_pipeline_from_file
        result = run_pipeline_from_file(
            audio_path=sample_wav_path,
            title="File Test",
            artist="Test Artist",
            output_dir=tmp_path,
        )
        assert isinstance(result, PipelineResult)

    def test_title_from_arg(self, sample_wav_path, tmp_path):
        from music_to_bass_score.pipeline import run_pipeline_from_file
        result = run_pipeline_from_file(
            audio_path=sample_wav_path, title="My Song", output_dir=tmp_path
        )
        assert result.metadata.title == "My Song"

    def test_title_falls_back_to_stem(self, sample_wav_path, tmp_path):
        from music_to_bass_score.pipeline import run_pipeline_from_file
        result = run_pipeline_from_file(
            audio_path=sample_wav_path, title="", output_dir=tmp_path
        )
        assert result.metadata.title == sample_wav_path.stem

    def test_pdf_created(self, sample_wav_path, tmp_path):
        from music_to_bass_score.pipeline import run_pipeline_from_file
        result = run_pipeline_from_file(
            audio_path=sample_wav_path, output_dir=tmp_path
        )
        assert result.export.pdf_path.exists()
        assert result.export.pdf_path.stat().st_size > 0

    def test_nonexistent_file_raises(self, tmp_path):
        from music_to_bass_score.pipeline import run_pipeline_from_file
        with pytest.raises(FileNotFoundError):
            run_pipeline_from_file(
                audio_path=Path("/no/such/file.wav"), output_dir=tmp_path
            )

    def test_unsupported_format_raises(self, tmp_path):
        from music_to_bass_score.pipeline import run_pipeline_from_file
        fake = tmp_path / "song.xyz"
        fake.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            run_pipeline_from_file(audio_path=fake, output_dir=tmp_path)

    def test_progress_callback(self, sample_wav_path, tmp_path):
        from music_to_bass_score.pipeline import run_pipeline_from_file
        calls = []
        run_pipeline_from_file(
            audio_path=sample_wav_path,
            output_dir=tmp_path,
            progress_cb=lambda msg, frac: calls.append(frac),
        )
        assert calls[-1] == pytest.approx(1.0)


@pytest.mark.integration
class TestPipelineIntegration:
    """Real end-to-end test requiring YouTube access and GPU/CPU time."""

    YOUTUBE_URL = "https://youtu.be/dQw4w9WgXcQ"

    def test_full_pipeline(self, tmp_path):
        from music_to_bass_score.pipeline import run_pipeline
        result = run_pipeline(
            youtube_url=self.YOUTUBE_URL,
            output_dir=tmp_path,
            include_tab=True,
        )
        assert isinstance(result, PipelineResult)
        assert result.export.pdf_path.exists()
        assert result.export.pdf_path.stat().st_size > 1024
        assert len(result.chord_labels) > 0
