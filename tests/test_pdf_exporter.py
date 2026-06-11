"""Tests for the pdf_exporter module."""

import pytest
from pathlib import Path

from music_to_bass_score.pdf_exporter import (
    ExportResult,
    check_lilypond_available,
    export_to_pdf,
)
from music_to_bass_score.score_builder import build_score


@pytest.fixture(scope="module")
def simple_score(sample_metadata, sample_analysis, sample_note_events):
    return build_score(
        song_metadata=sample_metadata,
        analysis=sample_analysis,
        note_events=sample_note_events,
        chord_labels=["Am", "F", "C", "G"],
        include_tab=False,
    )


class TestCheckLilypondAvailable:
    def test_returns_bool(self):
        result = check_lilypond_available()
        assert isinstance(result, bool)


class TestExportToPdf:
    def test_returns_export_result(self, simple_score, tmp_path):
        result = export_to_pdf(score=simple_score, output_dir=tmp_path, filename_stem="test")
        assert isinstance(result, ExportResult)

    def test_output_file_exists(self, simple_score, tmp_path):
        result = export_to_pdf(score=simple_score, output_dir=tmp_path, filename_stem="test")
        assert result.pdf_path.exists()

    def test_output_file_nonempty(self, simple_score, tmp_path):
        result = export_to_pdf(score=simple_score, output_dir=tmp_path, filename_stem="test")
        assert result.pdf_path.stat().st_size > 0

    def test_lilypond_method_produces_pdf(self, simple_score, tmp_path):
        if not check_lilypond_available():
            pytest.skip("LilyPond not installed")
        result = export_to_pdf(
            score=simple_score,
            output_dir=tmp_path,
            filename_stem="test_lily",
            method="lilypond",
        )
        assert result.pdf_path.suffix == ".pdf"
        assert result.lily_path is not None
        assert result.lily_path.exists()

    def test_musicxml_fallback(self, simple_score, tmp_path):
        result = export_to_pdf(
            score=simple_score,
            output_dir=tmp_path,
            filename_stem="test_xml",
            method="musicxml",
        )
        assert result.pdf_path.exists()

    def test_output_dir_created_if_missing(self, simple_score, tmp_path):
        new_dir = tmp_path / "nested" / "output"
        assert not new_dir.exists()
        result = export_to_pdf(score=simple_score, output_dir=new_dir, filename_stem="test")
        assert new_dir.exists()
        assert result.pdf_path.exists()
