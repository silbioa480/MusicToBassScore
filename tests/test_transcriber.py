"""Tests for the transcriber module (Basic-Pitch MIDI transcription)."""

import numpy as np
import pytest
import soundfile as sf

from music_to_bass_score.transcriber import (
    NoteEvent,
    TranscriptionResult,
    _convert_note_events,
    transcribe_bass,
)


@pytest.fixture(scope="module")
def bass_wav_path(tmp_path_factory, sample_wav_path):
    """Use the sample WAV as a proxy bass input for transcription tests."""
    return sample_wav_path


class TestNoteEvent:
    def test_fields(self):
        ev = NoteEvent(pitch=45, start_sec=0.0, end_sec=0.5, velocity=80)
        assert ev.pitch == 45
        assert ev.start_sec == pytest.approx(0.0)
        assert ev.end_sec == pytest.approx(0.5)
        assert ev.velocity == 80


class TestConvertNoteEvents:
    def test_empty_input(self):
        assert _convert_note_events([]) == []

    def test_tuple_format(self):
        raw = [(0.0, 0.5, 45, 0.8)]
        events = _convert_note_events(raw)
        assert len(events) == 1
        assert events[0].pitch == 45
        assert events[0].start_sec == pytest.approx(0.0)


class TestTranscribeBass:
    def test_returns_transcription_result(self, bass_wav_path, tmp_path):
        result = transcribe_bass(bass_wav_path=bass_wav_path, output_dir=tmp_path)
        assert isinstance(result, TranscriptionResult)

    def test_midi_file_created(self, bass_wav_path, tmp_path):
        result = transcribe_bass(bass_wav_path=bass_wav_path, output_dir=tmp_path)
        assert result.midi_path.exists()
        assert result.midi_path.suffix == ".mid"

    def test_midi_file_nonempty(self, bass_wav_path, tmp_path):
        result = transcribe_bass(bass_wav_path=bass_wav_path, output_dir=tmp_path)
        assert result.midi_path.stat().st_size > 0

    def test_note_events_is_list(self, bass_wav_path, tmp_path):
        result = transcribe_bass(bass_wav_path=bass_wav_path, output_dir=tmp_path)
        assert isinstance(result.note_events, list)

    def test_note_events_have_valid_midi_range(self, bass_wav_path, tmp_path):
        result = transcribe_bass(
            bass_wav_path=bass_wav_path,
            output_dir=tmp_path,
            minimum_frequency=30.0,
            maximum_frequency=300.0,
        )
        for ev in result.note_events:
            assert 0 <= ev.pitch <= 127

    def test_progress_callback_called(self, bass_wav_path, tmp_path):
        calls = []
        transcribe_bass(
            bass_wav_path=bass_wav_path,
            output_dir=tmp_path,
            progress_cb=lambda f: calls.append(f),
        )
        assert len(calls) > 0
        assert calls[-1] == pytest.approx(1.0)

    def test_missing_file_raises(self, tmp_path):
        from pathlib import Path
        with pytest.raises(Exception):
            transcribe_bass(
                bass_wav_path=Path("/nonexistent/bass.wav"),
                output_dir=tmp_path,
            )
