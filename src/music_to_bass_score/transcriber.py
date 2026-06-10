"""Bass MIDI transcription using Basic-Pitch (Spotify)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import (
    BASS_MAX_FREQUENCY,
    BASS_MIN_FREQUENCY,
    MIDI_DIR,
    MIN_NOTE_DURATION_SEC,
)


@dataclass
class NoteEvent:
    pitch: int
    start_sec: float
    end_sec: float
    velocity: int


@dataclass
class TranscriptionResult:
    midi_path: Path
    note_events: list[NoteEvent] = field(default_factory=list)


def transcribe_bass(
    bass_wav_path: Path,
    output_dir: Path = MIDI_DIR,
    min_note_duration: float = MIN_NOTE_DURATION_SEC,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_frequency: float = BASS_MIN_FREQUENCY,
    maximum_frequency: float = BASS_MAX_FREQUENCY,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> TranscriptionResult:
    """Transcribe bass audio to MIDI using Basic-Pitch."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    if progress_cb:
        progress_cb(0.1)

    _, midi_data, note_events = predict(
        str(bass_wav_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_duration_s=min_note_duration,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    if progress_cb:
        progress_cb(0.8)

    midi_path = output_dir / f"{bass_wav_path.stem}.mid"
    midi_data.write(str(midi_path))

    events = _convert_note_events(note_events)

    if progress_cb:
        progress_cb(1.0)

    return TranscriptionResult(midi_path=midi_path, note_events=events)


def _convert_note_events(raw_events) -> list[NoteEvent]:
    """Convert Basic-Pitch note events to NoteEvent dataclass list."""
    result = []
    for event in raw_events:
        if isinstance(event, (list, tuple)) and len(event) >= 3:
            pitch = int(event[2]) if len(event) > 2 else 60
            start = float(event[0])
            end = float(event[1])
            velocity = int(event[3]) if len(event) > 3 else 80
            result.append(NoteEvent(pitch=pitch, start_sec=start, end_sec=end, velocity=velocity))
        elif hasattr(event, "pitch"):
            result.append(NoteEvent(
                pitch=int(event.pitch),
                start_sec=float(event.start_time),
                end_sec=float(event.end_time),
                velocity=int(getattr(event, "amplitude", 0.8) * 127),
            ))
    return result
