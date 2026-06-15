"""Bass MIDI transcription using Basic-Pitch (Spotify)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import (
    BASS_MAX_FREQUENCY,
    BASS_MIDI_MAX,
    BASS_MIDI_MIN,
    BASS_MIN_FREQUENCY,
    MIDI_DIR,
    MIN_NOTE_DURATION_SEC,
)
from .logger import get_logger

logger = get_logger(__name__)


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
    onset_threshold: float = 0.4,   # aggressive: recover more 8th-note onsets
    frame_threshold: float = 0.25,  # aggressive: separated bass stem has weak signal; capture more frames
    minimum_frequency: float = BASS_MIN_FREQUENCY,
    maximum_frequency: float = BASS_MAX_FREQUENCY,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> TranscriptionResult:
    """Transcribe bass audio to MIDI using Basic-Pitch."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    logger.info(
        "Transcribing: %s (onset=%.2f frame=%.2f freq=%.0f–%.0fHz)",
        bass_wav_path, onset_threshold, frame_threshold,
        minimum_frequency, maximum_frequency,
    )

    if progress_cb:
        progress_cb(0.1)

    try:
        _, midi_data, note_events = predict(
            str(bass_wav_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_duration * 1000,  # seconds → milliseconds
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
    except Exception as exc:
        logger.error("Basic-Pitch transcription failed: %s", exc, exc_info=True)
        raise

    if progress_cb:
        progress_cb(0.8)

    events = _convert_note_events(note_events)

    # Post-process: filter range → make monophonic → merge fragments
    before = len(events)
    events = _filter_midi_range(events)
    events = _make_monophonic(events)
    events = _merge_consecutive(events, max_gap_sec=0.05)  # 50ms: only merge true Basic-Pitch dropout gaps, not repeated 8th notes
    logger.info(
        "Post-processing: %d raw → %d clean notes", before, len(events)
    )

    # Write cleaned MIDI
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=33)  # electric bass (finger)
    for ev in events:
        inst.notes.append(pretty_midi.Note(
            velocity=ev.velocity,
            pitch=ev.pitch,
            start=ev.start_sec,
            end=ev.end_sec,
        ))
    pm.instruments.append(inst)

    midi_path = output_dir / f"{bass_wav_path.stem}.mid"
    pm.write(str(midi_path))

    if progress_cb:
        progress_cb(1.0)

    logger.info("Transcription complete: %d notes → %s", len(events), midi_path)
    return TranscriptionResult(midi_path=midi_path, note_events=events)


# ── Post-processing helpers ───────────────────────────────────────────────────

def _filter_midi_range(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Remove notes outside the bass guitar MIDI range."""
    return [n for n in notes if BASS_MIDI_MIN <= n.pitch <= BASS_MIDI_MAX]


def _make_monophonic(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Convert polyphonic detections to a single-voice bass line.

    Basic-Pitch detects the same physical note as multiple simultaneous
    pitches (fundamental + harmonics). This reduces to one voice at a time:
    - Same-pitch overlaps → merged (extend end time)
    - Different-pitch overlaps → earlier note truncated at the later note's start;
      if the truncated note is < 50 ms it is discarded
    """
    if not notes:
        return notes

    notes = sorted(notes, key=lambda n: (n.start_sec, n.pitch))
    result: list[NoteEvent] = []

    for note in notes:
        if not result:
            result.append(note)
            continue

        prev = result[-1]
        overlap = prev.end_sec - note.start_sec

        if overlap <= 0.01:
            # No meaningful overlap
            result.append(note)
        elif prev.pitch == note.pitch:
            # Same pitch: extend
            result[-1] = NoteEvent(
                pitch=prev.pitch,
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, note.end_sec),
                velocity=max(prev.velocity, note.velocity),
            )
        else:
            # Different pitch: truncate previous at the new note's start
            new_end = note.start_sec
            if new_end - prev.start_sec >= 0.05:
                result[-1] = NoteEvent(
                    pitch=prev.pitch,
                    start_sec=prev.start_sec,
                    end_sec=new_end,
                    velocity=prev.velocity,
                )
                result.append(note)
            else:
                # Truncated too short → discard prev, keep new
                result.pop()
                result.append(note)

    return result


def _merge_consecutive(notes: list[NoteEvent], max_gap_sec: float = 0.12) -> list[NoteEvent]:
    """Merge consecutive same-pitch notes separated by a very short gap.

    Fragmentation happens when Basic-Pitch briefly drops a sustained note
    and re-detects it. A gap smaller than max_gap_sec for the same pitch
    is treated as a continuous note.
    """
    if not notes:
        return notes

    result: list[NoteEvent] = [notes[0]]
    for note in notes[1:]:
        prev = result[-1]
        gap = note.start_sec - prev.end_sec
        if prev.pitch == note.pitch and 0 <= gap < max_gap_sec:
            result[-1] = NoteEvent(
                pitch=prev.pitch,
                start_sec=prev.start_sec,
                end_sec=note.end_sec,
                velocity=max(prev.velocity, note.velocity),
            )
        else:
            result.append(note)

    return result


# ── Raw-event converter ───────────────────────────────────────────────────────

def _convert_note_events(raw_events) -> list[NoteEvent]:
    """Convert Basic-Pitch raw output to NoteEvent list.

    Basic-Pitch returns amplitude in [0, 1]; multiply by 127 for MIDI velocity.
    """
    result = []
    for event in raw_events:
        if isinstance(event, (list, tuple)) and len(event) >= 3:
            amp = float(event[3]) if len(event) > 3 else 0.8
            result.append(NoteEvent(
                pitch=int(event[2]),
                start_sec=float(event[0]),
                end_sec=float(event[1]),
                velocity=max(1, min(127, int(amp * 127))),
            ))
        elif hasattr(event, "pitch"):
            amp = float(getattr(event, "amplitude", 0.8))
            result.append(NoteEvent(
                pitch=int(event.pitch),
                start_sec=float(event.start_time),
                end_sec=float(event.end_time),
                velocity=max(1, min(127, int(amp * 127))),
            ))
    return result
