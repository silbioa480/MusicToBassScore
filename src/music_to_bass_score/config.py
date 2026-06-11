"""Global constants and path configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TMP_DIR = PROJECT_ROOT / "tmp"
AUDIO_DIR = TMP_DIR / "audio"
STEMS_DIR = TMP_DIR / "stems"
MIDI_DIR = TMP_DIR / "midi"
SCORES_DIR = TMP_DIR / "scores"

for _d in (AUDIO_DIR, STEMS_DIR, MIDI_DIR, SCORES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEMUCS_MODEL: str = "htdemucs_ft"

SAMPLE_RATE: int = 44100
HOP_LENGTH: int = 512
N_FFT: int = 2048

MIN_NOTE_DURATION_SEC: float = 0.08
BASS_MIN_FREQUENCY: float = 30.0
BASS_MAX_FREQUENCY: float = 300.0

LILYPOND_BIN: str = "lilypond"
LILYPOND_TIMEOUT_SEC: int = 120

YTDLP_FORMAT: str = "bestaudio/best"
MAX_AUDIO_DURATION_SEC: int = 600

BASS_MIDI_MIN: int = 28
BASS_MIDI_MAX: int = 67

BASS_STRING_TUNINGS = [28, 33, 38, 43]
