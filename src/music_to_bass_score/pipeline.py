"""Full pipeline orchestrator: YouTube URL → bass guitar PDF sheet music."""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from .analyzer import AudioAnalysis, analyze_audio
from .chord_detector import detect_chords_per_measure
from .config import AUDIO_DIR, MIDI_DIR, SAMPLE_RATE, SCORES_DIR, STEMS_DIR
from .downloader import SongMetadata, download_audio
from .logger import get_logger
from .pdf_exporter import ExportResult, export_to_pdf
from .score_builder import build_score
from .separator import SeparationResult, separate_bass
from .transcriber import TranscriptionResult, transcribe_bass

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    metadata: SongMetadata
    analysis: AudioAnalysis
    separation: SeparationResult
    transcription: TranscriptionResult
    chord_labels: list[str]
    export: ExportResult


ProgressCallback = Callable[[str, float], None]

_SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def run_pipeline(
    youtube_url: str,
    output_dir: Optional[Path] = None,
    include_tab: bool = True,
    pdf_method: Literal["lilypond", "musicxml"] = "lilypond",
    progress_cb: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """Run the full YouTube → bass score pipeline.

    progress_cb receives (status_message, fraction_0_to_1) at each stage.
    """
    def _cb(msg: str, frac: float) -> None:
        logger.debug("Pipeline progress [%.0f%%]: %s", frac * 100, msg)
        if progress_cb:
            progress_cb(msg, frac)

    scores_out = output_dir or SCORES_DIR
    logger.info("Pipeline started: url=%s tab=%s method=%s", youtube_url, include_tab, pdf_method)

    _cb("오디오 다운로드 중...", 0.0)
    song_metadata = download_audio(
        url=youtube_url,
        output_dir=AUDIO_DIR,
        progress_cb=lambda f: _cb("오디오 다운로드 중...", f * 0.15),
    )

    return _run_from_metadata(
        song_metadata=song_metadata,
        scores_out=scores_out,
        include_tab=include_tab,
        pdf_method=pdf_method,
        cb=_cb,
        start_frac=0.15,
    )


def run_pipeline_from_file(
    audio_path: Path,
    title: str = "",
    artist: str = "",
    output_dir: Optional[Path] = None,
    include_tab: bool = True,
    pdf_method: Literal["lilypond", "musicxml"] = "lilypond",
    progress_cb: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """Run the pipeline starting from a local audio file (skip YouTube download).

    Accepts WAV, MP3, FLAC, OGG, M4A, AAC, OPUS.
    Non-WAV files are converted to WAV via ffmpeg before processing.
    """
    def _cb(msg: str, frac: float) -> None:
        logger.debug("Pipeline(file) progress [%.0f%%]: %s", frac * 100, msg)
        if progress_cb:
            progress_cb(msg, frac)

    scores_out = output_dir or SCORES_DIR
    logger.info(
        "Pipeline(file) started: path=%s title=%r artist=%r",
        audio_path, title, artist,
    )

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = audio_path.suffix.lower()
    if ext not in _SUPPORTED_AUDIO_EXTS:
        raise ValueError(
            f"Unsupported audio format: {ext}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_AUDIO_EXTS))}"
        )

    _cb("오디오 파일 준비 중...", 0.0)
    wav_path = _ensure_wav(audio_path, _cb)

    import soundfile as sf
    info = sf.info(str(wav_path))
    duration_sec = info.duration

    song_metadata = SongMetadata(
        title=title or audio_path.stem,
        artist=artist or "Unknown Artist",
        duration_sec=duration_sec,
        youtube_url="",
        audio_path=wav_path,
    )
    logger.info(
        "File metadata: title=%r artist=%r duration=%.1fs",
        song_metadata.title, song_metadata.artist, duration_sec,
    )

    return _run_from_metadata(
        song_metadata=song_metadata,
        scores_out=scores_out,
        include_tab=include_tab,
        pdf_method=pdf_method,
        cb=_cb,
        start_frac=0.08,
    )


def _run_from_metadata(
    song_metadata: SongMetadata,
    scores_out: Path,
    include_tab: bool,
    pdf_method: str,
    cb: Callable,
    start_frac: float,
) -> PipelineResult:
    """Shared pipeline stages: analysis → separation → transcription → score → PDF."""

    cb("음원 분석 중...", start_frac)
    analysis = analyze_audio(song_metadata.audio_path)

    cb("베이스 트랙 분리 중... (시간이 걸립니다)", 0.25)
    separation = separate_bass(
        audio_path=song_metadata.audio_path,
        output_dir=STEMS_DIR,
        progress_cb=lambda f: cb("베이스 트랙 분리 중...", 0.25 + f * 0.40),
    )

    cb("음표 인식 중...", 0.65)
    transcription = transcribe_bass(
        bass_wav_path=separation.bass_path,
        output_dir=MIDI_DIR,
        progress_cb=lambda f: cb("음표 인식 중...", 0.65 + f * 0.15),
    )

    cb("코드 진행 분석 중...", 0.80)
    chord_labels = detect_chords_per_measure(
        audio_path=separation.bass_path,   # bass stem: far cleaner root detection than full mix
        bpm=analysis.bpm,
        time_sig_num=analysis.time_signature_num,
        beat_times=analysis.beat_times,
    )

    cb("악보 생성 중...", 0.88)
    score = build_score(
        song_metadata=song_metadata,
        analysis=analysis,
        note_events=transcription.note_events,
        chord_labels=chord_labels,
        include_tab=include_tab,
    )

    cb("PDF 렌더링 중...", 0.95)
    safe_title = _safe_filename(song_metadata.title)
    export = export_to_pdf(
        score=score,
        output_dir=scores_out,
        filename_stem=safe_title,
        method=pdf_method,
    )

    cb("완료!", 1.0)
    logger.info(
        "Pipeline complete: title=%r pdf=%s",
        song_metadata.title, export.pdf_path,
    )

    return PipelineResult(
        metadata=song_metadata,
        analysis=analysis,
        separation=separation,
        transcription=transcription,
        chord_labels=chord_labels,
        export=export,
    )


def _ensure_wav(audio_path: Path, cb: Callable) -> Path:
    """Return a WAV version of the audio file, converting via librosa+soundfile if needed."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    wav_path = AUDIO_DIR / (audio_path.stem + ".wav")
    if wav_path.exists():
        logger.info("WAV cache hit: %s", wav_path)
        return wav_path

    logger.info("Converting %s → %s via librosa", audio_path.suffix, wav_path)
    cb("오디오 형식 변환 중...", 0.03)

    try:
        import librosa
        import soundfile as sf
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=False)
        if y.ndim == 1:
            y = y[None, :]
        sf.write(str(wav_path), y.T, sr, subtype="PCM_16")
        logger.info("Conversion complete: %s (%dKB)", wav_path, wav_path.stat().st_size // 1024)
        return wav_path
    except Exception as exc:
        logger.warning("librosa conversion failed (%s), trying ffmpeg fallback", exc)

    cb("오디오 형식 변환 중 (ffmpeg)...", 0.04)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install it: sudo apt-get install ffmpeg")

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", str(SAMPLE_RATE), "-ac", "2",
            "-f", "wav", str(wav_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        logger.error("ffmpeg conversion failed:\n%s", result.stderr[-1000:])
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr[-500:]}")

    logger.info("Conversion complete: %s (%dKB)", wav_path, wav_path.stat().st_size // 1024)
    return wav_path


def _safe_filename(title: str) -> str:
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80] or "bass_score"
