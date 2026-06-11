"""Full pipeline orchestrator: YouTube URL → bass guitar PDF sheet music."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from .analyzer import AudioAnalysis, analyze_audio
from .logger import get_logger

logger = get_logger(__name__)
from .chord_detector import detect_chords_per_measure
from .config import AUDIO_DIR, MIDI_DIR, SCORES_DIR, STEMS_DIR
from .downloader import SongMetadata, download_audio
from .pdf_exporter import ExportResult, export_to_pdf
from .score_builder import build_score
from .separator import SeparationResult, separate_bass
from .transcriber import TranscriptionResult, transcribe_bass


@dataclass
class PipelineResult:
    metadata: SongMetadata
    analysis: AudioAnalysis
    separation: SeparationResult
    transcription: TranscriptionResult
    chord_labels: list[str]
    export: ExportResult


ProgressCallback = Callable[[str, float], None]


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

    _cb("음원 분석 중...", 0.15)
    analysis = analyze_audio(song_metadata.audio_path)

    _cb("베이스 트랙 분리 중... (시간이 걸립니다)", 0.25)
    separation = separate_bass(
        audio_path=song_metadata.audio_path,
        output_dir=STEMS_DIR,
        progress_cb=lambda f: _cb("베이스 트랙 분리 중...", 0.25 + f * 0.40),
    )

    _cb("음표 인식 중...", 0.65)
    transcription = transcribe_bass(
        bass_wav_path=separation.bass_path,
        output_dir=MIDI_DIR,
        progress_cb=lambda f: _cb("음표 인식 중...", 0.65 + f * 0.15),
    )

    _cb("코드 진행 분석 중...", 0.80)
    chord_labels = detect_chords_per_measure(
        audio_path=song_metadata.audio_path,
        bpm=analysis.bpm,
        time_sig_num=analysis.time_signature_num,
    )

    _cb("악보 생성 중...", 0.88)
    score = build_score(
        song_metadata=song_metadata,
        analysis=analysis,
        note_events=transcription.note_events,
        chord_labels=chord_labels,
        include_tab=include_tab,
    )

    _cb("PDF 렌더링 중...", 0.95)
    safe_title = _safe_filename(song_metadata.title)
    export = export_to_pdf(
        score=score,
        output_dir=scores_out,
        filename_stem=safe_title,
        method=pdf_method,
    )

    _cb("완료!", 1.0)
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


def _safe_filename(title: str) -> str:
    import re
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80] or "bass_score"
