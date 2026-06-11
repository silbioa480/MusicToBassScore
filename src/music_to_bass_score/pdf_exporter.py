"""PDF export of music21 Score using LilyPond."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from music21 import stream

from .config import LILYPOND_BIN, LILYPOND_TIMEOUT_SEC, SCORES_DIR
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExportResult:
    pdf_path: Path
    lily_path: Optional[Path] = None


def check_lilypond_available() -> bool:
    """Return True if LilyPond binary is accessible."""
    try:
        result = subprocess.run(
            [LILYPOND_BIN, "--version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def export_to_pdf(
    score: stream.Score,
    output_dir: Path = SCORES_DIR,
    filename_stem: str = "bass_score",
    method: Literal["lilypond", "musicxml"] = "lilypond",
) -> ExportResult:
    """Render a music21 Score to PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lily_available = check_lilypond_available()
    logger.info("Exporting PDF: method=%s lilypond_available=%s stem=%s", method, lily_available, filename_stem)

    if method == "lilypond" and lily_available:
        return _export_via_lilypond(score, output_dir, filename_stem)
    else:
        if method == "lilypond" and not lily_available:
            logger.warning("LilyPond not found — falling back to MusicXML export")
        return _export_via_musicxml(score, output_dir, filename_stem)


def _export_via_lilypond(
    score: stream.Score,
    output_dir: Path,
    filename_stem: str,
) -> ExportResult:
    ly_path = output_dir / f"{filename_stem}.ly"
    pdf_path = output_dir / f"{filename_stem}.pdf"

    score.write("lily", fp=str(ly_path))

    if not ly_path.exists():
        raise FileNotFoundError(f"music21 did not produce .ly file at {ly_path}")

    result = subprocess.run(
        [LILYPOND_BIN, "-o", str(output_dir / filename_stem), str(ly_path)],
        capture_output=True,
        text=True,
        timeout=LILYPOND_TIMEOUT_SEC,
    )

    if result.returncode != 0:
        logger.error("LilyPond failed (exit %d):\n%s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(
            f"LilyPond failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )

    if not pdf_path.exists():
        matches = list(output_dir.glob(f"{filename_stem}*.pdf"))
        if matches:
            pdf_path = matches[0]
        else:
            raise FileNotFoundError(
                f"LilyPond ran successfully but no PDF found in {output_dir}"
            )

    logger.info("PDF exported via LilyPond: %s (%dKB)", pdf_path, pdf_path.stat().st_size // 1024)
    return ExportResult(pdf_path=pdf_path, lily_path=ly_path)


def _export_via_musicxml(
    score: stream.Score,
    output_dir: Path,
    filename_stem: str,
) -> ExportResult:
    """Fallback: export to MusicXML and convert, or export as MusicXML only."""
    xml_path = output_dir / f"{filename_stem}.musicxml"
    score.write("musicxml", fp=str(xml_path))

    pdf_path = output_dir / f"{filename_stem}.pdf"

    try:
        from music21.converter.subConverters import ConverterMusicXML
        score.write("musicxml.pdf", fp=str(pdf_path))
        if pdf_path.exists():
            return ExportResult(pdf_path=pdf_path, lily_path=None)
    except Exception:
        pass

    return ExportResult(pdf_path=xml_path, lily_path=None)
