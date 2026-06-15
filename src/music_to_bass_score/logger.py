"""Centralized logging configuration for MusicToBassScore.

Usage in any module:
    from .logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
    logger.error("error", exc_info=True)
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "app.log"

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger("music_to_bass_score")
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        # Rotating file handler (10MB × 5 files)
        fh = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(fh)

        # Console handler (INFO and above)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(ch)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped to the given module name."""
    _configure()
    if not name.startswith("music_to_bass_score"):
        name = f"music_to_bass_score.{name.split('.')[-1]}"
    return logging.getLogger(name)
