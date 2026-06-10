"""YouTube audio download and metadata extraction using yt-dlp."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

from .config import AUDIO_DIR, MAX_AUDIO_DURATION_SEC, SAMPLE_RATE, YTDLP_FORMAT


@dataclass
class SongMetadata:
    title: str
    artist: str
    duration_sec: float
    youtube_url: str
    audio_path: Path


def validate_youtube_url(url: str) -> bool:
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+",
        r"(?:https?://)?youtu\.be/[\w-]+",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+",
    ]
    return any(re.match(p, url.strip()) for p in patterns)


def _extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]+)", url)
    return match.group(1) if match else "unknown"


def download_audio(
    url: str,
    output_dir: Path = AUDIO_DIR,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> SongMetadata:
    """Download audio from YouTube URL and return metadata with path to WAV file."""
    if not validate_youtube_url(url):
        raise ValueError(f"Invalid YouTube URL: {url}")

    video_id = _extract_video_id(url)
    output_path = output_dir / video_id
    wav_path = output_dir / f"{video_id}.wav"

    if wav_path.exists():
        info = _fetch_info(url)
        return SongMetadata(
            title=info.get("title", "Unknown Title"),
            artist=info.get("uploader", "Unknown Artist"),
            duration_sec=float(info.get("duration", 0)),
            youtube_url=url,
            audio_path=wav_path,
        )

    def _progress_hook(d: dict) -> None:
        if progress_cb is None:
            return
        if d["status"] == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes") or d.get("total_bytes_estimate", 1)
            progress_cb(downloaded / total if total else 0.0)
        elif d["status"] == "finished":
            progress_cb(1.0)

    ydl_opts = {
        "format": YTDLP_FORMAT,
        "outtmpl": str(output_path),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "postprocessor_args": ["-ar", str(SAMPLE_RATE)],
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(f"Failed to extract info from: {url}")

    duration = float(info.get("duration", 0))
    if duration > MAX_AUDIO_DURATION_SEC:
        wav_path.unlink(missing_ok=True)
        raise ValueError(
            f"Video duration {duration:.0f}s exceeds limit of {MAX_AUDIO_DURATION_SEC}s"
        )

    title = info.get("title", "Unknown Title")
    uploader = info.get("uploader", info.get("channel", "Unknown Artist"))

    return SongMetadata(
        title=title,
        artist=uploader,
        duration_sec=duration,
        youtube_url=url,
        audio_path=wav_path,
    )


def _fetch_info(url: str) -> dict:
    ydl_opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info or {}
