"""Bass stem separation using Demucs (htdemucs_ft model)."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import DEMUCS_MODEL, STEMS_DIR


@dataclass
class SeparationResult:
    bass_path: Path
    stems_dir: Path


def separate_bass(
    audio_path: Path,
    output_dir: Path = STEMS_DIR,
    model_name: str = DEMUCS_MODEL,
    device: str = "auto",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> SeparationResult:
    """Separate bass stem from full mix using Demucs."""
    resolved_device = _resolve_device(device)

    if progress_cb:
        progress_cb(0.05)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "bass",
        "-n", model_name,
        "-d", resolved_device,
        "-o", str(output_dir),
        str(audio_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stderr_lines = []
    while True:
        line = process.stderr.readline() if process.stderr else ""
        if not line and process.poll() is not None:
            break
        if line:
            stderr_lines.append(line.rstrip())
            if progress_cb and "%" in line:
                pct = _parse_progress_pct(line)
                if pct is not None:
                    progress_cb(0.05 + pct * 0.9)

    returncode = process.wait()
    if returncode != 0:
        error_output = "\n".join(stderr_lines[-20:])
        raise RuntimeError(
            f"Demucs failed (exit {returncode}):\n{error_output}"
        )

    if progress_cb:
        progress_cb(1.0)

    bass_path = _find_bass_output(output_dir, model_name, audio_path.stem)
    return SeparationResult(bass_path=bass_path, stems_dir=output_dir)


def _find_bass_output(output_dir: Path, model_name: str, stem: str) -> Path:
    """Locate the bass.wav file produced by Demucs."""
    candidates = [
        output_dir / model_name / stem / "bass.wav",
        output_dir / model_name / stem / "bass.mp3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(output_dir.rglob("bass.wav"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find Demucs bass output. Searched under: {output_dir}"
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _parse_progress_pct(line: str) -> Optional[float]:
    import re
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
    if match:
        return float(match.group(1)) / 100.0
    return None


def check_model_cached(model_name: str = DEMUCS_MODEL) -> bool:
    """Return True if the Demucs model is already downloaded."""
    try:
        import torch
        hub_dir = Path(torch.hub.get_dir())
        return any(hub_dir.rglob(f"*{model_name}*"))
    except Exception:
        return False
