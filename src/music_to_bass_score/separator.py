"""Bass stem separation using Demucs Python API (htdemucs_ft model).

Uses the Python API directly (not subprocess) to avoid ffprobe/libcaca dependency.
Audio I/O uses soundfile, which supports WAV natively without system codecs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from .config import DEMUCS_MODEL, SAMPLE_RATE, STEMS_DIR
from .logger import get_logger

logger = get_logger(__name__)


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
    """Separate bass stem from full mix using Demucs Python API."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import julius

    resolved_device = _resolve_device(device)
    logger.info("Separating bass: %s (model=%s device=%s)", audio_path, model_name, resolved_device)

    if progress_cb:
        progress_cb(0.05)

    logger.debug("Loading Demucs model: %s", model_name)
    if progress_cb:
        progress_cb(0.10)
    model = get_model(model_name)
    model.eval()
    model.to(resolved_device)

    if progress_cb:
        progress_cb(0.20)

    wav_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    wav_np = wav_np.T  # (channels, samples)

    if wav_np.shape[0] == 1:
        wav_np = np.repeat(wav_np, 2, axis=0)

    wav = torch.from_numpy(wav_np).float()

    if sr != model.samplerate:
        wav = julius.resample_frac(wav, sr, model.samplerate)

    ref = wav.mean()
    std = wav.std().clamp(min=1e-8)
    wav_norm = (wav - ref) / std

    if progress_cb:
        progress_cb(0.30)

    def _demucs_progress(progress: float) -> None:
        if progress_cb:
            progress_cb(0.30 + progress * 0.60)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav_norm.unsqueeze(0).to(resolved_device),
            device=resolved_device,
            progress=False,
            num_workers=0,
        )

    if progress_cb:
        progress_cb(0.92)

    sources = sources * std + ref

    bass_idx = model.sources.index("bass")
    bass_wav = sources[0, bass_idx].cpu().numpy()  # (channels, samples)

    stem_dir = output_dir / model_name / audio_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    bass_path = stem_dir / "bass.wav"

    bass_wav_T = bass_wav.T  # (samples, channels)
    sf.write(str(bass_path), bass_wav_T, model.samplerate)

    if progress_cb:
        progress_cb(1.0)

    size_kb = bass_path.stat().st_size // 1024
    logger.info("Bass separation complete: %s (%dKB)", bass_path, size_kb)
    return SeparationResult(bass_path=bass_path, stems_dir=output_dir)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def check_model_cached(model_name: str = DEMUCS_MODEL) -> bool:
    """Return True if the Demucs model is already downloaded."""
    try:
        import torch
        hub_dir = Path(torch.hub.get_dir())
        return any(hub_dir.rglob(f"*{model_name}*"))
    except Exception:
        return False
