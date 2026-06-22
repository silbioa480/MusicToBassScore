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
    vocals_path: Optional[Path] = None
    other_path: Optional[Path] = None
    drums_path: Optional[Path] = None


def separate_bass_cached(
    audio_path: Path,
    output_dir: Path = STEMS_DIR,
    model_name: str = DEMUCS_MODEL,
    device: str = "auto",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Optional[SeparationResult]:
    """Separate all stems, reusing a cached result if present.

    Returns a SeparationResult with all available stem paths, or None on failure.
    Callers fall back gracefully when vocals_path / other_path are None.
    """
    stem_dir = output_dir / model_name / audio_path.stem
    bass_path = stem_dir / "bass.wav"
    if bass_path.is_file() and bass_path.stat().st_size > 0:
        logger.info("Reusing cached stems: %s", stem_dir)
        if progress_cb:
            progress_cb(1.0)
        vocals_p = stem_dir / "vocals.wav"
        other_p = stem_dir / "other.wav"
        drums_p = stem_dir / "drums.wav"
        return SeparationResult(
            bass_path=bass_path,
            stems_dir=output_dir,
            vocals_path=vocals_p if vocals_p.is_file() else None,
            other_path=other_p if other_p.is_file() else None,
            drums_path=drums_p if drums_p.is_file() else None,
        )
    try:
        return separate_bass(
            audio_path, output_dir=output_dir, model_name=model_name,
            device=device, progress_cb=progress_cb,
        )
    except Exception as exc:  # noqa: BLE001 — separation is optional, never fatal
        logger.warning("Bass separation failed (%s); chord detection will use full mix", exc)
        return None


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

    stem_dir = output_dir / model_name / audio_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    # Save all stems (Demucs computed them all anyway — only disk I/O added)
    stem_paths: dict[str, Path] = {}
    for stem_name in model.sources:
        stem_idx = model.sources.index(stem_name)
        stem_wav = sources[0, stem_idx].cpu().numpy().T  # (samples, channels)
        stem_path = stem_dir / f"{stem_name}.wav"
        sf.write(str(stem_path), stem_wav, model.samplerate)
        stem_paths[stem_name] = stem_path

    bass_path = stem_paths["bass"]

    if progress_cb:
        progress_cb(1.0)

    size_kb = bass_path.stat().st_size // 1024
    logger.info(
        "Stem separation complete: %s (%dKB bass, stems=%s)",
        stem_dir, size_kb, list(stem_paths.keys()),
    )
    return SeparationResult(
        bass_path=bass_path,
        stems_dir=output_dir,
        vocals_path=stem_paths.get("vocals"),
        other_path=stem_paths.get("other"),
        drums_path=stem_paths.get("drums"),
    )


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
