"""BTC (Bi-directional Transformer for Chord recognition, ISMIR'19) wrapper.

Runs the pretrained large-vocabulary BTC model on an audio file and returns a
time-stamped chord timeline. The model + inference code live under
`third_party/BTC-ISMIR19/` (cloned from https://github.com/jayg996/BTC-ISMIR19).

This replaces the hand-rolled chroma template matcher with a SOTA learned model,
which is far more accurate (correct roots + rich qualities: 7, maj7, min7, dim,
hdim7, sus, aug, …) than chroma matching on a full mix.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .logger import get_logger

logger = get_logger(__name__)

# third_party/BTC-ISMIR19 relative to repo root (this file: src/music_to_bass_score/)
BTC_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "BTC-ISMIR19"
_CKPT = BTC_ROOT / "test" / "btc_model_large_voca.pt"

# Harte quality suffix → music21-parseable chord-symbol suffix.
_QUALITY_MAP = {
    "": "", "maj": "",
    "min": "m",
    "7": "7", "maj7": "maj7", "min7": "m7",
    "maj6": "6", "min6": "m6",
    "dim": "dim", "dim7": "dim7", "hdim7": "m7b5",
    "aug": "+",
    "sus2": "sus2", "sus4": "sus4",
    "minmaj7": "mM7",
}

# Lazily-initialised singletons (model load is ~1-2 s; reuse across calls).
_CACHE: dict = {}


def is_available() -> bool:
    """Return True if the BTC checkpoint and repo are present."""
    return _CKPT.is_file() and (BTC_ROOT / "btc_model.py").is_file()


def _harte_to_symbol(harte: str) -> str:
    """Convert a Harte chord label ('A:min7', 'D', 'N') to our symbol ('Am7', 'D', 'N.C.')."""
    if harte in ("N", "X", ""):
        return "N.C."
    if ":" in harte:
        root, qual = harte.split(":", 1)
    else:
        root, qual = harte, "maj"
    qual = qual.split("/", 1)[0]  # drop inversion, e.g. 'maj7/3'
    return root + _QUALITY_MAP.get(qual, "")


def _load_model():
    if "model" in _CACHE:
        return _CACHE["model"]

    if str(BTC_ROOT) not in sys.path:
        sys.path.insert(0, str(BTC_ROOT))

    import torch
    from btc_model import BTC_model, HParams
    from utils.mir_eval_modules import idx2voca_chord

    config = HParams.load(str(BTC_ROOT / "run_config.yaml"))
    config.feature["large_voca"] = True
    config.model["num_chords"] = 170

    model = BTC_model(config=config.model)
    ckpt = torch.load(str(_CKPT), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    _CACHE["model"] = (model, ckpt["mean"], ckpt["std"], idx2voca_chord(), config, torch)
    logger.info("BTC large-voca model loaded from %s", _CKPT)
    return _CACHE["model"]


def recognize_chords(audio_path: Path) -> list[tuple[float, float, str, float]]:
    """Run BTC inference; return [(start_sec, end_sec, symbol, confidence)] over the whole track.

    `confidence` is the mean max-softmax probability across all timesteps within the segment.
    Higher values (0–1) indicate stronger model certainty about the chord label.
    """
    model, mean, std, idx_to_chord, config, torch = _load_model()
    from utils.mir_eval_modules import audio_file_to_features

    feature, feature_per_second, song_length = audio_file_to_features(str(audio_path), config)
    feature = feature.T
    feature = (feature - mean) / std

    n = config.model["timestep"]
    num_pad = n - (feature.shape[0] % n)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n

    segments: list[tuple[float, float, str, float]] = []
    start_time = 0.0
    prev_idx: int | None = None
    conf_sum = 0.0
    conf_count = 0

    with torch.no_grad():
        ft = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        for t in range(num_instance):
            attn, _ = model.self_attn_layers(ft[:, n * t:n * (t + 1), :])
            # Extract logits directly from the projection layer to get softmax probabilities.
            # model.output_layer returns argmax indices; we need probabilities for confidence.
            logits = model.output_layer.output_projection(attn)  # (1, n, num_chords)
            probs = torch.softmax(logits, dim=-1)                # (1, n, num_chords)
            pred_idx = probs.argmax(dim=-1).squeeze(0)           # (n,) — argmax per frame
            pred_conf = probs.max(dim=-1).values.squeeze(0)      # (n,) — max prob per frame

            for i in range(n):
                gidx = n * t + i
                idx = int(pred_idx[i].item())
                conf = float(pred_conf[i].item())

                if t == 0 and i == 0:
                    prev_idx = idx
                    conf_sum = conf
                    conf_count = 1
                    continue

                if idx != prev_idx:
                    mean_conf = conf_sum / max(1, conf_count)
                    segments.append((
                        start_time,
                        feature_per_second * gidx,
                        idx_to_chord[prev_idx],
                        mean_conf,
                    ))
                    start_time = feature_per_second * gidx
                    prev_idx = idx
                    conf_sum = conf
                    conf_count = 1
                else:
                    conf_sum += conf
                    conf_count += 1

                if t == num_instance - 1 and i + num_pad == n:
                    if start_time != feature_per_second * gidx:
                        mean_conf = conf_sum / max(1, conf_count)
                        segments.append((
                            start_time,
                            feature_per_second * gidx,
                            idx_to_chord[prev_idx],
                            mean_conf,
                        ))
                    break

    timeline = [(s, e, _harte_to_symbol(c), conf) for s, e, c, conf in segments]
    logger.info(
        "BTC recognised %d chord segments over %.1fs (first: %s)",
        len(timeline), song_length,
        [c for _, _, c, *_ in timeline[:6]],
    )
    return timeline


def chord_at_window(
    timeline: list[tuple],
    start: float,
    end: float,
) -> tuple[str, float]:
    """Return (symbol, confidence) for the chord with highest confidence-weighted coverage.

    Weights each chord by overlap_seconds × mean_segment_confidence, so a chord that
    fills most of the window with high certainty beats a chord that fills slightly more
    time but with low certainty.
    """
    scores: dict[str, float] = {}
    conf_acc: dict[str, float] = {}
    for seg in timeline:
        s, e, sym = seg[0], seg[1], seg[2]
        conf = seg[3] if len(seg) > 3 else 1.0
        overlap = min(end, e) - max(start, s)
        if overlap > 0.0:
            scores[sym] = scores.get(sym, 0.0) + overlap * conf
            conf_acc[sym] = max(conf_acc.get(sym, 0.0), conf)
    if not scores:
        return "N.C.", 0.0
    best_sym = max(scores, key=scores.__getitem__)
    return best_sym, conf_acc[best_sym]
