#!/usr/bin/env python3
"""Chord recognition evaluation: WCSR metrics vs reference annotations.

Usage (timeline inspection — no reference needed):
  python scripts/eval_chords.py --audio song.wav

Usage (A/B comparison with .lab reference):
  python scripts/eval_chords.py --audio song.wav --ref reference.lab
  python scripts/eval_chords.py --audio song.wav --ref reference.lab --harmonic-mix tmp/stems/htdemucs/song_id/

Reference .lab format (Harte notation, space-separated):
  0.000000 2.345000 A
  2.345000 4.690000 A:min7
  4.690000 8.000000 N
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "BTC-ISMIR19"))


# Chord-symbol-to-Harte conversion
_QUALITY_TO_HARTE = {
    "": "maj", "m": "min", "7": "7", "m7": "min7", "maj7": "maj7",
    "dim": "dim", "dim7": "dim7", "m7b5": "hdim7", "+": "aug",
    "sus2": "sus2", "sus4": "sus4", "6": "maj6", "m6": "min6",
    "mM7": "minmaj7",
}


def _symbol_to_harte(sym: str) -> str:
    """Convert our chord symbol (e.g. 'Am7') to Harte notation ('A:min7')."""
    if sym in ("N.C.", "NC", ""):
        return "N"
    if sym.endswith("?"):
        sym = sym[:-1]
    upper = sym.split("/")[0]
    if len(upper) > 1 and upper[1] in ("#", "b"):
        root, quality = upper[:2], upper[2:]
    else:
        root, quality = upper[:1], upper[1:]
    harte_q = _QUALITY_TO_HARTE.get(quality, quality or "maj")
    return f"{root}:{harte_q}" if harte_q != "maj" else root


def _timeline_to_lab(timeline: list, path: str) -> None:
    with open(path, "w") as f:
        for seg in timeline:
            s, e, sym = seg[0], seg[1], seg[2]
            f.write(f"{s:.6f} {e:.6f} {_symbol_to_harte(sym)}\n")


def _build_harmonic_mix(stems_dir: Path, alpha: float, audio_path: Path) -> Path:
    """Create vocals+other+alpha*bass mix; return path to mix file."""
    import numpy as np
    import soundfile as sf

    vocals = stems_dir / "vocals.wav"
    other = stems_dir / "other.wav"
    bass = stems_dir / "bass.wav"

    if not vocals.exists() or not other.exists():
        print(f"WARNING: vocals.wav or other.wav not found in {stems_dir}; using full mix",
              file=sys.stderr)
        return audio_path

    v, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
    o, _ = sf.read(str(other), dtype="float32", always_2d=True)
    b = np.zeros_like(v)
    if bass.exists():
        b, _ = sf.read(str(bass), dtype="float32", always_2d=True)

    n = min(v.shape[0], o.shape[0], b.shape[0])
    mix = v[:n] + o[:n] + alpha * b[:n]

    mix_path = stems_dir / f"harmonic_mix_a{alpha:.2f}.wav"
    sf.write(str(mix_path), mix, sr)
    print(f"Harmonic mix written: {mix_path} ({mix_path.stat().st_size // 1024}KB)")
    return mix_path


def main() -> None:
    p = argparse.ArgumentParser(description="Chord recognition evaluation (WCSR via mir_eval)")
    p.add_argument("--audio", required=True, help="Input audio WAV file")
    p.add_argument("--ref", help="Reference annotation .lab file (Harte format) for WCSR scoring")
    p.add_argument(
        "--harmonic-mix", metavar="STEMS_DIR",
        help="Path to Demucs stems dir (must contain vocals.wav, other.wav, optional bass.wav); "
             "builds and uses a harmonic submix as BTC input instead of the full mix",
    )
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Bass attenuation in harmonic mix (default: 0.3)")
    args = p.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    from music_to_bass_score import btc_chord  # noqa: E402

    if not btc_chord.is_available():
        print("ERROR: BTC model checkpoint not found. Run scripts/setup_btc.py first.",
              file=sys.stderr)
        sys.exit(1)

    btc_input = audio_path
    if args.harmonic_mix:
        btc_input = _build_harmonic_mix(Path(args.harmonic_mix), args.alpha, audio_path)

    print(f"Running BTC on: {btc_input}")
    timeline = btc_chord.recognize_chords(btc_input)
    print(f"Recognised {len(timeline)} chord segment(s)")

    if not args.ref:
        # Pretty-print timeline for manual inspection
        print(f"\n{'Start':>8}  {'End':>8}  {'Chord':<14}  {'Conf':>6}")
        print("-" * 46)
        limit = 80
        for seg in timeline[:limit]:
            s, e, sym = seg[0], seg[1], seg[2]
            conf = f"{seg[3]:.3f}" if len(seg) > 3 else "  n/a"
            print(f"{s:8.3f}  {e:8.3f}  {sym:<14}  {conf:>6}")
        if len(timeline) > limit:
            print(f"  … ({len(timeline) - limit} more segments; add --ref to score)")
        return

    # Compute WCSR against reference annotation
    import mir_eval  # noqa: E402

    est_fd, est_path = tempfile.mkstemp(suffix=".lab")
    try:
        os.close(est_fd)
        _timeline_to_lab(timeline, est_path)

        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(args.ref)
        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels,
            ref_intervals.min(), ref_intervals.max(),
            mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD,
        )
        intervals, ref_l, est_l = mir_eval.util.merge_labeled_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels,
        )
        durations = mir_eval.util.intervals_to_durations(intervals)

        evaluations = {
            "root":   mir_eval.chord.root(ref_l, est_l),
            "triads": mir_eval.chord.triads(ref_l, est_l),
            "majmin": mir_eval.chord.majmin(ref_l, est_l),
            "mirex":  mir_eval.chord.mirex(ref_l, est_l),
        }
        mix_label = f"harmonic mix (α={args.alpha})" if args.harmonic_mix else "full mix"
        print(f"\nWCSR scores [{mix_label}] vs {args.ref}:")
        print(f"  {'Metric':<10}  {'Score':>8}")
        print(f"  {'-'*22}")
        for name, scores in evaluations.items():
            wa = mir_eval.chord.weighted_accuracy(scores, durations)
            print(f"  {name:<10}  {wa:8.4f}  ({wa * 100:.1f}%)")
    finally:
        os.unlink(est_path)


if __name__ == "__main__":
    main()
