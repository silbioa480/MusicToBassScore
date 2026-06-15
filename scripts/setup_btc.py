"""Clone and prepare the BTC chord-recognition model (one-time setup).

BTC (Bi-directional Transformer for Chord recognition, ISMIR'19) is the pretrained
deep-learning chord recogniser used by the chord chart pipeline. Its code and the
pretrained checkpoints (~24 MB) live in an external GitHub repo, which we vendor into
`third_party/BTC-ISMIR19/` instead of committing the weights.

This script:
  1. clones https://github.com/jayg996/BTC-ISMIR19 into third_party/ (if missing), and
  2. applies small compatibility patches so the 2019-era code runs on modern
     NumPy (no np.float/np.int aliases) and PyYAML (yaml.load needs a Loader).

Run once after installing requirements:

    python scripts/setup_btc.py
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/jayg996/BTC-ISMIR19"
ROOT = Path(__file__).resolve().parents[1]
BTC_DIR = ROOT / "third_party" / "BTC-ISMIR19"


def _clone() -> None:
    if (BTC_DIR / "btc_model.py").is_file():
        print(f"✓ BTC already present at {BTC_DIR}")
        return
    BTC_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} → {BTC_DIR} …")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(BTC_DIR)],
        check=True,
    )


def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text()
    original = text
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    if text != original:
        path.write_text(text)
        print(f"  patched {path.relative_to(ROOT)}")


def _apply_patches() -> None:
    # PyYAML ≥ 5.1 requires an explicit Loader.
    _patch_file(
        BTC_DIR / "utils" / "hparams.py",
        [(r"yaml\.load\(f\)", "yaml.load(f, Loader=yaml.SafeLoader)")],
    )
    # NumPy ≥ 1.24 removed np.float / np.int / np.bool aliases.
    alias = [
        (r"np\.float\b", "float"),
        (r"np\.int\b", "int"),
        (r"np\.bool\b", "bool"),
    ]
    for rel in ("audio_dataset.py", "utils/chords.py", "utils/transformer_modules.py"):
        f = BTC_DIR / rel
        if f.is_file():
            _patch_file(f, alias)


def main() -> int:
    try:
        _clone()
    except subprocess.CalledProcessError as exc:
        print(f"✗ git clone failed: {exc}", file=sys.stderr)
        print("  The app will fall back to the librosa chroma matcher.", file=sys.stderr)
        return 1
    _apply_patches()
    ckpt = BTC_DIR / "test" / "btc_model_large_voca.pt"
    if ckpt.is_file():
        print(f"✓ BTC ready ({ckpt.stat().st_size // (1024 * 1024)} MB checkpoint)")
        return 0
    print("✗ checkpoint missing after clone", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
