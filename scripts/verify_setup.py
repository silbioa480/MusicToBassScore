"""Verify that all dependencies are correctly installed."""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    msg = f"{icon} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main() -> None:
    print("=" * 50)
    print("MusicToBassScore 환경 검증")
    print("=" * 50)
    print()

    results = []

    version = sys.version_info
    results.append(check(
        "Python 버전",
        version >= (3, 11),
        f"{version.major}.{version.minor}.{version.micro} (3.11+ 필요)"
    ))

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    results.append(check("ffmpeg", ffmpeg_ok, "sudo apt-get install ffmpeg" if not ffmpeg_ok else ""))

    lily_ok = shutil.which("lilypond") is not None
    if lily_ok:
        try:
            r = subprocess.run(["lilypond", "--version"], capture_output=True, text=True, timeout=5)
            version_line = r.stdout.splitlines()[0] if r.stdout else ""
            results.append(check("LilyPond", True, version_line))
        except Exception:
            results.append(check("LilyPond", True, "설치됨"))
    else:
        print(f"{WARN} LilyPond — 미설치 (sudo apt-get install lilypond). musicxml fallback 사용됨")

    print()
    print("Python 패키지 검증:")

    packages = [
        ("streamlit", "streamlit"),
        ("yt_dlp", "yt-dlp"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("demucs", "demucs"),
        ("basic_pitch", "basic-pitch"),
        ("music21", "music21"),
        ("pretty_midi", "pretty-midi"),
    ]

    for module_name, pkg_name in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "?")
            results.append(check(f"  {pkg_name}", True, f"v{ver}"))
        except ImportError:
            results.append(check(f"  {pkg_name}", False, f"pip install {pkg_name}"))

    print()
    print("소스 패키지 검증:")
    try:
        from music_to_bass_score.config import PROJECT_ROOT, TMP_DIR
        results.append(check("  music_to_bass_score", True, f"ROOT={PROJECT_ROOT}"))
        results.append(check("  tmp/ 디렉터리", TMP_DIR.exists(), str(TMP_DIR)))
    except ImportError as e:
        results.append(check("  music_to_bass_score", False, str(e)))

    print()
    print("=" * 50)
    failures = [r for r in results if not r]
    if not failures:
        print(f"{PASS} 모든 검증 통과! 앱을 실행할 준비가 되었습니다.")
        print("   → streamlit run app.py")
    else:
        print(f"{FAIL} {len(failures)}개 항목이 실패했습니다. 위 오류를 해결 후 재실행하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
