"""Pre-download Demucs model (~1GB) to avoid delay on first app use."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    print("Demucs 모델 다운로드를 시작합니다 (~1GB, 시간이 걸릴 수 있습니다)...")
    print("모델명: htdemucs_ft\n")

    try:
        from demucs.pretrained import get_model
        model = get_model("htdemucs_ft")
        print(f"\n✅ 모델 다운로드 완료: {type(model).__name__}")
        print("이제 앱을 실행할 준비가 되었습니다: streamlit run app.py")
    except Exception as e:
        print(f"❌ 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
