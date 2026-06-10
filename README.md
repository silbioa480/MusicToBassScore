# MusicToBassScore

YouTube URL을 입력하면 AI가 자동으로 **베이스 기타 악보(PDF)** 를 생성하는 웹 앱입니다.

J-POP 커버 밴드 베이시스트를 위해 설계되었으며, 오선보 + TAB 형식으로 출력됩니다.

## 기능

- YouTube URL → 오디오 자동 다운로드
- BPM, 조성(Key), 박자 자동 분석
- AI 기반 베이스 트랙 분리 (Meta AI Demucs)
- AI 기반 음표 전사 (Spotify Basic-Pitch)
- 마디별 코드 진행 감지
- 오선보 + TAB 악보 PDF 생성

## 설치

```bash
# 시스템 패키지
sudo apt-get install lilypond ffmpeg

# Python 환경
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Demucs 모델 사전 다운로드 (선택, ~1GB)
python scripts/download_models.py
```

## 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속 후 YouTube URL을 입력하세요.

## 처리 시간

| 환경 | 4분 곡 기준 |
|---|---|
| CPU (일반 노트북) | 약 8~15분 |
| GPU (CUDA) | 약 2~5분 |

> ⚠️ 본 앱은 개인/교육 목적으로만 사용하세요.
