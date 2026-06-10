# MusicToBassScore — Claude Code Project Guide

## Overview
YouTube URL을 입력받아 베이스 기타 악보(PDF)를 자동 생성하는 웹 애플리케이션입니다.
J-POP 커버 밴드 베이시스트를 위해 설계되었으며, 다음 파이프라인으로 동작합니다:

```
YouTube URL → 오디오 다운로드 → 음원 분석 → 베이스 분리(AI) → 음표 전사(AI) → 악보 생성 → PDF 출력
```

사용자는 Streamlit 웹 UI에서 URL을 입력하고, 처리 완료 후 오선보 + TAB 형식의 PDF를 다운로드합니다.

---

## Tech Stack

| 역할 | 라이브러리 | 비고 |
|---|---|---|
| Web UI | Streamlit 1.58+ | 로컬 실행 또는 Streamlit Cloud 배포 |
| YouTube 다운로드 | yt-dlp | ffmpeg 필요 |
| 오디오 분석 | librosa 0.11 | BPM, 조성, 박자 |
| 베이스 분리 | Demucs 4.0 (htdemucs_ft) | Meta AI, ~1GB 모델 |
| 음표 전사 | Basic-Pitch 0.4 | Spotify, MIDI 출력 |
| 음악 이론/악보 | music21 10.3 | MIT |
| PDF 렌더링 | LilyPond (시스템 패키지) | subprocess 호출 |
| MIDI 파싱 | pretty-midi | Basic-Pitch 연동 |

---

## System Prerequisites

Python 설치 전에 시스템 패키지를 먼저 설치합니다:

```bash
# Ubuntu/Debian
sudo apt-get install lilypond ffmpeg

# macOS
brew install lilypond ffmpeg
```

---

## Setup

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 2. Python 의존성 설치 (GPU 환경이면 시간 절약 가능)
pip install -r requirements.txt

# 3. (선택) Demucs 모델 사전 다운로드 (~1GB, 첫 실행 전 권장)
python scripts/download_models.py

# 4. 설치 검증
python scripts/verify_setup.py
```

---

## Key Commands

```bash
# 앱 실행
streamlit run app.py

# 테스트 실행 (단위 테스트만)
pytest tests/ -v -m "not integration"

# 통합 테스트 (실제 YouTube URL 필요, 시간 소요)
pytest tests/ -v -m integration -s

# 커버리지 측정
pytest tests/ --cov=src/music_to_bass_score --cov-report=term-missing

# 코드 스타일 검사
ruff check src/ tests/
```

---

## Project Structure

```
MusicToBassScore/
├── CLAUDE.md                          # 이 파일
├── README.md                          # 사용자용 문서
├── requirements.txt                   # Python 의존성
├── app.py                             # Streamlit 메인 앱 진입점
│
├── src/
│   └── music_to_bass_score/           # 메인 패키지
│       ├── __init__.py
│       ├── config.py                  # 전역 상수 및 경로 설정
│       ├── downloader.py              # YouTube 오디오 다운로드 + 메타데이터
│       ├── analyzer.py                # BPM / 조성 / 박자 분석 (librosa)
│       ├── separator.py               # Demucs 베이스 스템 분리
│       ├── transcriber.py             # Basic-Pitch MIDI 전사
│       ├── chord_detector.py          # 마디별 코드 진행 감지
│       ├── score_builder.py           # music21 악보 구성
│       ├── pdf_exporter.py            # LilyPond PDF 렌더링
│       └── pipeline.py                # 전체 파이프라인 오케스트레이터
│
├── tests/                             # pytest 테스트
│   ├── conftest.py                    # 공유 픽스처
│   └── test_*.py
│
├── scripts/
│   ├── download_models.py             # Demucs 모델 사전 다운로드
│   └── verify_setup.py               # 환경 검증
│
├── assets/
│   └── sample_bass.wav               # 테스트용 짧은 오디오 샘플
│
└── tmp/                               # 런타임 임시 파일 (.gitignore)
    ├── audio/                         # 다운로드된 WAV
    ├── stems/                         # Demucs 출력
    ├── midi/                          # Basic-Pitch MIDI
    └── scores/                        # 생성된 PDF
```

---

## Module Responsibilities

| 모듈 | 역할 |
|---|---|
| `config.py` | 모든 상수, 경로, 모델명, 임계값 정의 |
| `downloader.py` | yt-dlp로 YouTube → WAV 변환, 메타데이터 추출 |
| `analyzer.py` | librosa로 BPM/조성/박자 분석 |
| `separator.py` | Demucs htdemucs_ft로 베이스 스템 분리 |
| `transcriber.py` | Basic-Pitch로 베이스 WAV → MIDI 전사 |
| `chord_detector.py` | 크로마 특성 기반 마디별 코드 감지 |
| `score_builder.py` | music21로 오선보+TAB 악보 객체 구성 |
| `pdf_exporter.py` | LilyPond subprocess로 PDF 렌더링 |
| `pipeline.py` | 모든 모듈을 순서대로 호출하는 오케스트레이터 |

---

## Architecture: Data Flow

```
YouTube URL
    │
    ▼ downloader.py
SongMetadata + tmp/audio/<id>.wav
    │
    ├──▶ analyzer.py ──▶ AudioAnalysis (BPM, key, time_sig)
    │
    ├──▶ separator.py ──▶ tmp/stems/<id>/bass.wav
    │         │
    │         ▼ transcriber.py
    │      tmp/midi/<id>.mid + NoteEvents
    │
    ├──▶ chord_detector.py ──▶ [chord_label per measure]
    │
    ▼ score_builder.py
music21.stream.Score (오선보 + TAB)
    │
    ▼ pdf_exporter.py (LilyPond)
tmp/scores/<id>.pdf
    │
    ▼ app.py
st.download_button → 사용자 브라우저
```

---

## Output Format

생성되는 PDF 악보 구성 요소:
- **헤더**: 곡 제목, 원곡자, BPM, 조성, 박자
- **오선보 (Bass Clef)**: 베이스 음표 (전사된 음정 + 음표 길이)
- **코드 표시**: 각 마디 위에 코드 심볼 (예: Am, F, C, G)
- **TAB 악보**: 오선보 아래에 4현 베이스 기타 줄/프렛 번호

---

## Known Limitations

| 항목 | 내용 |
|---|---|
| 처리 시간 | 4분 곡 기준 CPU에서 약 8~15분 (Demucs가 대부분) |
| 조성 정확도 | ~75~80%. 복잡한 전조가 많은 곡은 부정확할 수 있음 |
| 음표 전사 정확도 | 베이스 분리 품질에 의존. 복잡한 믹스에서 오류 발생 가능 |
| 저작권 | 개인/교육 목적으로만 사용. YouTube 이용약관 준수 필요 |
| LilyPond 미설치 | music21 MusicXML fallback 사용 시 PDF 품질 저하 |
| GPU 없는 환경 | Demucs가 CPU로 폴백, 처리 속도 크게 느려짐 |

---

## Troubleshooting

```
Demucs 모델 없음:   python scripts/download_models.py
LilyPond 없음:      sudo apt-get install lilypond
ffmpeg 없음:        sudo apt-get install ffmpeg
CUDA OOM:           Demucs가 자동으로 CPU로 폴백됨
```
