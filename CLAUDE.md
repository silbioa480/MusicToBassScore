# MusicToBassScore — Claude Code Project Guide

## Overview
YouTube URL/음원 파일을 입력받아 **코드 차트(코드 + 조표 기준 로마숫자 도수) PDF**를 자동 생성하는 웹 애플리케이션입니다.
J-POP 커버 밴드 연주자를 위해 설계되었으며, 다음 파이프라인으로 동작합니다:

```
입력(URL/파일) → 오디오 다운로드 → 음원 분석(BPM/조성/박자) → 코드 감지(전체음원) → 로마숫자 도수 변환 → 코드 차트 생성 → PDF 출력
```

사용자는 Streamlit 웹 UI에서 입력하고, 처리 완료 후 **단일 오선보 코드 차트**(마디별 코드 2개 + 로마숫자 도수) PDF를 다운로드합니다.

> **버전 안내**
> - **v1.0** (`v1.0` 태그 / `claude/youtube-bass-tab-generator-w6av93` 브랜치): 베이스 오선보 + TAB. Demucs 베이스 분리 + Basic-Pitch 전사 기반.
> - **v2 (`claude/chord-analysis`)**: 코드 확인용 차트. 베이스 노트 전사 한계로 목적을 코드/도수 분석으로 전환. 코드 인식은 **BTC Transformer**(전체음원) 사용.
> - **현재(v2.7)**: 정확도 향상을 위해 **Demucs(`htdemucs`) 베이스 분리 재도입** — 코드는 전체음원 BTC, 베이스 음(전위 슬래시)은 깨끗한 분리 스템에서 추출. 분리로 처리 시간이 4분 곡 기준 수 분으로 늘어남(정확도 우선). Basic-Pitch 전사는 여전히 미사용.

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
| `analyzer.py` | librosa로 BPM/조성/박자 분석 + 정속 마디 그리드(`build_measure_grid`, `detect_first_onset`) |
| `chord_detector.py` | **전체음원 BTC Transformer**로 코드 감지(적응형 화성 리듬: 마디당 1~2개). **Demucs 베이스 스템**으로 깨끗한 베이스 음을 얻어 전위 슬래시(G/B 등) 표기. 크로마 템플릿 매칭은 폴백 |
| `roman_numeral.py` | 코드 심볼 → 조표 기준 로마숫자 도수 변환 (music21 `romanNumeralFromChord` 기반) |
| `score_builder.py` | `build_chord_chart`: 단일 오선보 코드 차트 music21 객체 구성 |
| `pdf_exporter.py` | LilyPond로 단일 스태프 차트 렌더링 (`_chart_to_ly`) |
| `pipeline.py` | 차트 파이프라인 오케스트레이터 (다운로드→분석→베이스 분리→코드 감지→도수→PDF) |
| `separator.py` | Demucs(`htdemucs`, bag-of-1)로 베이스 스템 분리 → 정확한 베이스 음/전위 표기용. 실패 시 전체음원 저역 크로마로 graceful fallback. `separate_bass_cached`로 결과 캐시 |
| `transcriber.py` | (v1 전용) Basic-Pitch 전사 — v2 차트 흐름에서는 미사용 |

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
    ├──▶ separator.py ──▶ tmp/stems/htdemucs/<id>/bass.wav (깨끗한 베이스)
    │                              │
    ▼ chord_detector.py ◀──────────┘ (코드=전체음원 BTC, 베이스음=분리 스템)
[chord_label per measure] (적응형 마디당 1~2개 + 전위 슬래시)
    │
    ▼ score_builder.py
music21.stream.Score (코드 차트)
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
