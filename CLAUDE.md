# MusicToBassScore — Claude Code Project Guide

## Overview
YouTube URL/음원 파일을 입력받아 **코드 차트(코드 + 조표 기준 로마숫자 도수) PDF**를 자동 생성하는 웹 애플리케이션입니다.
J-POP 커버 밴드 연주자를 위해 설계되었으며, 다음 파이프라인으로 동작합니다:

```
입력(URL/파일) → 오디오 다운로드 → 음원 분석(BPM/조성/박자) → 구간별 전조 감지
→ Demucs 4스템 분리 → 하모닉 서브믹스 → BTC 코드 감지 → 도수 보정/변환 → 코드 차트 생성 → PDF 출력
```

사용자는 Streamlit 웹 UI에서 입력하고, 처리 완료 후 **단일 코드 차트**(마디별 코드 1~2개 + 조표 기준 로마숫자 도수) PDF를 다운로드합니다.

> **버전 안내**
> - **v1.0** (`v1.0` 태그 / `claude/youtube-bass-tab-generator-w6av93` 브랜치): 베이스 오선보 + TAB. Demucs 베이스 분리 + Basic-Pitch 전사 기반.
> - **v2 (`claude/chord-analysis`)**: 코드 확인용 차트. 베이스 노트 전사 한계로 목적을 코드/도수 분석으로 전환. 코드 인식은 **BTC Transformer**(전체음원) 사용.
> - **v2.7**: 정확도 향상을 위해 **Demucs(`htdemucs`) 베이스 분리 재도입** — 코드는 전체음원 BTC, 베이스 음(전위 슬래시)은 깨끗한 분리 스템에서 추출. Basic-Pitch 전사 미사용.
> - **v2.8**: **전조(Key Modulation) 감지** — 슬라이딩 윈도우 크로마로 구간별 조표를 추정하고, 각 마디의 조표에 맞는 로마숫자 도수를 표기. 조표 변화 마디에 `[Key: B]` 마커 삽입.
> - **현재(v2.9)**: **코드 인식 정확도 개선** (ADR Option B — 파이프라인 수정). ① 검증 인프라(`scripts/eval_chords.py`, mir_eval WCSR), ② **하모닉 서브믹스**(vocals+other+0.3·bass, drums 제거)를 BTC 입력으로 사용해 베이스 기음 오염 완화, ③ **비트 동기화 풀링**(마디 경계를 BTC 세그먼트 경계에 ±0.5비트 스냅 + 마디 중앙 50% 경계 기준 2코드 판정), ④ **BTC 소프트맥스 신뢰도** 노출 → 저신뢰 코드에 `?` 마커, ⑤ **코드 진행 기반 조표 보정**(평행 조 혼동 해결). Demucs는 이제 4개 스템을 모두 저장(추가 연산 없음).

---

## Tech Stack

| 역할 | 라이브러리 | 비고 |
|---|---|---|
| Web UI | Streamlit 1.35+ | 로컬 실행 또는 Streamlit Cloud 배포 |
| YouTube 다운로드 | yt-dlp | ffmpeg 필요, 다중 player client 폴백 |
| 오디오 분석 | librosa 0.10+ | BPM, 조성, 박자, 전조 감지, 크로마 |
| 음원 분리 | Demucs 4.0 (htdemucs) | Meta AI, ~1GB 모델, 4스템 저장 |
| 코드 인식 | BTC Transformer (ISMIR'19) | `third_party/`에 vendored, large-voca 170코드 |
| 코드 평가 | mir_eval 0.6+ | WCSR 정확도 측정 + 라벨 매핑 |
| 음악 이론/악보 | music21 9.1+ | MIT, `romanNumeralFromChord` |
| PDF 렌더링 | LilyPond (시스템 패키지) | subprocess 호출 |

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

# 3. BTC 코드 인식 모델 준비 (필수, ~24MB — third_party/에 clone + 호환 패치)
python scripts/setup_btc.py

# 4. (선택) Demucs 모델 사전 다운로드 (~1GB, 첫 실행 전 권장)
python scripts/download_models.py

# 5. 설치 검증
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

# 코드 인식 정확도 평가 (WCSR / A·B 비교)
python scripts/eval_chords.py --audio song.wav                       # 타임라인 확인
python scripts/eval_chords.py --audio song.wav --ref reference.lab   # WCSR 점수
python scripts/eval_chords.py --audio song.wav --ref reference.lab \
    --harmonic-mix tmp/stems/htdemucs/<id>/                          # 서브믹스 A/B
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
│       ├── logger.py                  # 로깅 설정 (logs/app.log)
│       ├── downloader.py              # YouTube 오디오 다운로드 + 메타데이터
│       ├── analyzer.py                # BPM/조성/박자 + 전조 감지 + 도수 보정 (librosa)
│       ├── separator.py               # Demucs 4스템 분리 (bass + 하모닉 서브믹스용)
│       ├── btc_chord.py               # BTC Transformer 래퍼 (코드 + 신뢰도)
│       ├── chord_detector.py          # 마디별 코드 진행 감지 (BTC + 전위 슬래시 + ?)
│       ├── roman_numeral.py           # 코드 → 조표 기준 로마숫자 도수
│       ├── score_builder.py           # music21 코드 차트 구성
│       ├── pdf_exporter.py            # LilyPond PDF 렌더링
│       ├── transcriber.py             # (v1 전용) Basic-Pitch 전사 — 미사용
│       └── pipeline.py                # 전체 파이프라인 오케스트레이터
│
├── tests/                             # pytest 테스트
│   ├── conftest.py                    # 공유 픽스처
│   └── test_*.py
│
├── scripts/
│   ├── setup_btc.py                   # BTC 모델 clone + 호환 패치 (필수)
│   ├── download_models.py             # Demucs 모델 사전 다운로드
│   ├── eval_chords.py                 # 코드 인식 정확도 평가 (mir_eval WCSR)
│   └── verify_setup.py               # 환경 검증
│
├── third_party/                       # vendored BTC 모델 (.gitignore, setup_btc.py로 fetch)
│   └── BTC-ISMIR19/
│
├── assets/
│   └── sample_bass.wav               # 테스트용 짧은 오디오 샘플
│
└── tmp/                               # 런타임 임시 파일 (.gitignore)
    ├── audio/                         # 다운로드된 WAV
    ├── stems/                         # Demucs 4스템 + 하모닉 서브믹스
    └── scores/                        # 생성된 PDF
```

---

## Module Responsibilities

| 모듈 | 역할 |
|---|---|
| `config.py` | 모든 상수, 경로, 모델명(`DEMUCS_MODEL`), 임계값 정의 |
| `logger.py` | `logs/app.log` 로테이션 로깅 설정 |
| `downloader.py` | yt-dlp로 YouTube → WAV 변환, 메타데이터 추출. `_PLAYER_CLIENTS` 폴백으로 HTTP 403 회피, 업로드 파일명 경로 traversal 방어 |
| `analyzer.py` | librosa로 BPM/조성/박자 분석 + 정속 마디 그리드(`build_measure_grid`, `detect_first_onset`) + 구간별 전조 감지(`detect_key_per_section`, `_score_key`) + 코드 진행 기반 조표 보정(`refine_key_with_chords`) |
| `btc_chord.py` | BTC large-voca Transformer 래퍼. `recognize_chords`는 `(start, end, symbol, confidence)` 4-튜플 타임라인 반환(softmax 확률 노출). `chord_at_window`는 신뢰도 가중 겹침으로 코드 선택 |
| `chord_detector.py` | BTC로 코드 감지(적응형 화성 리듬: 마디당 1~2개). 마디 경계를 BTC 세그먼트 경계에 ±0.5비트 스냅(`_snap_to_btc_boundary`), 마디 중앙 50% 경계 기준 2코드 판정. **Demucs 베이스 스템**으로 전위 슬래시(G/B 등) 표기. 저신뢰(<0.35) 코드에 `?` 마커. 크로마 템플릿 매칭은 폴백 |
| `roman_numeral.py` | 코드 심볼 → 조표 기준 로마숫자 도수 변환 (music21 `romanNumeralFromChord`). `measures_to_roman`은 마디별 조표 리스트 지원, `?` 마커 제거 후 분석 |
| `score_builder.py` | `build_chord_chart`: 코드 차트 music21 객체 구성. 전조 마디에 `[Key: B]` TextExpression 삽입 |
| `pdf_exporter.py` | LilyPond로 코드 차트 렌더링 (`_chart_to_ly`). `[Key:]` 마커 + `?` 저신뢰 마커(작은 회색) 렌더링 |
| `pipeline.py` | 파이프라인 오케스트레이터 (다운로드→분석→전조→분리→하모닉 서브믹스(`_build_harmonic_mix`)→코드 감지→도수 보정→PDF) |
| `separator.py` | Demucs(`htdemucs`)로 4스템 분리 → `SeparationResult`(bass/vocals/other/drums). 깨끗한 베이스(전위 표기) + 하모닉 서브믹스(BTC 입력)용. 실패 시 graceful fallback. `separate_bass_cached`로 결과 캐시 |
| `transcriber.py` | (v1 전용) Basic-Pitch 전사 — v2 차트 흐름에서는 미사용 |

---

## Architecture: Data Flow

```
YouTube URL / 음원 파일
    │
    ▼ downloader.py (또는 _ensure_wav)
SongMetadata + tmp/audio/<id>.wav
    │
    ├──▶ analyzer.py ──▶ AudioAnalysis (BPM, key, time_sig) + measure_grid
    │                  └▶ detect_key_per_section ──▶ key_labels (마디별 조표)
    │
    ├──▶ separator.py ──▶ tmp/stems/htdemucs/<id>/{bass,vocals,other,drums}.wav
    │                          │
    │                          ├▶ bass.wav ──────────────┐ (전위 슬래시용 깨끗한 베이스)
    │                          └▶ _build_harmonic_mix ──┐ │ (vocals+other+0.3·bass)
    │                                                   │ │
    ▼ chord_detector.py ◀──────────────────────────────┘ ┘
[chord_label per measure] (BTC 코드 + 신뢰도 + 적응형 1~2개 + 전위 슬래시 + ?)
    │
    ▼ analyzer.refine_key_with_chords (코드 진행으로 평행 조 보정)
    ▼ roman_numeral.measures_to_roman (마디별 조표 기준 도수)
    │
    ▼ score_builder.py
music21.stream.Score (코드 차트 + [Key:] 마커)
    │
    ▼ pdf_exporter.py (LilyPond)
tmp/scores/<title>.pdf
    │
    ▼ app.py
st.download_button → 사용자 브라우저
```

---

## Output Format

생성되는 코드 차트 PDF 구성 요소:
- **헤더**: 곡 제목, 아티스트, 조성, 박자, BPM
- **코드 차트**: 마디별 박스 그리드 (한 줄 4마디)
  - 각 마디 위에 코드 심볼 (예: `Am`, `F`, `G/B`) — 적응형 마디당 1~2개
  - 각 마디 아래에 조표 기준 로마숫자 도수 (예: `vi`, `IV`, `I/III`)
  - **전조 마커**: 조표 변화 마디에 `[Key: B]` 표시 + 해당 구간 도수 재계산
  - **신뢰도 마커**: BTC 저신뢰 코드에 작은 회색 `?` (청취 확인 권장)

---

## Known Limitations

| 항목 | 내용 |
|---|---|
| 처리 시간 | 4분 곡 기준 CPU에서 약 8~15분 (Demucs가 대부분) |
| 조성 정확도 | ~75~80%. 코드 진행 보정으로 평행 조 혼동은 일부 개선되나 복잡한 전조는 부정확할 수 있음 |
| 코드 인식 정확도 | BTC 기반. 베이스 기음 오염은 하모닉 서브믹스로 완화되나 화성이 복잡한 구간은 오류 가능. 저신뢰 구간은 `?`로 표시 |
| 마디 그리드 드리프트 | 정속 그리드 + BTC 경계 스냅으로 보정하나 BPM 오차가 크면 긴 곡 후반부에서 어긋날 수 있음 |
| BTC 모델 없음 | `scripts/setup_btc.py` 미실행 시 크로마 템플릿 매칭으로 폴백 (정확도 크게 저하) |
| 저작권 | 개인/교육 목적으로만 사용. YouTube 이용약관 준수 필요 |
| LilyPond 미설치 | music21 MusicXML fallback 사용 시 PDF 품질 저하 |
| GPU 없는 환경 | Demucs/BTC가 CPU로 폴백, 처리 속도 크게 느려짐 |

---

## Troubleshooting

```
BTC 모델 없음:      python scripts/setup_btc.py
Demucs 모델 없음:   python scripts/download_models.py
LilyPond 없음:      sudo apt-get install lilypond
ffmpeg 없음:        sudo apt-get install ffmpeg
CUDA OOM:           Demucs/BTC가 자동으로 CPU로 폴백됨
YouTube 403:        downloader가 player client를 순차 폴백 (android→ios→tv→web)
```
