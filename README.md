# MusicToBassScore

YouTube URL을 입력하면 AI가 자동으로 **베이스 기타 악보(PDF)** 를 생성하는 웹 앱입니다.

J-POP 커버 밴드 베이시스트를 위해 설계되었으며, **오선보 + TAB** 형식으로 출력됩니다.

---

## 처리 파이프라인

```
YouTube URL
    ↓ yt-dlp
오디오 다운로드 (WAV)
    ↓ librosa
BPM / 조성 / 박자 분석
    ↓ Demucs (htdemucs_ft)
베이스 트랙 분리
    ↓ Basic-Pitch
음표 → MIDI 전사
    ↓ music21 + LilyPond
PDF 악보 생성 (오선보 + TAB + 코드)
```

---

## 사용 라이브러리

### Web UI

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `streamlit` | ≥1.35.0 | 웹 UI 프레임워크 — URL 입력, 진행 표시, PDF 다운로드 버튼 제공 |

### YouTube 다운로드

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `yt-dlp` | ≥2024.5.0 | YouTube 영상에서 오디오를 추출하고 곡 제목·아티스트 메타데이터를 수집 |

### 오디오 처리 및 분석

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `librosa` | ≥0.10.2 | BPM(템포), 조성(Key), 박자(Time Signature) 분석 및 크로마 특성 추출 |
| `soundfile` | ≥0.12.1 | WAV 파일 읽기/쓰기 (ffprobe 없이 순수 Python으로 처리) |
| `numpy` | ≥1.26, <2.3 | 오디오 배열 연산의 핵심 수치 계산 라이브러리 |
| `scipy` | ≥1.13.0 | 신호 처리 보조 연산 |
| `resampy` | ≥0.2.2, <0.4.3 | 오디오 샘플레이트 변환 (librosa·basic-pitch 내부 사용) |
| `matplotlib` | ≥3.8.0 | 오디오 파형·스펙트럼 시각화 (librosa 의존) |

### AI 베이스 트랙 분리

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `demucs` | ≥4.0.1 | Meta AI의 음원 분리 모델. `htdemucs_ft`로 베이스 트랙만 추출 |
| `torch` | ≥2.2.0 | Demucs 모델 구동을 위한 딥러닝 프레임워크 (CPU/GPU 자동 선택) |
| `torchaudio` | ≥2.2.0 | Demucs 내부 오디오 처리 (torch 버전과 반드시 일치해야 함) |

### 음표 전사 (Audio → MIDI)

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `basic-pitch` | ≥0.3.0 | Spotify 개발 신경망 기반 음표 전사 도구. 베이스 WAV → MIDI 변환 |
| `pretty-midi` | ≥0.2.9 | Basic-Pitch가 출력하는 MIDI 데이터 파싱 및 `.mid` 파일 저장 |

### 악보 생성 및 PDF 출력

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `music21` | ≥9.1.0 | 악보 구조 구성 — 오선보(Bass Clef), TAB, 코드 기호, 박자·조표 배치 |
| `requests` | ≥2.31.0 | music21 내부 의존 라이브러리 |

> **시스템 패키지도 필요합니다**  
> `LilyPond` — music21이 생성한 `.ly` 파일을 고품질 PDF로 렌더링  
> `ffmpeg` — yt-dlp의 오디오 변환 후처리 (MP4/webm → WAV)

---

## 시스템 요구 사항

- Python 3.11+
- Ubuntu/Debian: `sudo apt-get install lilypond ffmpeg`
- macOS: `brew install lilypond ffmpeg`

---

## 설치

```bash
# 1. 저장소 클론
git clone https://github.com/silbioa480/MusicToBassScore.git
cd MusicToBassScore

# 2. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Python 의존성 설치
pip install -r requirements.txt

# 4. Demucs 모델 사전 다운로드 (선택, ~1GB)
python scripts/download_models.py

# 5. 환경 검증
python scripts/verify_setup.py
```

---

## 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속 후 YouTube URL을 입력하세요.

---

## 처리 시간

| 환경 | 4분 곡 기준 예상 시간 |
|---|---|
| CPU (일반 노트북) | 약 8~15분 |
| GPU (CUDA) | 약 2~5분 |

Demucs 베이스 분리가 전체 시간의 약 70~80%를 차지합니다.

---

## 로그

실행 로그는 `logs/app.log`에 기록됩니다. 오류 발생 시 이 파일을 확인하세요.

```
logs/
└── app.log   # 타임스탬프·모듈명·라인번호 포함, 최대 10MB × 5회 로테이션
```

---

## 테스트

```bash
# 단위 테스트 (네트워크·GPU 불필요)
pytest tests/ -v -m "not integration"

# 통합 테스트 (실제 YouTube URL 필요)
pytest tests/ -v -m integration -s

# 커버리지 측정
pytest tests/ --cov=src/music_to_bass_score --cov-report=term-missing
```

---

## 주의 사항

> ⚠️ 본 앱은 **개인/교육 목적**으로만 사용하세요. YouTube 이용약관을 준수하세요.
