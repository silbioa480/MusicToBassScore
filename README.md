# MusicToBassScore

YouTube URL 또는 음원 파일을 입력하면 AI가 자동으로 **코드 차트 PDF**(마디별 코드 + 조표 기준 로마숫자 도수)를 생성하는 웹 앱입니다.

J-POP 커버 밴드 연주자를 위해 설계되었으며, 곡의 화성 진행을 한눈에 확인할 수 있는 **단일 오선보 코드 차트**로 출력됩니다.

> **버전 안내**
> 초기 버전(v1.0)은 베이스 오선보 + TAB을 목표로 했으나, 베이스 음 전사의 한계로
> 목적을 **코드/도수 분석**으로 전환했습니다. 현재(v2.9)는 코드 인식 정확도 개선에 집중합니다.
> 자세한 버전 히스토리는 `CLAUDE.md`를 참고하세요.

---

## 처리 파이프라인

```
입력 (YouTube URL / 음원 파일)
    ↓ yt-dlp + ffmpeg
오디오 다운로드·변환 (WAV)
    ↓ librosa
BPM / 조성 / 박자 분석 + 정속 마디 그리드
    ↓ librosa (슬라이딩 윈도우)
구간별 전조(Key Modulation) 감지
    ↓ Demucs (htdemucs)
4개 스템 분리 (vocals / drums / bass / other)
    ↓ 하모닉 서브믹스 (vocals + other + 0.3·bass)
    ↓ BTC Transformer (전체음원/서브믹스)
마디별 코드 진행 감지 (적응형 화성 리듬 + 전위 슬래시 + 신뢰도)
    ↓ music21 (romanNumeralFromChord)
조표 기준 로마숫자 도수 변환 (전조 마디별 재계산)
    ↓ music21 + LilyPond
코드 차트 PDF 생성
```

---

## 사용 라이브러리

### Web UI

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `streamlit` | ≥1.35.0 | 웹 UI — URL/파일 입력, 진행 표시, 코드 진행 미리보기, PDF 다운로드 |

### YouTube 다운로드

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `yt-dlp` | ≥2025.1.1 | YouTube 오디오 추출 + 메타데이터 수집. 다중 player client 폴백으로 HTTP 403 회피 |

### 오디오 처리 및 분석

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `librosa` | ≥0.10.2 | BPM·조성·박자 분석, 마디 그리드, 크로마 추출(전조 감지·전위 베이스) |
| `soundfile` | ≥0.12.1 | WAV·스템 읽기/쓰기 + 하모닉 서브믹스 합성 (ffprobe 불필요) |
| `numpy` | ≥1.26, <2.3 | 오디오 배열 연산 핵심 수치 계산 |
| `scipy` | ≥1.13.0 | 신호 처리 보조 연산 |
| `resampy` | ≥0.2.2, <0.4.3 | 오디오 샘플레이트 변환 (librosa 내부) |
| `matplotlib` | ≥3.8.0 | librosa 의존 (시각화) |

### AI 음원 분리

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `demucs` | ≥4.0.1 | Meta AI 음원 분리. `htdemucs`로 4개 스템 분리 → 깨끗한 베이스(전위 표기) + 하모닉 서브믹스(BTC 입력) |
| `torch` | ≥2.2.0 | Demucs·BTC 구동 딥러닝 프레임워크 (CPU/GPU 자동 선택) |
| `torchaudio` | ≥2.2.0 | Demucs 내부 오디오 처리 (torch 버전과 일치 필요) |

### 코드 인식 (BTC Transformer)

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `mir_eval` | ≥0.6 | 코드 인식 정확도 평가(WCSR) + BTC 라벨 매핑 유틸 |
| `pyyaml` | ≥5.1 | BTC 모델 설정(`run_config.yaml`) 파싱 |

> BTC 모델 코드·체크포인트(~24 MB)는 `scripts/setup_btc.py`로 `third_party/`에 내려받습니다 (저장소에 포함되지 않음).

### 악보 생성 및 PDF 출력

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `music21` | ≥9.1.0 | 코드 차트 구성 + `romanNumeralFromChord` 로마숫자 분석 |
| `requests` | ≥2.31.0 | music21 내부 의존 |

> **시스템 패키지도 필요합니다**
> `LilyPond` — 코드 차트를 고품질 PDF로 렌더링 (미설치 시 MusicXML 폴백)
> `ffmpeg` — yt-dlp 오디오 변환 / 비-WAV 입력 변환

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

# 4. BTC 코드 인식 모델 준비 (필수, ~24MB)
python scripts/setup_btc.py

# 5. Demucs 모델 사전 다운로드 (선택, ~1GB)
python scripts/download_models.py

# 6. 환경 검증
python scripts/verify_setup.py
```

> BTC 모델이 없으면 librosa 크로마 템플릿 매칭으로 graceful fallback 하지만, 정확도가 크게 떨어집니다.

---

## 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속 후 YouTube URL을 입력하거나 음원 파일(WAV/MP3/FLAC/OGG/M4A/AAC/OPUS)을 업로드하세요.

---

## 출력 형식

생성되는 코드 차트 PDF 구성 요소:

- **헤더**: 곡 제목, 아티스트, 조성, 박자, BPM
- **코드 차트**: 마디별 박스 그리드 (한 줄 4마디)
  - 각 마디 위에 코드 심볼 (예: `Am`, `F`, `G/B`)
  - 각 마디 아래에 조표 기준 로마숫자 도수 (예: `vi`, `IV`, `I/III`)
  - 적응형 화성 리듬: 마디당 코드 1~2개
  - **전조 마커**: 조표가 바뀌는 마디에 `[Key: B]` 표시 + 해당 구간 도수 재계산
  - **신뢰도 마커**: BTC 확신이 낮은 코드에 작은 회색 `?` 표시 (청취 확인 권장)

---

## 처리 시간

| 환경 | 4분 곡 기준 예상 시간 |
|---|---|
| CPU (일반 노트북) | 약 8~15분 |
| GPU (CUDA) | 약 2~5분 |

Demucs 음원 분리가 전체 시간의 대부분을 차지합니다 (정확도 우선). 동일 곡 재실행 시 분리 결과는 캐시됩니다.

---

## 코드 인식 정확도 평가

참조 어노테이션(Harte `.lab`) 대비 WCSR 점수를 측정하거나, 전체음원 vs 하모닉 서브믹스 A/B 비교에 사용합니다.

```bash
# BTC 타임라인 확인 (참조 불필요)
python scripts/eval_chords.py --audio song.wav

# 참조 어노테이션 대비 WCSR (root/triads/majmin/mirex)
python scripts/eval_chords.py --audio song.wav --ref reference.lab

# 하모닉 서브믹스 A/B 비교
python scripts/eval_chords.py --audio song.wav --ref reference.lab \
    --harmonic-mix tmp/stems/htdemucs/<song_id>/
```

---

## 로그

실행 로그는 `logs/app.log`에 기록됩니다. 오류 발생 시 이 파일을 확인하세요 (UI에도 최근 40줄 표시).

```
logs/
└── app.log   # 타임스탬프·모듈명·라인번호 포함, 로테이션 적용
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

# 코드 스타일 검사
ruff check src/ tests/
```

---

## 주의 사항

> ⚠️ 본 앱은 **개인/교육 목적**으로만 사용하세요. YouTube 이용약관을 준수하세요.
