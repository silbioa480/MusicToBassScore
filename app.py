"""MusicToBassScore — Streamlit Web Application."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="MusicToBassScore",
    page_icon="🎸",
    layout="centered",
)

# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ 설정")
    include_tab = st.toggle("TAB 악보 포함", value=True)
    pdf_method = st.selectbox(
        "PDF 생성 방법",
        options=["lilypond", "musicxml"],
        index=0,
        help="LilyPond: 고품질 PDF (시스템 설치 필요) / musicxml: 폴백 옵션",
    )
    st.divider()
    st.caption("⚠️ 본 앱은 개인/교육 목적으로만 사용하세요. YouTube 이용약관을 준수하세요.")

# ─── Header ─────────────────────────────────────────────────────────────────

st.title("🎸 MusicToBassScore")
st.subheader("베이스 기타 악보 자동 생성")
st.markdown(
    "AI가 베이스 파트를 분리하고 **오선보 + TAB** 형식의 PDF 악보를 자동으로 생성합니다.\n\n"
    "> ⏱️ 처리 시간: 4분 곡 기준 약 **5~15분** (CPU 환경)"
)

st.divider()

# ─── Input tabs ─────────────────────────────────────────────────────────────

tab_youtube, tab_file = st.tabs(["🔗 YouTube URL", "📁 음원 파일 업로드"])

with tab_youtube:
    url_input = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="youtube.com/watch?v=... 또는 youtu.be/... 형식을 지원합니다",
        key="url_input",
    )
    youtube_btn = st.button(
        "🎵 악보 생성", type="primary", use_container_width=True, key="youtube_btn"
    )

with tab_file:
    st.info(
        "💡 YouTube 다운로드 없이 로컬 음원 파일로 직접 악보를 생성합니다.\n\n"
        "지원 형식: **WAV, MP3, FLAC, OGG, M4A, AAC**"
    )
    uploaded_file = st.file_uploader(
        "음원 파일 선택",
        type=["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"],
        key="uploaded_file",
    )
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        file_title = st.text_input(
            "곡 제목 (선택)",
            placeholder="예: Lemon",
            key="file_title",
        )
    with col_meta2:
        file_artist = st.text_input(
            "아티스트 (선택)",
            placeholder="예: 米津玄師",
            key="file_artist",
        )
    file_btn = st.button(
        "🎵 악보 생성", type="primary", use_container_width=True, key="file_btn",
        disabled=(uploaded_file is None),
    )

col_clear, _ = st.columns([1, 3])
with col_clear:
    if st.button("초기화", key="clear_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ─── Progress helpers ────────────────────────────────────────────────────────

def _make_progress_cb():
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def cb(msg: str, frac: float) -> None:
        progress_bar.progress(min(frac, 1.0))
        status_text.text(f"⏳ {msg}")

    return cb, progress_bar, status_text


def _handle_error(exc: Exception, progress_bar, status_text) -> None:
    progress_bar.empty()
    status_text.empty()
    label = type(exc).__name__
    st.error(f"❌ {label}: {exc}")
    from music_to_bass_score.config import PROJECT_ROOT
    log_path = PROJECT_ROOT / "logs" / "app.log"
    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        last_lines = "".join(lines[-40:])
        with st.expander("🪵 오류 로그 (최근 40줄)"):
            st.code(last_lines, language="text")
    st.stop()


# ─── YouTube pipeline ────────────────────────────────────────────────────────

if youtube_btn and url_input.strip():
    from music_to_bass_score.downloader import validate_youtube_url

    if not validate_youtube_url(url_input.strip()):
        st.error("❌ 유효하지 않은 YouTube URL입니다.")
        st.stop()

    cb, progress_bar, status_text = _make_progress_cb()

    try:
        from music_to_bass_score.pipeline import run_pipeline
        result = run_pipeline(
            youtube_url=url_input.strip(),
            include_tab=include_tab,
            pdf_method=pdf_method,
            progress_cb=cb,
        )
        st.session_state["result"] = result
    except Exception as exc:
        _handle_error(exc, progress_bar, status_text)

# ─── File upload pipeline ────────────────────────────────────────────────────

if file_btn and uploaded_file is not None:
    from music_to_bass_score.config import AUDIO_DIR

    # Strip directory components from the browser-supplied filename to prevent
    # path traversal (e.g. "../../../etc/passwd" → "passwd").
    safe_name = Path(uploaded_file.name).name
    save_path = (AUDIO_DIR / safe_name).resolve()
    if not save_path.is_relative_to(AUDIO_DIR.resolve()):
        st.error("❌ 유효하지 않은 파일명입니다.")
        st.stop()
    save_path.write_bytes(uploaded_file.read())

    cb, progress_bar, status_text = _make_progress_cb()

    try:
        from music_to_bass_score.pipeline import run_pipeline_from_file
        result = run_pipeline_from_file(
            audio_path=save_path,
            title=file_title.strip() or Path(uploaded_file.name).stem,
            artist=file_artist.strip(),
            include_tab=include_tab,
            pdf_method=pdf_method,
            progress_cb=cb,
        )
        st.session_state["result"] = result
    except Exception as exc:
        _handle_error(exc, progress_bar, status_text)

# ─── Result display ──────────────────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]

    st.divider()
    st.success("✅ 악보 생성 완료!")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("### 🎵 곡 정보")
        st.markdown(f"**제목**: {result.metadata.title}")
        st.markdown(f"**아티스트**: {result.metadata.artist}")
        dur = result.metadata.duration_sec
        st.markdown(f"**길이**: {int(dur // 60)}분 {int(dur % 60)}초")

    with col_info2:
        st.markdown("### 📊 분석 결과")
        st.markdown(f"**조성**: {result.analysis.key}")
        st.markdown(f"**BPM**: {result.analysis.bpm_rounded}")
        st.markdown(
            f"**박자**: "
            f"{result.analysis.time_signature_num}/{result.analysis.time_signature_den}"
        )
        st.markdown(f"**감지된 마디 수**: {len(result.chord_labels)}")

    st.divider()
    st.markdown("### 🎼 코드 진행")
    if result.chord_labels:
        chords_per_row = 8
        rows = [
            result.chord_labels[i : i + chords_per_row]
            for i in range(0, len(result.chord_labels), chords_per_row)
        ]
        roman = result.roman_labels or []

        def _join(measure):
            # measure is list[(offset, label)] | list[str] | str
            if isinstance(measure, str):
                return measure
            out = []
            for item in measure:
                out.append(item[1] if isinstance(item, (list, tuple)) else str(item))
            return " ".join(out)

        for row_idx, row in enumerate(rows):
            cols = st.columns(len(row))
            for col_idx, (col, chord) in enumerate(zip(cols, row)):
                measure_num = row_idx * chords_per_row + col_idx + 1
                m_i = row_idx * chords_per_row + col_idx
                deg = roman[m_i] if m_i < len(roman) else ""
                col.metric(label=f"마디 {measure_num}", value=_join(chord),
                           delta=_join(deg), delta_color="off")

    st.divider()
    st.markdown("### 📄 악보 다운로드")

    export_path = result.export.pdf_path
    if export_path.exists():
        with open(export_path, "rb") as f:
            file_bytes = f.read()

        suffix = export_path.suffix
        mime = "application/pdf" if suffix == ".pdf" else "application/xml"
        label = "⬇️ PDF 다운로드" if suffix == ".pdf" else "⬇️ MusicXML 다운로드"
        filename = f"{result.metadata.title}_bass{suffix}"

        st.download_button(
            label=label,
            data=file_bytes,
            file_name=filename,
            mime=mime,
            use_container_width=True,
        )
        if suffix != ".pdf":
            st.info(
                "ℹ️ LilyPond이 설치되지 않아 MusicXML 형식으로 출력되었습니다. "
                "`sudo apt-get install lilypond` 후 재실행하면 PDF를 받을 수 있습니다."
            )
    else:
        st.warning("⚠️ 출력 파일을 찾을 수 없습니다.")

elif not youtube_btn and not file_btn:
    st.info("👆 YouTube URL을 입력하거나 음원 파일을 업로드하고 '악보 생성' 버튼을 눌러주세요.")
