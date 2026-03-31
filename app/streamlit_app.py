from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_runtime import SpeakerDemoRuntime
from app.registry import SpeakerRegistry


DEFAULT_CHECKPOINT = (
    "train/outputs/experiments/Mode3_concat_train_raw_wavlm_mfbe_pitch/best_model.pth"
)
DEFAULT_REGISTRY = "app/data/speaker_registry.pt"


def _save_upload_to_temp(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def _audio_input_block(label: str, key_prefix: str):
    source = st.radio(
        f"Nguồn audio - {label}",
        options=["Tải file", "Ghi âm trực tiếp"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    if source == "Tải file":
        return st.file_uploader(
            f"{label}",
            type=["wav", "mp3", "flac"],
            key=f"{key_prefix}_upload",
        )

    return st.audio_input(f"{label}", key=f"{key_prefix}_record")


def _cleanup_temp(paths: Iterable[str]) -> None:
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


@st.cache_resource
def get_runtime(checkpoint_path: str, device: str) -> SpeakerDemoRuntime:
    return SpeakerDemoRuntime(checkpoint_path=checkpoint_path, device=device)


@st.cache_resource
def get_registry(registry_path: str) -> SpeakerRegistry:
    return SpeakerRegistry(registry_path=registry_path)


def _render_registry_state(registry: SpeakerRegistry) -> None:
    speakers = registry.list_speakers()
    if not speakers:
        st.info("Chưa có speaker nào được đăng ký.")
        return

    rows = [{"speaker_id": spk, "num_samples": registry.num_samples(spk)} for spk in speakers]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def register_tab(runtime: SpeakerDemoRuntime, registry: SpeakerRegistry) -> None:
    st.subheader("Đăng ký người nói mới")
    speaker_id = st.text_input("Speaker ID", placeholder="vd: speaker_nguyen_a")
    input_mode = st.radio(
        "Chế độ đăng ký",
        options=["Nhiều file upload", "Ghi âm trực tiếp 1 mẫu"],
        horizontal=True,
        key="register_mode",
    )

    uploads = None
    recorded = None
    if input_mode == "Nhiều file upload":
        uploads = st.file_uploader(
            "Tải lên 1 hoặc nhiều file audio cho speaker này",
            type=["wav", "mp3", "flac"],
            accept_multiple_files=True,
            key="register_uploads",
        )
    else:
        recorded = st.audio_input("Ghi âm mẫu cho speaker", key="register_record")

    if st.button("Đăng ký speaker", use_container_width=True):
        if not speaker_id.strip():
            st.error("Bạn cần nhập Speaker ID.")
            return

        temp_paths: List[str] = []
        try:
            added = 0
            if input_mode == "Nhiều file upload":
                if not uploads:
                    st.error("Bạn cần tải lên ít nhất 1 audio.")
                    return
                for up in uploads:
                    path = _save_upload_to_temp(up)
                    temp_paths.append(path)
                    emb = runtime.embedding_from_audio(path)
                    registry.add_embedding(speaker_id.strip(), emb)
                    added += 1
            else:
                if recorded is None:
                    st.error("Bạn cần ghi âm trước khi đăng ký.")
                    return
                path = _save_upload_to_temp(recorded)
                temp_paths.append(path)
                emb = runtime.embedding_from_audio(path)
                registry.add_embedding(speaker_id.strip(), emb)
                added = 1

            registry.save()
            st.success(f"Đã đăng ký speaker '{speaker_id}' với {added} audio mới.")
        except Exception as exc:
            st.exception(exc)
        finally:
            _cleanup_temp(temp_paths)

    st.markdown("### Danh sách speaker đã đăng ký")
    _render_registry_state(registry)


def compare_tab(runtime: SpeakerDemoRuntime) -> None:
    st.subheader("So sánh 2 audio")
    threshold = st.slider("Threshold accept/reject", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

    col1, col2 = st.columns(2)
    with col1:
        audio_a = _audio_input_block("Audio A", "cmp_a")
    with col2:
        audio_b = _audio_input_block("Audio B", "cmp_b")

    if st.button("So sánh", use_container_width=True):
        if audio_a is None or audio_b is None:
            st.error("Bạn cần tải đủ 2 audio.")
            return

        temp_paths: List[str] = []
        try:
            path_a = _save_upload_to_temp(audio_a)
            path_b = _save_upload_to_temp(audio_b)
            temp_paths.extend([path_a, path_b])

            score = runtime.compare_two_audio(path_a, path_b)
            decision = "Cùng người" if score >= threshold else "Khác người"

            st.metric("Cosine similarity", f"{score:.4f}")
            if score >= threshold:
                st.success(f"Kết luận: {decision} (>= {threshold:.2f})")
            else:
                st.warning(f"Kết luận: {decision} (< {threshold:.2f})")
        except Exception as exc:
            st.exception(exc)
        finally:
            _cleanup_temp(temp_paths)


def identify_tab(runtime: SpeakerDemoRuntime, registry: SpeakerRegistry) -> None:
    st.subheader("Nhận diện 1 audio trong danh sách đã đăng ký")
    top_k = st.slider("Hiển thị Top-K", min_value=1, max_value=10, value=5)
    threshold = st.slider("Threshold xác nhận Top-1", min_value=0.0, max_value=1.0, value=0.6, step=0.01, key="id_th")

    query_audio = _audio_input_block("Audio cần nhận diện", "id_query")

    if st.button("Nhận diện", use_container_width=True):
        if query_audio is None:
            st.error("Bạn cần tải lên 1 file audio.")
            return

        centers = registry.centroids()
        if not centers:
            st.error("Chưa có speaker nào trong registry. Hãy đăng ký trước.")
            return

        temp_path = None
        try:
            temp_path = _save_upload_to_temp(query_audio)
            emb = runtime.embedding_from_audio(temp_path)
            best_speaker, scores = runtime.identify(emb, centers)

            if best_speaker is None:
                st.error("Không thể nhận diện vì registry rỗng.")
                return

            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rows: List[Dict[str, object]] = [
                {"rank": idx + 1, "speaker_id": sid, "similarity": float(sc)}
                for idx, (sid, sc) in enumerate(sorted_scores[:top_k])
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            top_score = sorted_scores[0][1]
            if top_score >= threshold:
                st.success(f"Kết quả Top-1: {best_speaker} (score={top_score:.4f})")
            else:
                st.warning(f"Top-1: {best_speaker} nhưng score={top_score:.4f} < threshold={threshold:.2f}")
        except Exception as exc:
            st.exception(exc)
        finally:
            if temp_path:
                _cleanup_temp([temp_path])


def main() -> None:
    st.set_page_config(page_title="Vietnamese Speaker Demo", layout="wide")
    st.title("Demo nhận diện người nói tiếng Việt")

    with st.sidebar:
        st.markdown("### Cấu hình")
        checkpoint_path = st.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)
        registry_path = st.text_input("Registry path", value=DEFAULT_REGISTRY)
        device_options = ["cpu"]
        if __import__("torch").cuda.is_available():
            device_options.append("cuda")
        device = st.selectbox("Device", device_options, index=0)

    runtime = get_runtime(checkpoint_path=checkpoint_path, device=device)
    registry = get_registry(registry_path=registry_path)

    tab_register, tab_compare, tab_identify = st.tabs(
        ["Đăng ký speaker", "So sánh 2 audio", "Nhận diện trong registry"]
    )

    with tab_register:
        register_tab(runtime, registry)
    with tab_compare:
        compare_tab(runtime)
    with tab_identify:
        identify_tab(runtime, registry)


if __name__ == "__main__":
    main()
