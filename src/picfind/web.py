from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from picfind.config import DEFAULT_DB_PATH, Settings
from picfind.image_io import register_heif_opener  # noqa: F401
from picfind.search import search_images


st.set_page_config(page_title="PicFind", layout="wide")


def render_result_card(index: int, path: Path, caption: str, score: float) -> None:
    left, right = st.columns([1, 1.4])
    with left:
        try:
            with Image.open(path) as image:
                st.image(image.copy(), use_container_width=True)
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            st.warning("이미지를 열 수 없습니다.")
    with right:
        st.markdown(f"### 결과 {index}")
        st.write(f"유사도 점수: `{score:.4f}`")
        st.write(f"경로: `{path}`")
        if caption:
            st.write(f"설명: {caption}")
        else:
            st.write("설명: 저장된 캡션 없음")


def main() -> None:
    st.title("PicFind")
    st.caption("로컬 사진을 문장으로 검색하는 CLIP 기반 사진 검색기")

    with st.sidebar:
        st.header("설정")
        db_path_input = st.text_input("DB 경로", str(DEFAULT_DB_PATH))
        limit = st.slider("결과 개수", min_value=1, max_value=20, value=8)
        st.markdown("인덱싱은 CLI에서 수행합니다.")
        st.code('picfind index --source "D:\\Photos"')

    query = st.text_input("검색 문장", placeholder="예: 노란색이 많이 들어간 풍경")
    search_clicked = st.button("검색", type="primary")

    if not search_clicked:
        st.info("문장을 입력하고 검색 버튼을 누르세요.")
        return

    if not query.strip():
        st.warning("검색 문장을 입력하세요.")
        return

    settings = Settings(db_path=Path(db_path_input))
    if not settings.db_path.exists():
        st.error(f"DB 파일을 찾을 수 없습니다: {settings.db_path}")
        return

    with st.spinner("검색 중..."):
        results = search_images(query.strip(), settings, limit=limit)

    if not results:
        st.warning("검색 결과가 없습니다. 먼저 인덱싱을 실행했는지 확인하세요.")
        return

    st.success(f"{len(results)}개의 결과를 찾았습니다.")
    for index, result in enumerate(results, start=1):
        render_result_card(index, result.path, result.caption, result.score)
        st.divider()
