import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import gc
import os

st.set_page_config(page_title="고급 텍스트 제거기", layout="wide")

# 모델 저장 경로 설정 (서버 용량 문제 방지)
@st.cache_resource
def load_ocr():
    # 모델을 다운로드할 때 상태를 표시함
    with st.spinner('인공지능 모델을 준비 중입니다... (최초 실행 시 1~3분 소요)'):
        # gpu=False를 명시하여 CUDA 에러 방지
        return easyocr.Reader(['ko', 'en'], gpu=False)

# 메인 실행부
def main():
    st.title("🚀 고효율 이미지 텍스트 편집기")
    
    # OCR 로드
    try:
        reader = load_ocr()
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return

    if 'ocr_results' not in st.session_state:
        st.session_state.ocr_results = None

    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 이미지 크기 최적화 (메모리 부족 방지)
        h, w = img_bgr.shape[:2]
        if w > 1000:
            rate = 1000 / w
            img_bgr = cv2.resize(img_bgr, (int(w * rate), int(h * rate)))
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if st.session_state.ocr_results is None:
            with st.spinner('이미지 분석 중...'):
                st.session_state.ocr_results = reader.readtext(img_bgr)
                gc.collect()

        results = st.session_state.ocr_results
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("원본 및 선택")
            st.image(img_rgb, use_container_width=True)
            select_all = st.checkbox("전체 선택")
            selected_indices = []
            for i, (bbox, text, prob) in enumerate(results):
                if st.checkbox(f"[{i+1}] {text}", value=select_all, key=f"check_{i}"):
                    selected_indices.append(i)

        with col2:
            st.subheader("결과 및 추출")
            if st.button("텍스트 지우기 실행", type="primary"):
                if selected_indices:
                    with st.spinner('배경 복원 중...'):
                        mask = np.zeros(img_bgr.shape[:2], dtype="uint8")
                        for idx in selected_indices:
                            points = np.array(results[idx][0]).astype(np.int32)
                            cv2.fillPoly(mask, [points], 255)
                        
                        res_cv = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
                        res_rgb = cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, use_container_width=True)
                        
                        res_pil = Image.fromarray(res_rgb)
                        buf = io.BytesIO()
                        res_pil.save(buf, format="PNG")
                        st.download_button("이미지 저장", buf.getvalue(), "cleaned.png", "image/png")
                        gc.collect()
                else:
                    st.warning("지울 항목을 선택하세요.")

            st.divider()
            all_text = "\n".join([res[1] for res in results])
            st.text_area("추출된 텍스트", value=all_text, height=200)
    else:
        st.session_state.ocr_results = None
        gc.collect()

if __name__ == "__main__":
    main()
