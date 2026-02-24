import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import gc

st.set_page_config(page_title="고급 텍스트 제거기", layout="wide")

# 1. OCR 엔진 로드
@st.cache_resource
def load_ocr():
    with st.spinner('인공지능 모델을 준비 중입니다...'):
        return easyocr.Reader(['ko', 'en'], gpu=False)

reader = load_ocr()

# 2. 세션 상태 초기화 (데이터 유지)
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = None
if 'selected_list' not in st.session_state:
    st.session_state.selected_list = []

def main():
    st.title("🖼️ 스마트 이미지 텍스트 편집기")

    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # 이미지 처리 (메모리 절약을 위해 리사이징)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        h, w = img_bgr.shape[:2]
        if w > 1000:
            rate = 1000 / w
            img_bgr = cv2.resize(img_bgr, (int(w * rate), int(h * rate)))
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 텍스트 추출 (세션에 저장)
        if st.session_state.ocr_results is None:
            with st.spinner('이미지 분석 중...'):
                st.session_state.ocr_results = reader.readtext(img_bgr)
                gc.collect()

        results = st.session_state.ocr_results
        
        # 선택 도구용 리스트 생성 (인덱스 + 텍스트 내용)
        options = [f"[{i}] {res[1]}" for i, res in enumerate(results)]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. 원본 이미지")
            st.image(img_rgb, use_container_width=True)
            
            # 전체 선택/해제 버튼
            st.write("---")
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("전체 선택"):
                st.session_state.selected_list = options
            if btn_col2.button("선택 해제"):
                st.session_state.selected_list = []

            # 텍스트 선택 (멀티셀렉트 도구)
            selected_items = st.multiselect(
                "지우고 싶은 텍스트를 선택하세요 (직접 클릭):",
                options,
                default=st.session_state.selected_list,
                key="multi_select"
            )
            # 선택 상태 업데이트
            st.session_state.selected_list = selected_items

        with col2:
            st.subheader("2. 작업 실행 및 결과")
            
            # 지우기 버튼
            if st.button("선택한 텍스트 삭제 실행", type="primary"):
                if selected_items:
                    with st.spinner('이미지 복원 중...'):
                        mask = np.zeros(img_bgr.shape[:2], dtype="uint8")
                        for item in selected_items:
                            # 선택된 항목의 인덱스 추출
                            idx = int(item.split(']')[0].replace('[', ''))
                            points = np.array(results[idx][0]).astype(np.int32)
                            cv2.fillPoly(mask, [points], 255)
                        
                        # 배경 복원
                        res_cv = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
                        res_rgb = cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, caption="삭제 완료", use_container_width=True)
                        
                        # 다운로드
                        res_pil = Image.fromarray(res_rgb)
                        buf = io.BytesIO()
                        res_pil.save(buf, format="PNG")
                        st.download_button("이미지 저장하기", buf.getvalue(), "cleaned.png", "image/png")
                        gc.collect()
                else:
                    st.warning("선택된 텍스트가 없습니다.")

            st.divider()
            
            # 텍스트 추출 영역
            st.subheader("📝 추출된 전체 텍스트")
            full_text = "\n".join([res[1] for res in results])
            st.text_area("텍스트를 복사하려면 아래 창을 이용하세요:", value=full_text, height=250)
            
            if selected_items:
                st.write("📍 **현재 선택된 텍스트만 보기:**")
                selected_text_only = "\n".join([item.split(" ", 1)[1] for item in selected_items])
                st.code(selected_text_only)

    else:
        st.session_state.ocr_results = None
        st.session_state.selected_list = []
        st.info("이미지를 업로드해 주세요.")
        gc.collect()

if __name__ == "__main__":
    main()
