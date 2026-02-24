import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import gc

st.set_page_config(page_title="텍스트 박스 편집기", layout="wide")

# OCR 엔진 로드
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ko', 'en'], gpu=False)

reader = load_ocr()

# 세션 상태 초기화
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = None
if 'text_content' not in st.session_state:
    st.session_state.text_content = ""

def main():
    st.title("✂️ 텍스트 편집기: 글자를 지우면 이미지에서도 삭제")

    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # 이미지 로드 및 리사이징
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        h, w = img_bgr.shape[:2]
        if w > 1000:
            rate = 1000 / w
            img_bgr = cv2.resize(img_bgr, (int(w * rate), int(h * rate)))
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. 텍스트 추출 (최초 1회)
        if st.session_state.ocr_results is None:
            with st.spinner('이미지 분석 중...'):
                st.session_state.ocr_results = reader.readtext(img_bgr)
                # 초기 텍스트 박스 내용 생성 (번호: 내용)
                lines = [f"{i}: {res[1]}" for i, res in enumerate(st.session_state.ocr_results)]
                st.session_state.text_content = "\n".join(lines)
                gc.collect()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. 텍스트 편집창")
            st.info("지우고 싶은 글자의 줄(Line)을 통째로 삭제하세요.")
            
            # 전체 삭제/복구 버튼
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("모두 지우기 (전체 삭제)"):
                st.session_state.text_content = ""
            if btn_col2.button("원본 텍스트 복구"):
                lines = [f"{i}: {res[1]}" for i, res in enumerate(st.session_state.ocr_results)]
                st.session_state.text_content = "\n".join(lines)

            # 텍스트 에어리어 (편집용)
            edited_text = st.text_area(
                "추출된 텍스트 목록 (번호를 유지하며 줄을 삭제하세요)",
                value=st.session_state.text_content,
                height=400,
                key="text_editor"
            )
            # 수동으로 상태 업데이트
            st.session_state.text_content = edited_text

        with col2:
            st.subheader("2. 미리보기 및 저장")
            
            if st.button("이미지에 반영하기 (삭제 실행)", type="primary"):
                with st.spinner('처리 중...'):
                    mask = np.zeros(img_bgr.shape[:2], dtype="uint8")
                    
                    # 현재 텍스트 박스에 남아있는 번호들 파악
                    remaining_indices = []
                    for line in edited_text.split('\n'):
                        if ':' in line:
                            try:
                                idx = int(line.split(':')[0])
                                remaining_indices.append(idx)
                            except:
                                pass
                    
                    # 원본 인덱스 중 '남아있지 않은' 번호들만 마스킹(지우기)
                    for i in range(len(st.session_state.ocr_results)):
                        if i not in remaining_indices:
                            points = np.array(st.session_state.ocr_results[i][0]).astype(np.int32)
                            cv2.fillPoly(mask, [points], 255)
                    
                    # 배경 복원
                    res_cv = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
                    res_rgb = cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB)
                    st.image(res_rgb, use_container_width=True)
                    
                    # 저장 버튼
                    res_pil = Image.fromarray(res_rgb)
                    buf = io.BytesIO()
                    res_pil.save(buf, format="PNG")
                    st.download_button("이미지 다운로드", buf.getvalue(), "result.png", "image/png")
            else:
                # 결과 확인 전에는 원본 표시
                st.image(img_rgb, use_container_width=True)

    else:
        st.session_state.ocr_results = None
        st.session_state.text_content = ""
        st.info("이미지를 업로드하면 글자를 추출합니다.")

if __name__ == "__main__":
    main()
