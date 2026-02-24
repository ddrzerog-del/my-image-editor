import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import io

# 페이지 설정
st.set_page_config(page_title="고급 이미지 텍스트 편집기", layout="wide")

st.title("🚀 고급 이미지 텍스트 편집기")
st.markdown("글자를 지우거나, 선택해서 삭제하고, 텍스트만 따로 복사할 수 있습니다.")

# OCR 엔진 초기화 (캐싱 처리하여 속도 향상)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ko', 'en'])

reader = load_ocr()

# 세션 상태 초기화 (이미지 재처리 방지)
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = None
if 'image_brg' not in st.session_state:
    st.session_state.image_bgr = None

# 파일 업로드
uploaded_file = st.file_uploader("편집할 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 이미지 로드 및 세션 저장
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.image_bgr = img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. 텍스트 추출 (처음 한 번만 실행)
    with st.spinner('이미지 분석 중...'):
        if st.session_state.ocr_results is None:
            st.session_state.ocr_results = reader.readtext(img_bgr)

    results = st.session_state.ocr_results

    # 화면 레이아웃 분할
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("원본 및 텍스트 선택")
        st.image(img_rgb, use_container_width=True)
        
        # 전체 선택 기능
        select_all = st.checkbox("전체 선택")
        
        # 개별 선택 리스트
        selected_indices = []
        st.write("삭제할 항목을 선택하세요:")
        for i, (bbox, text, prob) in enumerate(results):
            is_checked = st.checkbox(f"[{i+1}] {text}", value=select_all, key=f"check_{i}")
            if is_checked:
                selected_indices.append(i)

    with col2:
        st.subheader("결과 및 추출된 텍스트")
        
        # 기능 1 & 2: 삭제 처리 버튼
        if st.button("선택한 텍스트 지우기 실행", type="primary"):
            if not selected_indices:
                st.warning("지울 항목을 선택해주세요.")
            else:
                mask = np.zeros(img_bgr.shape[:2], dtype="uint8")
                for idx in selected_indices:
                    bbox = results[idx][0]
                    points = np.array(bbox).astype(np.int32)
                    cv2.fillPoly(mask, [points], 255)
                
                # 배경 복원 (Inpainting)
                res_cv = cv2.inpaint(img_bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
                res_rgb = cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB)
                res_pil = Image.fromarray(res_rgb)
                
                st.image(res_pil, caption="텍스트가 제거된 이미지", use_container_width=True)
                
                # 다운로드 버튼
                buf = io.BytesIO()
                res_pil.save(buf, format="PNG")
                st.download_button("결과 이미지 저장", buf.getvalue(), "cleaned_image.png", "image/png")
        
        st.divider()
        
        # 기능 3: 텍스트 추출 및 복사 영역
        st.subheader("📝 추출된 전체 텍스트")
        all_text = "\n".join([res[1] for res in results])
        
        if all_text:
            # 텍스트 에어리어에 넣어 사용자가 복사하기 쉽게 함
            st.text_area("아래 텍스트를 복사해서 사용하세요:", value=all_text, height=300)
            
            # 간편 복사 버튼 (텍스트만 별도 제공)
            st.download_button("텍스트 파일로 저장", all_text, "extracted_text.txt")
        else:
            st.write("추출된 텍스트가 없습니다.")

else:
    # 이미지 업로드 전 세션 초기화
    st.session_state.ocr_results = None
    st.info("이미지를 업로드하면 분석이 시작됩니다.")