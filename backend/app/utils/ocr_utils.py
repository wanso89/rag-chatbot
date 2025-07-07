"""
OCR 유틸리티 모듈

이 모듈은 PaddleOCR을 사용하여 이미지, PDF 파일에서 텍스트를 추출하는 기능을 제공합니다.
표, 도표, 이미지 등 다양한 형식의 콘텐츠에서 텍스트를 추출하는 기능이
포함되어 있습니다.
"""

import os
import re
import traceback
import numpy as np
import logging
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import asyncio
import tempfile

# 이미지 처리 관련 라이브러리
from PIL import Image
import cv2


# PDF 처리 라이브러리
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdf2image import convert_from_path
import paddleocr
from paddleocr import PPStructure

# 로깅 설정
logger = logging.getLogger(__name__)
ocr_processing_logger = logging.getLogger("ocr_processing") # 새로운 로거
ocr_processing_logger.setLevel(logging.INFO)
# 파일 핸들러 설정 (필요시)
# import logging.handlers
# file_handler = logging.handlers.RotatingFileHandler('ocr_process.log', maxBytes=1024*1024, backupCount=5)
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# ocr_processing_logger.addHandler(file_handler)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 사용
# PaddleOCR 로케일 설정 (한국어 + 영어)

_paddle_ocr_instance = None
_paddle_structure_instance = None 
PKG_DIR = Path(os.path.dirname(paddleocr.__file__)) 
str(Path(os.path.dirname(paddleocr.__file__)) / 'ppocr/utils/dict/korean_dict.txt')

def get_paddle_ocr():
    """기본 OCR 인스턴스"""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            logger.info("PaddleOCR 인스턴스 초기화 중...")
            _paddle_ocr_instance = paddleocr.PaddleOCR(
                det_model_dir='/root/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer',
                rec_model_dir='/root/.paddleocr/whl/rec/korean/korean_PP-OCRv3_rec_infer',    
                cls_model_dir='/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer',
                use_angle_cls=True,          # ✅ 회전 텍스트 처리
                lang='korean',               # ✅ 한국어 모델
                use_gpu=False,               # ✅ GPU 사용
                show_log=False,             # 🔧 로그 숨김
                use_space_char=True,

                # 🚀 높인 임계값 (기존 0.3 → 0.5)
                det_db_thresh=0.6,          # 🔧 검출 임계값 (0.3→0.5)
                det_db_box_thresh=0.6,      # 🔧 박스 임계값 (0.6→0.7)
                rec_image_shape="3, 32, 640",  # ✅ 높이32, 너비32
                rec_batch_num      = 16,   
            )
            logger.info("PaddleOCR 인스턴스 초기화 완료")
        except Exception as e:
            logger.error(f"PaddleOCR 초기화 오류: {e}")
            raise
    return _paddle_ocr_instance

def get_paddle_structure():
    global _paddle_structure_instance
    if _paddle_structure_instance is None:
        _paddle_structure_instance = PPStructure(
            # ──────────────────────────────────────────────
            # 1)  layout / table → 지원되는 'en' 로 고정
            # ──────────────────────────────────────────────
            lang='en',
            layout=True, table=True, ocr=True,
            use_gpu=False, show_log=False,

            # ──────────────────────────────────────────────
            # 2)  직접 받은 모델 경로들
            # ──────────────────────────────────────────────
            det_model_dir = '/root/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer',
            rec_model_dir = '/root/.paddleocr/whl/rec/korean/korean_PP-OCRv3_rec_infer',
            cls_model_dir = '/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer',

            layout_model_dir = '/root/.paddleocr/whl/layout/picodet_lcnet_x1_0_fgd_layout_infer',
            table_model_dir  = '/root/.paddleocr/whl/table/en_ppstructure_mobile_v2.0_SLANet_infer',

            # ──────────────────────────────────────────────
            # 3)  각 dict 경로를 명시적으로 맞춰 줌
            #     - 레이아웃/테이블용
            # ──────────────────────────────────────────────
            table_char_dict_path = str(PKG_DIR / 'ppocr/utils/dict/table_structure_dict.txt'),
            #     - 한글 OCR용
            rec_char_dict_path   = str(PKG_DIR / 'ppocr/utils/dict/korean_dict.txt'),
        )
    return _paddle_structure_instance

def get_paddle_ocr_version():
    """
    설치된 PaddleOCR 버전을 확인합니다.
    """
    try:
        import paddleocr
        return paddleocr.__version__
    except Exception as e:
        logger.error(f"PaddleOCR 버전 확인 중 오류: {e}")
        return None

def extract_text_from_image_sync(image_path: str, min_confidence: float = 0.5) -> str:
    """
    이미지 파일에서 텍스트를 추출합니다.
    
    Args:
        image_path: 이미지 파일 경로
        min_confidence: 최소 신뢰도 (0.0 ~ 1.0)
        
    Returns:
        추출된 텍스트
    """
    try:
        start_time = time.time()
        
        # PaddleOCR 인스턴스 가져오기
        ocr = get_paddle_ocr()
        
        # 이미지 로드 및 전처리
        image = preprocess_image_for_ocr(image_path)
        
        # OCR 처리
        result = ocr.ocr(image, cls=True)
        
        # 결과 텍스트 추출 및 정렬
        extracted_texts = []
        if result:
            for idx, line_result in enumerate(result):
                if not line_result:
                    continue
                    
                # 신뢰도 기준으로 필터링
                line_texts = []
                for box, (text, confidence) in line_result:
                    if confidence >= min_confidence:
                        line_texts.append(text)
                
                if line_texts:
                    extracted_texts.append(" ".join(line_texts))
        
        # 결과 텍스트 구성
        text = "\n".join(extracted_texts)
        
        # 후처리: 불필요한 줄바꿈, 공백 정리
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        elapsed = time.time() - start_time
        logger.info(f"이미지 OCR 처리 완료: {elapsed:.2f}초")
        
        return text
    except Exception as e:
        logger.error(f"이미지에서 텍스트 추출 중 오류 발생: {e}")
        traceback.print_exc()
        return ""

def extract_text_from_pdf_with_ocr_sync(pdf_path: str, min_confidence: float = 0.5) -> str:
    """
    PDF 파일에서 텍스트를 추출합니다. 
    먼저 PyPDF를 통한 직접 추출을 시도하고, 충분한 텍스트가 없는 경우 OCR을 적용합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        min_confidence: 추출된 텍스트의 최소 신뢰도 (0.0 ~ 1.0)
        
    Returns:
        추출된 텍스트
    """
    try:
        # 1. 먼저 일반적인 텍스트 추출 시도 (PDFMiner)
        ocr_processing_logger.info(f"[PDF OCR START] 파일 처리 시작: {pdf_path}")
        logger.info(f"PDF에서 텍스트 직접 추출 시도: {pdf_path}")
        extracted_text_raw = pdfminer_extract_text(pdf_path)
        
        # 추출된 텍스트에서 실제 유효 문자 수 확인
        meaningful_text_threshold = 50  # 실제 의미있는 문자의 최소 개수
        valid_text_for_skip_ocr = False
        if extracted_text_raw:
            # 공백, 줄바꿈, form feed 등 제외하고 실제 문자만 카운트
            # 정규표현식을 사용하여 한글, 영어 알파벳, 숫자만 카운트
            meaningful_chars = re.sub(r'[^a-zA-Z0-9가-힣]', '', extracted_text_raw)
            num_meaningful_chars = len(meaningful_chars)
            
            total_chars_no_whitespace = len(re.sub(r'\\s+', '', extracted_text_raw))

            logger.info(f"PDFMiner 추출: 총 문자(공백제거): {total_chars_no_whitespace}, 유효 문자(a-zA-Z0-9가-힣): {num_meaningful_chars}")
            ocr_processing_logger.info(f"[PDF OCR INFO] PDFMiner 추출 결과 - 총 문자(공백제거): {total_chars_no_whitespace}, 유효 문자: {num_meaningful_chars} (임계값: {meaningful_text_threshold})")

            if total_chars_no_whitespace >= 100 and num_meaningful_chars >= meaningful_text_threshold:
                valid_text_for_skip_ocr = True
        
        if valid_text_for_skip_ocr:
            logger.info(f"PDF에서 텍스트 직접 추출 성공 (유효 문자 충분): {len(extracted_text_raw)} 글자")
            ocr_processing_logger.info(f"[PDF OCR SUCCESS] 직접 텍스트 추출 성공 (PDFMiner): {pdf_path}, 글자수: {len(extracted_text_raw)}")
            return extracted_text_raw
            
        # 2. 직접 추출이 불충분한 경우, OCR 적용
        logger.info(f"직접 추출 불충분 (또는 유효 문자 부족). PaddleOCR 적용 중: {pdf_path}")
        ocr_processing_logger.info(f"[PDF OCR INFO] 직접 추출 불충분/유효문자 부족, 이미지 변환 및 PaddleOCR 진행: {pdf_path}")
        
        # PDF를 이미지로 변환
        ocr_processing_logger.info(f"[PDF OCR INFO] PDF -> 이미지 변환 시작: {pdf_path}")
        images = convert_from_path(
            pdf_path,
            dpi=500,  # 해상도 (높을수록 더 정확하지만 처리 시간 증가)
            thread_count=4,  # 멀티스레딩
            use_pdftocairo=True,  # pdftocairo 사용 (더 빠르고 정확함)
            grayscale=False,  # 컬러 유지 (표와 도표 인식 향상)
            transparent=False  # 투명도 제거
        )
        ocr_processing_logger.info(f"[PDF OCR INFO] PDF -> 이미지 변환 완료: {pdf_path}, 페이지 수: {len(images)}")
        logger.info(f"PDF 이미지 변환 완료: {len(images)} 페이지")
        
        # 각 이미지에 OCR 적용 (순차 처리)
        page_texts = []
        for i, image in enumerate(images):
            ocr_processing_logger.info(f"[PDF OCR INFO] 페이지 {i+1}/{len(images)} OCR 처리 시작...")
            logger.info(f"페이지 {i+1}/{len(images)} OCR 처리 중...")
            
            # 이미지를 임시 파일로 저장 (PaddleOCR은 이미지 객체보다 파일 경로로 처리가 더 안정적)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # 이미지가 너무 크면 리사이징 (성능 향상)
                max_dim = 3000
                if image.width > max_dim or image.height > max_dim:
                    ratio = min(max_dim / image.width, max_dim / image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.LANCZOS)
                
                # 이미지 저장
                image.save(temp_path, 'JPEG', quality=95)
                
            try:
                # OCR 처리
                ocr = get_paddle_ocr()
                result = ocr.ocr(temp_path, cls=True)
                
                # 결과 텍스트 추출 및 정렬
                extracted_texts = []
                if result:
                    for line_result in result:
                        if not line_result:
                            continue
                            
                        line_texts = []
                        for box, (text, confidence) in line_result:
                            if confidence >= min_confidence:
                                line_texts.append(text)
                        
                        if line_texts:
                            extracted_texts.append(" ".join(line_texts))
                
                # 결과 텍스트 구성
                page_text = "\n".join(extracted_texts)
                
                # 후처리: 불필요한 줄바꿈, 공백 정리
                page_text = re.sub(r'\s*\n\s*', '\n', page_text)
                page_text = re.sub(r' +', ' ', page_text)
                
                # 임시 파일 삭제
                os.unlink(temp_path)
                
                ocr_processing_logger.info(f"[PDF OCR INFO] 페이지 {i+1}/{len(images)} OCR 처리 완료, 글자수: {len(page_text)}")
                page_texts.append(page_text)
            except Exception as e:
                # 오류 발생 시 임시 파일 삭제 시도
                ocr_processing_logger.error(f"[PDF OCR ERROR] 페이지 {i+1}/{len(images)} OCR 처리 중 오류: {e}", exc_info=True)
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e
        
        # 모든 페이지 텍스트 결합
        all_text = ""
        for i, page_text in enumerate(page_texts):
            all_text += f"--- 페이지 {i+1} ---\n{page_text}\n"
        
        logger.info(f"PDF OCR 처리 완료: {len(all_text)} 글자")
        ocr_processing_logger.info(f"[PDF OCR SUCCESS] 전체 PDF OCR 처리 완료: {pdf_path}, 총 글자수: {len(all_text)}")
        
        return all_text
    
    except Exception as e:
        ocr_processing_logger.error(f"[PDF OCR ERROR] PDF 처리 중 심각한 오류 발생: {pdf_path}, 오류: {e}", exc_info=True)
        traceback.print_exc()
        return ""

def extract_text_from_file_sync(file_path: str, min_confidence: float = 0.5) -> str:
    """
    파일 확장자에 따라 적절한 텍스트 추출 방식을 적용합니다.
    
    Args:
        file_path: 파일 경로
        min_confidence: 최소 신뢰도 (0.0 ~ 1.0)
        
    Returns:
        추출된 텍스트
    """
    file_ext = Path(file_path).suffix.lower()
    
    try:
        # 텍스트 파일 처리 (OCR 없이 직접 읽기)
        if file_ext == '.txt':
            logger.info(f"텍스트 파일 직접 읽기: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        # 이미지 파일 처리
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
            logger.info(f"이미지 파일 OCR 처리 중: {file_path}")
            return extract_text_from_image_sync(file_path, min_confidence)
        
        # PDF 파일 처리
        elif file_ext == '.pdf':
            logger.info(f"PDF 파일 처리 중: {file_path}")
            return extract_text_from_pdf_with_ocr_sync(file_path, min_confidence)
        
        # Office 파일 처리 (PDF로 변환 후 OCR)
        elif file_ext in ['.pptx', '.xlsx', '.docx']:
            logger.info(f"Office 파일 처리 중: {file_path}")
            from .indexing_utils import convert_office_to_pdf_sync
            temp_dir = "temp_conversions"
            pdf_path = convert_office_to_pdf_sync(file_path, temp_dir)
            if pdf_path:
                logger.info(f"Office 파일을 PDF로 변환 성공: {pdf_path}")
                return extract_text_from_pdf_with_ocr_sync(pdf_path, min_confidence)
            else:
                logger.error(f"Office 파일을 PDF로 변환 실패: {file_path}")
                return None
        
        # 기타 파일은 None 반환 (기존 로더 사용)
        else:
            logger.info(f"OCR이 지원하지 않는 파일 형식: {file_ext}")
            return None
            
    except Exception as e:
        logger.error(f"파일 텍스트 추출 중 오류 발생: {file_path}, 오류: {e}")
        traceback.print_exc()
        return None

# 텍스트 인식 향상을 위한 이미지 전처리 함수
def preprocess_image_for_ocr(image_path: str) -> np.ndarray:
    """
    OCR 인식률을 높이기 위한 이미지 전처리 함수
    
    Args:
        image_path: 입력 이미지 경로
        
    Returns:
        전처리된 이미지 (OpenCV 형식)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 이미지 크기 조정 (OCR 성능 향상)
    max_dim = 3000
    h, w = image.shape[:2]
    if w > max_dim or h > max_dim:
        ratio = min(max_dim / w, max_dim / h)
        new_size = (int(w * ratio), int(h * ratio))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    # 노이즈 제거를 위한 가우시안 블러
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 대비 향상
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_img


async def extract_images_from_pdf_with_layout(pdf_path: str, doc_id: str) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """PDF 페이지별 이미지 저장 + PP-Structure 문서 파싱"""
    
    # PDF → 이미지 변환
    images = convert_from_path(pdf_path, dpi=200, thread_count=2)
    # UUID 제거된 clean doc_id 사용
    from .indexing_utils import strip_uuid_prefix
    clean_doc_id = strip_uuid_prefix(doc_id) if doc_id else "unknown"
    if '.' in clean_doc_id:
        clean_doc_id = clean_doc_id.rsplit('.', 1)[0]
    
    # 저장 디렉토리
    img_dir = Path("app/static/document_images") / clean_doc_id
    img_dir.mkdir(parents=True, exist_ok=True)
    
    image_texts = {}
    image_paths = {}
    
    # PP-Structure 인스턴스
    structure = get_paddle_structure()
    ocr = get_paddle_ocr()
    
    for page_num, page_image in enumerate(images, 1):
        # 이미지 저장
        img_name = f"page_{page_num}.png"
        img_path = img_dir / img_name
        page_image.save(img_path, 'PNG')
        
        # PP-Structure 실행
        import numpy as np
        img_array = np.array(page_image)
        page_texts = []
        
        try:
            if structure is not None:
                layout_result = structure(img_array)
                
                # 텍스트 추출
                for region in layout_result:
                    region_type = region.get('type', '')
                    
                    if region_type in ['text', 'title']:
                        # res가 리스트이므로 각 OCR 결과에서 텍스트 추출
                        res_list = region.get('res', [])
                        if isinstance(res_list, list):
                            for ocr_item in res_list:
                                if isinstance(ocr_item, dict):
                                    text = ocr_item.get('text', '').strip()
                                    if text:
                                        prefix = "[제목]" if region_type == 'title' else "[텍스트]"
                                        page_texts.append(f"{prefix} {text}")
                    
                    elif region_type == 'table':
                        # 테이블의 경우 html 키가 있는지 확인 후 처리
                        res_list = region.get('res', [])
                        if isinstance(res_list, list):
                            for ocr_item in res_list:
                                if isinstance(ocr_item, dict) and 'html' in ocr_item:
                                    html = ocr_item.get('html', '')
                                    if html:
                                        table_text = parse_table_html_to_text(html)
                                        page_texts.append(f"[테이블] {table_text}")
                                elif isinstance(ocr_item, dict):
                                    # html이 없으면 일반 텍스트로 처리
                                    text = ocr_item.get('text', '').strip()
                                    if text:
                                        page_texts.append(f"[테이블텍스트] {text}")
                    
                    elif region_type == 'figure':
                        bbox = region.get('bbox', [])
                        page_texts.append(f"[이미지] 좌표: {bbox}")
        except Exception as e:
            logger.error(f"PP-Structure 처리 중 오류: {e}")
            traceback.print_exc()
        
        # PP-Structure 결과 없거나 오류 발생 시 일반 OCR
        if not page_texts:
            try:
                ocr_result = ocr.ocr(img_array, cls=True)
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                            confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.0
                            
                            if confidence > 0.6 and text.strip():
                                page_texts.append(f"[OCR] {text.strip()}")
            except Exception as e:
                logger.error(f"OCR 처리 중 오류: {e}")
                traceback.print_exc()
        
        # 결과 저장
        image_texts[page_num] = page_texts
        relative_path = f"document_images/{clean_doc_id}/{img_name}"
        image_paths[page_num] = [relative_path]
    
    return image_texts, image_paths


def parse_table_html_to_text(html_content: str) -> str:
    """HTML 테이블을 텍스트로 변환"""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    
    if not table:
        return html_content
    
    rows = []
    for tr in table.find_all('tr'):
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        if cells:
            rows.append(' | '.join(cells))
    
    return '\n'.join(rows)

# save_ocr_result 함수 제거됨 - race condition 방지
