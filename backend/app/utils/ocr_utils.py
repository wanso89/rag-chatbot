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
from typing import List, Dict, Any
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

# 로깅 설정
logger = logging.getLogger(__name__)
ocr_processing_logger = logging.getLogger("ocr_processing") # 새로운 로거
ocr_processing_logger.setLevel(logging.INFO)
# 파일 핸들러 설정 (필요시)
# import logging.handlers
# file_handler = logging.handlers.RotatingFileHandler('ocr_process.log', maxBytes=1024*1024, backupCount=5)
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# ocr_processing_logger.addHandler(file_handler)

# 전역 PaddleOCR 인스턴스 (초기화는 시간이 많이 걸리므로 싱글톤으로 유지)
_paddle_ocr_instance = None

# PaddleOCR 로케일 설정 (한국어 + 영어)
PADDLE_LANG = "korean"  # 한국어 지원, 영어도 함께 인식함

def get_paddle_ocr():
    """
    PaddleOCR 인스턴스를 싱글톤 패턴으로 가져옵니다.
    """
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            logger.info("PaddleOCR 인스턴스 초기화 중...")
            # 한국어(+영어) 설정으로 PaddleOCR 초기화
            # use_angle_cls: 기울어진 텍스트 감지
            # lang: 언어 설정 (korean은 한국어/영어 모델 사용)
            # use_gpu: 가능하면 GPU 사용
            # show_log: 로깅 비활성화
            _paddle_ocr_instance = paddleocr.PaddleOCR(
                use_angle_cls=True, 
                lang=PADDLE_LANG,
                use_gpu=True,
                det_model_dir=None,  # 기본 모델 사용
                rec_model_dir=None,  # 기본 모델 사용
                cls_model_dir=None,  # 기본 모델 사용
                show_log=False
            )
            logger.info("PaddleOCR 인스턴스 초기화 완료")
        except Exception as e:
            logger.error(f"PaddleOCR 초기화 오류: {e}")
            traceback.print_exc()
            # 실패 시 빈 인스턴스 반환 대신 예외 발생
            raise
    return _paddle_ocr_instance

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

async def extract_text_from_image(image_path: str, min_confidence: float = 0.5) -> str:
    """
    이미지 파일에서 텍스트를 추출합니다.
    
    Args:
        image_path: 이미지 파일 경로
        min_confidence: 최소 신뢰도 (0.0 ~ 1.0)
        
    Returns:
        추출된 텍스트
    """
    try:
        loop = asyncio.get_event_loop()
        
        # 이미지 로드 및 OCR 처리를 비동기로 실행
        def process_image():
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
            
        return await loop.run_in_executor(None, process_image)
    except Exception as e:
        logger.error(f"이미지에서 텍스트 추출 중 오류 발생: {e}")
        traceback.print_exc()
        return ""

async def extract_text_from_pdf_with_ocr(pdf_path: str, min_confidence: float = 0.5) -> str:
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
        loop = asyncio.get_event_loop()
        extracted_text_raw = await loop.run_in_executor(None, lambda: pdfminer_extract_text(pdf_path))
        
        # 추출된 텍스트에서 실제 유효 문자 수 확인
        meaningful_text_threshold = 10  # 실제 의미있는 문자의 최소 개수
        valid_text_for_skip_ocr = False
        if extracted_text_raw:
            # 공백, 줄바꿈, form feed 등 제외하고 실제 문자만 카운트
            # 정규표현식을 사용하여 한글, 영어 알파벳, 숫자만 카운트
            meaningful_chars = re.sub(r'[^a-zA-Z0-9가-힣]', '', extracted_text_raw)
            num_meaningful_chars = len(meaningful_chars)
            
            total_chars_no_whitespace = len(re.sub(r'\\s+', '', extracted_text_raw))

            logger.info(f"PDFMiner 추출: 총 문자(공백제거): {total_chars_no_whitespace}, 유효 문자(a-zA-Z0-9가-힣): {num_meaningful_chars}")
            ocr_processing_logger.info(f"[PDF OCR INFO] PDFMiner 추출 결과 - 총 문자(공백제거): {total_chars_no_whitespace}, 유효 문자: {num_meaningful_chars} (임계값: {meaningful_text_threshold})")

            if total_chars_no_whitespace >= 30 and num_meaningful_chars >= meaningful_text_threshold:
                valid_text_for_skip_ocr = True
        
        if valid_text_for_skip_ocr:
            logger.info(f"PDF에서 텍스트 직접 추출 성공 (유효 문자 충분): {len(extracted_text_raw)} 글자")
            ocr_processing_logger.info(f"[PDF OCR SUCCESS] 직접 텍스트 추출 성공 (PDFMiner): {pdf_path}, 글자수: {len(extracted_text_raw)}")
            return extracted_text_raw
            
        # 2. 직접 추출이 불충분한 경우, OCR 적용
        logger.info(f"직접 추출 불충분 (또는 유효 문자 부족). PaddleOCR 적용 중: {pdf_path}")
        ocr_processing_logger.info(f"[PDF OCR INFO] 직접 추출 불충분/유효문자 부족, 이미지 변환 및 PaddleOCR 진행: {pdf_path}")
        
        # PDF를 이미지로 변환
        def convert_pdf_to_images():
            ocr_processing_logger.info(f"[PDF OCR INFO] PDF -> 이미지 변환 시작: {pdf_path}")
            images = convert_from_path(
                pdf_path,
                dpi=300,  # 해상도 (높을수록 더 정확하지만 처리 시간 증가)
                thread_count=4,  # 멀티스레딩
                use_pdftocairo=True,  # pdftocairo 사용 (더 빠르고 정확함)
                grayscale=False,  # 컬러 유지 (표와 도표 인식 향상)
                transparent=False  # 투명도 제거
            )
            ocr_processing_logger.info(f"[PDF OCR INFO] PDF -> 이미지 변환 완료: {pdf_path}, 페이지 수: {len(images)}")
            return images
        
        # 비동기로 PDF를 이미지로 변환
        images = await loop.run_in_executor(None, convert_pdf_to_images)
        logger.info(f"PDF 이미지 변환 완료: {len(images)} 페이지")
        
        # 각 이미지에 OCR 적용 (병렬 처리)
        async def process_page(i, image):
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
                return page_text
            except Exception as e:
                # 오류 발생 시 임시 파일 삭제 시도
                ocr_processing_logger.error(f"[PDF OCR ERROR] 페이지 {i+1}/{len(images)} OCR 처리 중 오류: {e}", exc_info=True)
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e
        
        # 각 페이지 처리 작업 생성
        tasks = [process_page(i, image) for i, image in enumerate(images)]
        
        # 병렬로 모든 페이지 처리
        page_texts = await asyncio.gather(*tasks)
        
        # 모든 페이지 텍스트 결합
        all_text = ""
        for i, page_text in enumerate(page_texts):
            all_text += f"--- 페이지 {i+1} ---\n{page_text}\n"
        
        logger.info(f"PDF OCR 처리 완료: {len(all_text)} 글자")
        ocr_processing_logger.info(f"[PDF OCR SUCCESS] 전체 PDF OCR 처리 완료: {pdf_path}, 총 글자수: {len(all_text)}")
        return all_text
    
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 중 오류 발생: {e}")
        ocr_processing_logger.error(f"[PDF OCR ERROR] PDF 처리 중 심각한 오류 발생: {pdf_path}, 오류: {e}", exc_info=True)
        traceback.print_exc()
        return ""

async def extract_text_from_file(file_path: str, min_confidence: float = 0.5) -> str:
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
        # 이미지 파일 처리
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
            logger.info(f"이미지 파일 OCR 처리 중: {file_path}")
            return await extract_text_from_image(file_path, min_confidence)
        
        # PDF 파일 처리
        elif file_ext == '.pdf':
            logger.info(f"PDF 파일 처리 중: {file_path}")
            return await extract_text_from_pdf_with_ocr(file_path, min_confidence)
        
        # 기타 파일은 None 반환 (기존 로더 사용)
        else:
            logger.info(f"OCR이 지원하지 않는 파일 형식: {file_ext}")
            return None
            
    except Exception as e:
        logger.error(f"파일 텍스트 추출 중 오류 발생: {file_path}, 오류: {e}")
        traceback.print_exc()
        return None

# 표 및 구조화된 데이터 추출 (향상된 표 감지 로직)
async def detect_tables_in_image(image_path: str) -> List[Dict[str, Any]]:
    """
    이미지에서 표를 감지하고 구조화된 데이터로 변환합니다.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        감지된 표 목록 (각 표는 행과 열로 구성된 딕셔너리)
    """
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 이미지 이진화
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 수평 및 수직 선 감지
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.erode(thresh, horizontal_kernel, iterations=3)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)
        
        vertical_lines = cv2.erode(thresh, vertical_kernel, iterations=3)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=3)
        
        # 수평 및 수직 선 결합
        table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # 표 윤곽선 찾기
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 결과 리스트
        tables = []
        
        for contour in contours:
            # 표 영역 추출
            x, y, w, h = cv2.boundingRect(contour)
            
            # 너무 작은 영역은 무시
            if w < 100 or h < 100:
                continue
                
            # 표 영역 이미지 추출
            table_roi = image[y:y+h, x:x+w]
            
            # PaddleOCR로 표 내용 추출
            ocr = get_paddle_ocr()
            result = ocr.ocr(table_roi, cls=True)
            
            # 추출된 텍스트 위치 기반으로 표 구조화
            cells = []
            if result:
                for idx, line_result in enumerate(result[0]):
                    if not line_result:
                        continue
                        
                    for box, (text, confidence) in line_result:
                        # 좌표는 상대적 위치이므로 절대 위치로 변환
                        points = np.array(box)
                        points[:, 0] += x
                        points[:, 1] += y
                        
                        # 셀 정보 저장
                        cells.append({
                            "box": points.tolist(),
                            "text": text,
                            "confidence": confidence
                        })
            
            # 표 정보 추가
            tables.append({
                "bbox": [x, y, x+w, y+h],
                "cells": cells
            })
            
        return tables
        
    except Exception as e:
        logger.error(f"표 탐지 중 오류 발생: {e}")
        traceback.print_exc()
        return []

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

# DOCX 문서에서 텍스트 추출 (future enhancement)
async def extract_text_from_docx(docx_path: str) -> str:
    """
    DOCX 문서에서 텍스트를 추출합니다. 실패 시 OCR로 처리합니다.
    
    Args:
        docx_path: DOCX 파일 경로
        
    Returns:
        추출된 텍스트
    """
    # 이 기능은 향후 확장을 위한 플레이스홀더입니다.
    return None 