# indexing_utils.py
import time
import subprocess
import os, re, asyncio
import hashlib  # 파일 중복 체크를 위한 해시 라이브러리 추가
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import markdown
from io import StringIO
import torch
from transformers import AutoTokenizer

from langchain_community.document_loaders import (
    UnstructuredExcelLoader,
    TextLoader,
    PyPDFLoader,  # PyPDFLoader 임포트
    UnstructuredFileLoader,
)
import fitz
from langchain.schema import Document
from elasticsearch.helpers import bulk
import traceback
from datetime import datetime
import logging

from .ocr_utils import (
        extract_text_from_file,
        extract_text_from_image,
        extract_text_from_pdf_with_ocr,
)
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
    print("✅ Docling 사용 가능")
except ImportError:
    DOCLING_AVAILABLE = False
    print("⚠️ Docling 없음, 기존 방식 사용")

# PyPDF 체크 (기존에 있을 수도 있음)
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False

# 전역 Docling 인스턴스 (한번만 초기화)
_docling_layout_converter = None
_last_docling_init_time = None

# 로깅 설정
logger = logging.getLogger(__name__)

ES_INDEX_NAME = "rag_documents_kure_v1"
IMAGE_DIR = "static/document_images"  # 사용 안되면 제거 가능
os.makedirs(IMAGE_DIR, exist_ok=True)  # 사용 안되면 제거 가능

# LOADER_MAPPING: 이미지 파일 확장자 추가
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {"mode": "paged"}),
    ".xls": (UnstructuredExcelLoader, {"mode": "paged"}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    
    # PPT 파일 추가
    ".ppt": (UnstructuredFileLoader, {"mode": "paged"}),
    ".pptx": (UnstructuredFileLoader, {"mode": "paged"}),
    
    # 이미지 파일
    ".jpg": ("OCR_LOADER", {}),
    ".jpeg": ("OCR_LOADER", {}),
    ".png": ("OCR_LOADER", {}),
    ".bmp": ("OCR_LOADER", {}),
    ".tiff": ("OCR_LOADER", {}),
    ".tif": ("OCR_LOADER", {}),
    ".webp": ("OCR_LOADER", {}),
}
#토크나이저
LLM_MODEL_NAME = r"/home/root/Gukbap-Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        use_fast=True,  # 빠른 토크나이저 사용
        padding_side="left",  # 왼쪽 패딩 (생성 모델에 적합)
        use_auth_token=None,  # 인증 토큰 불필요 시 명시적으로 None
        trust_remote_code=True,  # 원격 코드 신뢰 (일부 모델에 필요)
    )


# --- DOCX를 PDF로 변환하는 함수 (이전과 동일하게 유지) ---
def convert_docx_to_pdf_sync(docx_path: str, output_dir: str) -> Optional[str]:
    # (이전 답변의 libreoffice 사용하는 코드를 그대로 사용합니다.)
    try:
        print(
            f"DOCX를 PDF로 변환 시도 (libreoffice): '{docx_path}' -> '{output_dir}' 디렉토리로"
        )
        command = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            docx_path,
        ]
        env = os.environ.copy()
        env["HOME"] = "/tmp"
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=120, env=env
        )
        expected_pdf_filename = Path(docx_path).stem + ".pdf"
        converted_pdf_path = os.path.join(output_dir, expected_pdf_filename)
        if process.returncode == 0 and os.path.exists(converted_pdf_path):
            print(f"PDF 변환 성공 (libreoffice): '{converted_pdf_path}'")
            return converted_pdf_path
        else:
            print(f"PDF 변환 실패 (libreoffice). Return code: {process.returncode}")
            print(f"Stdout: {process.stdout.strip()}")
            print(f"Stderr: {process.stderr.strip()}")
            if os.path.exists(converted_pdf_path):
                try:
                    os.remove(converted_pdf_path)
                except Exception as e_rem:
                    print(f"실패한 PDF 파일 삭제 중 오류: {e_rem}")
            return None
    except FileNotFoundError:  # libreoffice가 설치되지 않았거나 경로에 없을 때
        print(
            "PDF 변환 실패: 'libreoffice' 명령어를 찾을 수 없습니다. 서버에 libreoffice가 설치되어 있는지 확인하세요."
        )
        return None
    except subprocess.TimeoutExpired:
        print(f"PDF 변환 시간 초과 (libreoffice): {docx_path}")
        return None
    except Exception as e:
        print(f"DOCX -> PDF 변환 중 예외 발생 (libreoffice, {docx_path}): {e}")
        traceback.print_exc()
        return None


async def convert_docx_to_pdf(docx_path: str, output_dir: str) -> Optional[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, convert_docx_to_pdf_sync, docx_path, output_dir
    )


# --- 파일 내용을 읽어 Langchain Document 객체 리스트로 만드는 함수 (OCR 기능 추가) ---
async def load_document(file_path_to_load: str, loader_selector_ext: str) -> List[Document]:
    print(
        f"load_document 호출: file_path_to_load='{file_path_to_load}', loader_selector_ext='{loader_selector_ext}'"
    )

    loader_info = LOADER_MAPPING.get(loader_selector_ext)

    # OCR 로더 처리
    if loader_info and loader_info[0] == "OCR_LOADER":
        logger.info(f"OCR 로더를 사용하여 파일 처리: {file_path_to_load}")
        return await load_document_with_ocr(file_path_to_load)
    
    # PDF 파일 처리 강화 (OCR 보조)
    if loader_selector_ext == '.pdf':
        # 먼저 기존 PyPDFLoader로 처리 시도
        pdf_loader_class, pdf_loader_kwargs = loader_info
        pdf_loader = pdf_loader_class(file_path_to_load)
        
        try:
            docs = pdf_loader.load()
            
            # 추출된 텍스트가 충분한지 확인
            total_text = "".join([doc.page_content for doc in docs])
            clean_text = clean_ocr_text(total_text)
            
            
            if len(clean_text) < 100:  # 텍스트가 충분하지 않으면 PaddleOCR 시도
                logger.info(f"PDF에서 추출된 텍스트가 부족함 ({len(clean_text)} 글자). PaddleOCR 시도: {file_path_to_load}")
                ocr_docs = await load_document_with_ocr(file_path_to_load)
                
                if ocr_docs and len(ocr_docs) > 0:
                    logger.info(f"PaddleOCR을 통해 PDF에서 텍스트 추출 성공: {len(ocr_docs)} 페이지")
                    return ocr_docs
            
            # 기존 로더로 충분한 텍스트 추출에 성공한 경우
            print(f"DEBUG (load_document): 총 {len(docs)}개의 Document 객체 로드됨 (path: {file_path_to_load})")
            for i, loaded_doc in enumerate(docs):
                print(f"  Loaded doc {i} metadata: {loaded_doc.metadata}")
                
            processed_docs = []
            for doc_idx, doc in enumerate(docs):
                cleaned_content = doc.page_content
                cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
                
                if cleaned_content:
                    doc.page_content = cleaned_content
                    doc.metadata["loaded_at"] = datetime.now().isoformat()
                    
                    # 페이지 번호 설정
                    page_number_to_set = None
                    if "page" in doc.metadata and isinstance(doc.metadata["page"], int):
                        page_number_to_set = doc.metadata["page"] + 1
                    else:
                        print(f"Warning: PyPDFLoader가 Doc {doc_idx}의 'page' 메타데이터를 제공하지 않음. 순번 사용.")
                        page_number_to_set = doc_idx + 1
                        
                    doc.metadata["page"] = int(page_number_to_set)
                    processed_docs.append(doc)
            processed_docs = merge_table_chunks(processed_docs)        
            return processed_docs
            
        except Exception as e:
            logger.error(f"기본 PDF 로더 실패, PaddleOCR 시도: {file_path_to_load}, 오류: {e}")
            return await load_document_with_ocr(file_path_to_load)

    if (
        not loader_info
    ):  # LOADER_MAPPING에 없는 확장자 (예: DOCX 변환 실패 후 원본 .docx)
        print(
            f"LOADER_MAPPING에서 '{loader_selector_ext}' 로더를 찾지 못함. UnstructuredFileLoader로 시도: {file_path_to_load}"
        )
        if not os.path.isfile(file_path_to_load):
            print(
                f"ERROR (load_document): file_path_to_load '{file_path_to_load}'는 실제 파일이 아닙니다."
            )
            return []
        # UnstructuredFileLoader는 페이지 정보를 제대로 주지 않을 가능성이 높음
        loader = UnstructuredFileLoader(
            file_path_to_load
        )  # mode="paged"는 PDF 외에는 의미 없을 수 있음
    else:
        loader_class, loader_kwargs = loader_info
        if loader_class == PyPDFLoader:  # PyPDFLoader 특별 처리
            loader = loader_class(file_path_to_load)
        else:  # 다른 로더들 (Excel, Text 등)
            loader = loader_class(file_path_to_load, **loader_kwargs)

    try:
        docs = loader.load()  # 파일 로드! PyPDFLoader는 페이지별로 Document 객체 생성

        # --- 로드된 문서 디버깅 로그 (매우 중요!) ---
        print(
            f"DEBUG (load_document): 총 {len(docs)}개의 Document 객체 로드됨 (path: {file_path_to_load})"
        )
        for i, loaded_doc in enumerate(docs):
            print(f"  Loaded doc {i} metadata: {loaded_doc.metadata}")
        # --- 디버깅 로그 끝 ---

        processed_docs = []
        for doc_idx, doc in enumerate(
            docs
        ):  # doc_idx는 로드된 Document 객체의 순서 (0부터 시작)
            # 원본 코드의 전처리 로직 적용 (신중하게)
            cleaned_content = doc.page_content
            # cleaned_content = re.sub(r"Cloudera 운영자메뉴얼|Version \d+\.\d+|Page \d+/\d+|네오오토|취업규칙", "", cleaned_content)
            cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()
            # cleaned_content = re.sub(r"(습니다|합니다|입니다)\s*", " ", cleaned_content) # 문맥 왜곡 가능성

            if cleaned_content:
                doc.page_content = cleaned_content
                doc.metadata["loaded_at"] = datetime.now().isoformat()  # 로드 시간 기록

                # --- 페이지 번호 설정 (핵심!) ---
                page_number_to_set = None
                if loader_selector_ext == ".pdf":  # PyPDFLoader를 사용한 경우
                    # PyPDFLoader는 metadata에 'page' 키로 0부터 시작하는 페이지 번호를 줌
                    if "page" in doc.metadata and isinstance(doc.metadata["page"], int):
                        page_number_to_set = (
                            doc.metadata["page"] + 1
                        )  # 1부터 시작하도록 +1
                    else:  # PyPDFLoader가 페이지 정보를 못 준 경우 (거의 없음)
                        print(
                            f"Warning (load_document): PyPDFLoader가 Doc {doc_idx}의 'page' 메타데이터를 제공하지 않음. 순번 사용."
                        )
                        page_number_to_set = doc_idx + 1
                elif (
                    "page_number" in doc.metadata
                ):  # 다른 Unstructured 로더가 'page_number'를 줄 경우
                    page_number_to_set = doc.metadata["page_number"]
                else:  # 페이지 정보를 어떤 로더에서도 얻지 못한 경우
                    print(
                        f"Warning (load_document): Doc {doc_idx}에서 페이지 정보를 찾을 수 없음. loader: {loader_selector_ext}. 페이지 1로 설정."
                    )
                    page_number_to_set = 1  # 기본값 1로 설정 (단일 페이지 문서로 간주)

                doc.metadata["page"] = int(
                    page_number_to_set
                )  # 최종적으로 'page' 키에 정수형으로 저장
                # --- 페이지 번호 설정 끝 ---

                processed_docs.append(doc)
        processed_docs = merge_table_chunks(processed_docs)         
        return processed_docs
    except Exception as e:
        print(
            f"파일 로딩 중 오류 발생 ({file_path_to_load}, loader_ext: {loader_selector_ext}): {e}"
        )
        traceback.print_exc()
        
        # 로딩 실패 시 PaddleOCR 시도 (새로운 코드)
        if loader_selector_ext != "OCR_LOADER":  # OCR 로더가 아닌 경우에만 시도
            logger.info(f"일반 로더 실패, PaddleOCR 시도: {file_path_to_load}")
            try:
                ocr_docs = await load_document_with_ocr(file_path_to_load)
                if ocr_docs and len(ocr_docs) > 0:
                    logger.info(f"PaddleOCR을 통해 텍스트 추출 성공: {len(ocr_docs)} 페이지")
                    return ocr_docs
            except Exception as ocr_e:
                logger.error(f"PaddleOCR 대체 시도 실패: {file_path_to_load}, 오류: {ocr_e}")
                
        return []

def crop_pdf_region(pdf_path: str, page_num: int, bbox) -> bytes:
    """PDF 특정 영역을 이미지로 크롭"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    
    # bbox를 fitz 좌표계로 변환
    rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
    img_data = pix.tobytes("png")
    doc.close()
    
    return img_data

async def extract_image_ocr_texts(pdf_path: str, layout_info) -> Dict[int, List[str]]:
    """Docling 영역으로 이미지 OCR - 실패한 이미지는 제외"""
    image_texts = {}
    
    if layout_info and hasattr(layout_info, 'pictures'):
        for image_data in layout_info.pictures:
            page_num = getattr(image_data, 'page', 1)
            bbox = getattr(image_data, 'bbox', None)
            
            if bbox:
                img_data = crop_pdf_region(pdf_path, page_num, bbox)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                    tmp.write(img_data)
                    tmp.flush()
                    extracted_text = await extract_text_from_image(tmp.name, 0.7)
                    
                    # 최소 길이 체크 (의미있는 텍스트만)
                    if extracted_text and len(extracted_text.strip()) > 5:
                        if page_num not in image_texts:
                            image_texts[page_num] = []
                        image_texts[page_num].append(extracted_text)
    
    return image_texts

def get_docling_converter():
    """Docling 컨버터 싱글톤 (메모리 효율)"""
    global _docling_layout_converter, _last_docling_init_time
    
    if not DOCLING_AVAILABLE:
        return None
    
    if _docling_layout_converter is None:
        try:
            # 레이아웃 분석만 (OCR 없이)
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = False  # 속도를 위해 OCR 비활성화
            pdf_options.do_table_structure = True  # 테이블 구조만
            pdf_options.table_structure_options.do_cell_matching = True
            
            _docling_layout_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
                }
            )
            _last_docling_init_time = datetime.now()
            print(f"✅ Docling 레이아웃 분석기 초기화 완료")
        except Exception as e:
            print(f"⚠️ Docling 초기화 실패: {e}")
            _docling_layout_converter = None
    
    return _docling_layout_converter

def extract_pdf_text_fast(file_path: str) -> Dict[int, str]:
    """빠른 PDF 텍스트 추출 (기존 방식)"""
    pdf_texts = {}
    
    if not PYPDF_AVAILABLE:
        return pdf_texts
    
    try:
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                cleaned_text = re.sub(r'\s+', ' ', text).strip()
                pdf_texts[page_num] = cleaned_text
    except Exception as e:
        print(f"⚠️ PDF 텍스트 추출 실패: {e}")
    
    return pdf_texts

def create_image_caption_from_context(
    page_text: str, 
    image_index: int, 
    total_images: int,
    previous_sentences: List[str] = None
) -> str:
    """이미지 캡션을 주변 텍스트에서 생성 (LLAVA 대신)"""
    
    # 기본 캡션
    base_caption = f"이미지 {image_index + 1}"
    
    if not page_text:
        return base_caption
    
    # 페이지 텍스트를 문장으로 분할
    sentences = re.split(r'[.!?]\s+', page_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 이미지 관련 키워드 찾기
    image_keywords = ['그림', '표', '도표', '차트', '이미지', '사진', '스크린샷', '화면', '예시', '참고']
    
    context_sentences = []
    
    # 1. 이미지 번호가 언급된 문장 찾기
    for sentence in sentences:
        if f"그림 {image_index + 1}" in sentence or f"Figure {image_index + 1}" in sentence:
            context_sentences.append(sentence)
            break
    
    # 2. 이미지 키워드가 있는 문장들 찾기
    if not context_sentences:
        for sentence in sentences:
            if any(keyword in sentence for keyword in image_keywords):
                context_sentences.append(sentence)
                if len(context_sentences) >= 2:  # 최대 2개 문장
                    break
    
    # 3. 이전 문장과 연관 (사용자 요청)
    if previous_sentences and not context_sentences:
        # 이전 2-3 문장을 컨텍스트로 사용
        recent_context = previous_sentences[-2:] if len(previous_sentences) >= 2 else previous_sentences
        if recent_context:
            context_text = " ".join(recent_context)
            return f"{base_caption}: {context_text}과 연관된 이미지"
    
    # 4. 컨텍스트 문장들을 캡션으로 사용
    if context_sentences:
        context_text = " ".join(context_sentences[:1])  # 첫 번째 문장만
        # 너무 길면 자르기
        if len(context_text) > 100:
            context_text = context_text[:100] + "..."
        return f"{base_caption}: {context_text}"
    
    # 5. 기본값: 페이지의 첫 문장과 연관
    if sentences:
        first_sentence = sentences[0]
        if len(first_sentence) > 50:
            first_sentence = first_sentence[:50] + "..."
        return f"{base_caption}: '{first_sentence}'과 연관된 이미지"
    
    return base_caption

def merge_table_chunks(docs: List[Document]) -> List[Document]:
    if not docs:
        return docs

    merged, buf = [], []
    table_pat = re.compile(r'[\|\t]')        # 파이프·탭 포함 여부

    def flush_table_buf():
        """buf에 모인 표 조각을 하나로 병합해 merged에 추가"""
        if not buf:
            return
        first = buf[0]

        # ─── ① one-liner(캡션)가 있으면 첫 줄만 따로 꺼낸다 ───
        head = ""
        if first.page_content.startswith("이 행은"):
            head, _ = first.page_content.split("\n", 1)
            head += " | "

        # ─── ② 본문 병합 + 캡션 보존 ───
        merged_text = clean_ocr_text(" ".join(x.page_content for x in buf))
        first.page_content = head + merged_text

        # ─── ③ element_type 유지 ───
        first.metadata["element_type"] = "table"

        merged.append(first)
        buf.clear()

    # ──────────────────────────────────────────
    for d in docs:
        text = d.page_content
        looks_like_table = (
            len(text) < 400
            or any(k in text for k in ["표", "table", "차트"])
            or bool(table_pat.search(text))
        )

        if looks_like_table:
            buf.append(d)
            continue

        # 표 블록 종료 지점
        flush_table_buf()
        merged.append(d)

    flush_table_buf()          # 파일 끝에 표로 끝난 경우
    return merged

# 기존 load_document_with_ocr 함수를 하이브리드 방식으로 교체
# 하지만 기존 인터페이스는 완전히 유지
async def load_document_with_ocr(file_path: str) -> List[Document]:
    """
    기존 함수 시그니처 유지하면서 하이브리드 방식 적용
    
    Args:
        file_path: 파일 경로
        
    Returns:
        Document 객체 리스트 (기존과 동일한 형태)
    """
    try:
        logger.info(f"하이브리드 방식으로 파일 처리 중: {file_path}")
        file_extension = Path(file_path).suffix.lower()
        
        # PDF가 아니면 기존 방식 그대로
        if file_extension != '.pdf':
            logger.info(f"PDF가 아닌 파일은 기존 방식 사용: {file_path}")
            return await load_document_with_ocr_original(file_path)
        
        # PDF 하이브리드 처리
        start_time = time.time()
        
        
        # 1단계: Docling 레이아웃 분석 (OCR 없이)
        layout_info = None
        if DOCLING_AVAILABLE:
            try:
                converter = get_docling_converter()
                if converter:
                    layout_result = await asyncio.to_thread(converter.convert, file_path)
                    layout_info = layout_result.document
                    logger.info(f"✅ 레이아웃 분석 완료")
            except Exception as e:
                logger.warning(f"레이아웃 분석 실패, 기존 방식 사용: {e}")
        
        # 2단계: 빠른 PDF 텍스트 추출
        pdf_texts = extract_pdf_text_fast(file_path)
        # 3단계: 문서 생성 (기존 형태 유지)
        documents = []
        
        if pdf_texts:
            # PDF 텍스트가 있으면 하이브리드 방식
            documents = await create_hybrid_documents(
                pdf_texts, layout_info, file_path
            )
        else:
            # PDF 텍스트가 없으면 기존 OCR 방식으로 fallback
            logger.info(f"PDF 텍스트 없음, 기존 OCR 방식 사용: {file_path}")
            return await load_document_with_ocr_original(file_path)
        
        processing_time = time.time() - start_time
        logger.info(f"하이브리드 처리 완료: {len(documents)} 문서, {processing_time:.2f}초")
        
        return documents
        
    except Exception as e:
        logger.error(f"하이브리드 처리 실패, 기존 방식으로 fallback: {file_path}, 오류: {e}")
        return await load_document_with_ocr_original(file_path)
def clean_ocr_text(text: str) -> str:
    """OCR 결과에서 불필요한 숫자 패턴 제거"""
    if not text:
        return text
    
    # 1. 끝에 오는 숫자 패턴 제거 ("본인 및 배우자 부모 1 2." → "본인 및 배우자 부모")
    text = re.sub(r'\s+\d+\.?\s*$', '', text)
    text = re.sub(r'\s+\d+\s+\d+\.?\s*$', '', text) 
    text = re.sub(r'\s+\d+\s+\d+\s+\d+\.?\s*$', '', text)
    
    # 2. 중간에 있는 의미없는 숫자들 제거
    text = re.sub(r'\s+\d{1,2}\.\s*(?=\S)', ' ', text)  # "2. 자녀" → " 자녀"
    
    # 3. 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

async def create_hybrid_documents(
    pdf_texts: Dict[int, str],
    layout_info,
    file_path: str,
    *,
    llm_model=LLM_MODEL_NAME,          # ← Qwen (없으면 None)
    tokenizer=tokenizer           # ← Tokenizer (없으면 None)
) -> List[Document]:
    """
    ‣ 표  : Docling → markdown → 행 단위 청크
    ‣ 본문: 페이지 텍스트 그대로 단락-청킹
    ‣ 이미지: OCR 있으면 붙이고, 없으면 문맥 기반 캡션
    """
    documents: list[Document] = []
    file_name = Path(file_path).name

    # ──────────────────────────────
    # 1. Docling 구조 정보
    # ──────────────────────────────
    tables_info, images_info = [], []
    if layout_info:
        if hasattr(layout_info, "tables"):
            for i, tbl in enumerate(layout_info.tables):
                md = getattr(tbl, "export_to_markdown", lambda: "")() or ""
                tables_info.append(
                    {
                        "page": getattr(tbl, "page", 1),
                        "index": i,
                        "caption": getattr(tbl, "caption", "") or "",
                        "markdown": md,
                    }
                )
        if hasattr(layout_info, "pictures"):
            for i, fig in enumerate(layout_info.pictures):
                images_info.append(
                    {
                        "page": getattr(fig, "page", 1),
                        "index": i,
                        "caption": getattr(fig, "caption", "") or "",
                    }
                )

    # ──────────────────────────────
    # 2. 표 → 행 단위 청크
    # ──────────────────────────────
    for tbl in tables_info:
        if tbl["markdown"]:
            row_docs = await markdown_to_row_chunks(
                tbl["markdown"],
                page=tbl["page"],
                file_path=file_path,
                llm_model=llm_model,
                tokenizer=tokenizer,
            )
            documents.extend(row_docs)
    table_pages = {t["page"] for t in tables_info}
    table_pages |= {p + 1 for p in table_pages} 
    # ──────────────────────────────
    # 3. 이미지 OCR & 캡션
    # ──────────────────────────────
    image_ocr_texts = await extract_image_ocr_texts(file_path, layout_info)

    # ──────────────────────────────
    # 4. 본문 단락-단위 청킹 (이미지 텍스트·캡션 삽입)
    # ──────────────────────────────
    for page_num, page_text in pdf_texts.items():
        if page_num in table_pages:
            continue 
        if not page_text.strip():
            continue

        combined = page_text

        # 4-1) OCR 성공 이미지 텍스트
        if page_num in image_ocr_texts:
            combined += "\n\n" + "\n".join(image_ocr_texts[page_num])

        # 4-2) OCR 실패 이미지 → 문맥 캡션
        fail_imgs = [
            img
            for img in images_info
            if img["page"] == page_num and page_num not in image_ocr_texts
        ]
        if fail_imgs:
            for img in fail_imgs:
                cap = create_image_caption_from_context(page_text, img["index"], len(images_info))
                combined += f"\n\n이 이미지 {img['index']+1}: {cap}"

        # 4-3) 단락 분할
        for para in re.split(r"\n\s*\n", combined):
            para = para.strip()
            if len(para) < 10:
                continue
            documents.append(
                Document(
                    page_content=para,
                    metadata={
                        "source": file_name,
                        "page": page_num,
                        "element_type": "text",
                        "loaded_at": datetime.now().isoformat(),
                    },
                )
            )

    return documents

# 기존 함수 백업 (기존 로직 보존)
async def load_document_with_ocr_original(file_path: str) -> List[Document]:
    """
    PaddleOCR 엔진을 사용하여 이미지 또는 PDF 파일에서 텍스트를 추출하고 Document 객체로 변환합니다.
    
    Args:
        file_path: 파일 경로
        
    Returns:
        Document 객체 리스트
    """
    try:
        logger.info(f"PaddleOCR을 사용하여 파일 처리 중: {file_path}")
        file_extension = Path(file_path).suffix.lower()
        
        # 최소 신뢰도 설정 (0.0 ~ 1.0)
        min_confidence = 0.7
        
        # 파일 종류에 따라 적절한 OCR 함수 호출
        if file_extension == '.pdf':
            extracted_text = await extract_text_from_pdf_with_ocr(file_path, min_confidence)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
            extracted_text = await extract_text_from_image(file_path, min_confidence)
        else:
            extracted_text = await extract_text_from_file(file_path, min_confidence)
            
        if not extracted_text:
            logger.warning(f"PaddleOCR 텍스트 추출 실패: {file_path}")
            return []
            
        # 텍스트 정리
        cleaned_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        # 추출된 텍스트로 Document 객체 생성
        # 여러 페이지가 있으면 페이지마다 분리해서 Document 생성
        documents = []
        page_texts = re.split(r'---\s*페이지\s*(\d+)\s*---', cleaned_text)
        
        if len(page_texts) > 1:  # 페이지 구분자가 있는 경우
            # 홀수 인덱스는 페이지 번호, 짝수 인덱스는 텍스트 내용
            for i in range(1, len(page_texts), 2):
                if i + 1 < len(page_texts):
                    page_num = int(page_texts[i])
                    page_content = page_texts[i + 1].strip()
                    
                    if page_content:
                        documents.append(Document(
                            page_content=page_content,
                            metadata={
                                "source": file_path,
                                "page": page_num,
                                "loaded_at": datetime.now().isoformat(),
                                "ocr_processed": True,
                                "ocr_engine": "PaddleOCR"
                            }
                        ))
        else:  # 페이지 구분자가 없는 경우 (단일 페이지 처리)
            if cleaned_text:
                documents.append(Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": file_path,
                        "page": 1,
                        "loaded_at": datetime.now().isoformat(),
                        "ocr_processed": True,
                        "ocr_engine": "PaddleOCR"
                    }
                ))
        documents = merge_table_chunks(documents)
        logger.info(f"PaddleOCR 텍스트 추출 완료: {len(documents)} 페이지/섹션 생성")
        return documents
        
    except Exception as e:
        logger.error(f"PaddleOCR 문서 로드 중 오류 발생: {file_path}, 오류: {e}")
        traceback.print_exc()
        return []

# --- index_chunks_to_elasticsearch 함수 (페이지 번호 사용 명확화) ---
async def index_chunks_to_elasticsearch(
    es_client: Any, embedding_function: Any, chunks: List[Document], category: str
):
    # (이전 답변에서 제공된 index_chunks_to_elasticsearch 함수 코드와 거의 동일하게 유지)
    # 핵심: page_number_to_index = int(chunk_doc.metadata.get("page", 1)) # page 메타데이터 사용
    if not chunks:
        print("인덱싱할 청크가 없습니다.")
        return False
    batch_size = 500
    success_count = 0
    failure_count = 0

    async def process_batch(batch_chunks_input, batch_num_for_log):
        nonlocal success_count, failure_count
        valid_chunks_in_batch = [
            chk
            for chk in batch_chunks_input
            if chk.page_content and chk.page_content.strip()
        ]
        if not valid_chunks_in_batch:
            return
        chunk_texts_for_embedding = [chk.page_content for chk in valid_chunks_in_batch]
        try:
            embeddings = await asyncio.to_thread(
                embedding_function, chunk_texts_for_embedding
            )
            actions_for_bulk = []
            for i, chunk_doc in enumerate(valid_chunks_in_batch):
                page_number_to_index = chunk_doc.metadata.get("page")
                if (
                    page_number_to_index is None
                ):  # load_document에서 page를 못가져온 경우
                    print(
                        f"CRITICAL WARNING (index_chunks): Chunk (source: {chunk_doc.metadata.get('source', 'N/A')}, chunk_id: {chunk_doc.metadata.get('chunk_id', 'N/A')}) 에서 'page' 메타데이터를 찾을 수 없음! 기본값 1 사용."
                    )
                    page_number_to_index = 1

                source_filename = chunk_doc.metadata.get("source", "unknown_source")
                chunk_id_val = chunk_doc.metadata.get(
                    "chunk_id", f"batch{batch_num_for_log}_{i}"
                )
                es_doc_id = f"{source_filename.replace('.', '_')}_{page_number_to_index}_{chunk_id_val}"
                es_source_doc = {
                    "text": chunk_doc.page_content,
                    "embedding": embeddings[i],
                    "source": source_filename,
                    "page": int(page_number_to_index),
                    "category": category,
                    "chunk_id": chunk_id_val,
                    "total_chunks": chunk_doc.metadata.get("total_chunks", len(chunks)),
                    "indexed_at": datetime.now().isoformat(),
                    "element_type": chunk_doc.metadata.get("element_type", "text"),
                }
                actions_for_bulk.append(
                    {
                        "_index": ES_INDEX_NAME,
                        "_id": es_doc_id,
                        "_source": es_source_doc,
                    }
                )
            if actions_for_bulk:
                success_num, failed_items = await asyncio.to_thread(
                    bulk,
                    es_client,
                    actions_for_bulk,
                    chunk_size=len(actions_for_bulk),
                    request_timeout=180,
                    max_retries=3,
                    raise_on_error=False,
                )
                success_count += success_num
                if failed_items:
                    failure_count += len(failed_items)
                    print(
                        f"배치 {batch_num_for_log} 처리 중 {len(failed_items)}개 문서 인덱싱 실패."
                    )
        except Exception as e_batch:
            print(f"배치 {batch_num_for_log} 처리 중 예외 발생: {e_batch}")
            failure_count += len(valid_chunks_in_batch)
            traceback.print_exc()

    tasks = [
        process_batch(chunks[i : i + batch_size], (i // batch_size) + 1)
        for i in range(0, len(chunks), batch_size)
    ]
    await asyncio.gather(*tasks)
    print(
        f"인덱싱 완료: 총 {len(chunks)} 청크 중 {success_count}개 성공, {failure_count}개 실패"
    )
    return success_count > 0


# 파일 중복 체크를 위한 함수 개선
async def check_file_exists(es_client: Any, file_path: str) -> Tuple[bool, str]:
    """
    파일의 해시값을 계산하고 ES에서 중복 여부를 확인합니다.
    
    Args:
        es_client: Elasticsearch 클라이언트
        file_path: 파일 경로
        
    Returns:
        Tuple[bool, str]: (파일 존재 여부, 파일 해시값)
    """
    # 파일 해시값 계산 - 메모리 효율적인 방식으로 업데이트
    file_hash = ""
    try:
        # 큰 파일을 처리할 때 메모리 사용량을 줄이기 위해 청크 단위로 읽음
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()
        
        # 파일 크기도 로깅 (디버깅용)
        file_size = os.path.getsize(file_path)
        print(f"파일 해시 계산 완료: {file_path}, 크기: {format_file_size(file_size)}, 해시: {file_hash[:8]}...")
    except Exception as e:
        print(f"파일 해시 계산 중 오류: {e}")
        return False, ""
    
    # 인덱스 존재 확인
    if not es_client.indices.exists(index=ES_INDEX_NAME):
        return False, file_hash
    
    # ES에서 해당 해시값을 가진 문서 검색
    try:
        query = {
            "query": {
                "term": {
                    "file_hash": file_hash
                }
            },
            "size": 1
        }
        response = es_client.search(index=ES_INDEX_NAME, body=query)
        
        # 검색 결과 확인
        hits = response.get("hits", {}).get("hits", [])
        exists = len(hits) > 0
        
        if exists:
            # 중복 파일 정보 출력
            duplicate_doc = hits[0]["_source"]
            print(f"중복 파일 발견: {os.path.basename(file_path)}, 기존 문서: {duplicate_doc.get('source', 'unknown')}")
        
        return exists, file_hash
    except Exception as e:
        print(f"ES에서 파일 중복 확인 중 오류: {e}")
        return False, file_hash


# 파일 크기를 사람이 읽기 쉬운 형식으로 변환하는 함수 추가
def format_file_size(size_in_bytes):
    """파일 크기를 읽기 쉬운 형식으로 변환합니다."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"


async def process_and_index_file(
    es_client,
    embedding_function,
    uploaded_file_path: str,
    category: str,
    *,                       # 키워드 전용
    reindex: bool = False    # reindex=True → 사본 만들지 않음
) -> bool:
    """
    · reindex=True  → uploads 내부 파일을 그대로 사용
    · reindex=False → 새 업로드: SHA-1 해시 12자리 프리픽스 붙여 저장
    """

    uploads_dir = Path("app/static/uploads")
    src_path    = Path(uploaded_file_path)

    # ──────────────────────────────
    # 0) temp 변수 기본값 (NameError 방지)
    temp_pdf_file_path: str | None = None
    temp_conversion_output_dir: str | None = None
    # ──────────────────────────────

    # ── 1) 저장 경로 결정 ──────────────────────────────────────
    HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{12}_")

    # case A: reindex 모드이거나 이미 our-hash_ 가 붙어 있는 파일
    if reindex or (src_path.parent == uploads_dir and HASH_PREFIX_RE.match(src_path.name)):
        saved_path = src_path
        # 해시는 파일명에서 바로 추출(없으면 나중에 계산)
        match = HASH_PREFIX_RE.match(src_path.name)
        file_hash = match.group(0)[:-1] if match else None
    # case B: 새 업로드 — 해시 계산 후 복사
    else:
        content_bytes = src_path.read_bytes()
        file_hash     = hashlib.sha1(content_bytes).hexdigest()
        saved_name    = f"{file_hash[:12]}_{src_path.name}"
        saved_path    = uploads_dir / saved_name
        if not saved_path.exists():
            saved_path.write_bytes(content_bytes)

    if file_hash is None:               # (reindex + 정상 이름이지만 해시 미포함)
        file_hash = hashlib.sha1(Path(saved_path).read_bytes()).hexdigest()

    # ── 2) 문서 로드 · 청킹 ────────────────────────────────────
    ext = saved_path.suffix.lower()
    documents: List = await load_document(str(saved_path), ext)
    if not documents:
        print(f"[!] 문서 로드 실패: {saved_path.name}")
        return False

    for doc in documents:
        doc.metadata["source"]    = saved_path.name          # 파일명만 저장
        doc.metadata["file_hash"] = file_hash

    # ── 3) ES 인덱싱 ─────────────────────────────────────────
    success = await index_chunks_to_elasticsearch(
        es_client, embedding_function, documents, category
    )

    if not success:
        print(f"[!] 인덱싱 실패: {saved_path.name}")

    # (DOCX 변환·임시 파일 로직은 제거했으므로 clean-up 불필요)
    return success

async def markdown_to_row_chunks(
    md: str,
    page: int,
    file_path: str,
    llm_model=None,
    tokenizer=None,
) -> list[Document]:
    """
    마크다운 표 → 행 단위 Document 리스트.
    헤더명을 각 셀 앞에 붙여 문맥을 살린다.
    """
    # 1) md ➜ HTML ➜ DataFrame
    html = markdown.markdown(md, extensions=["tables"])
    dfs  = pd.read_html(StringIO(html))
    if not dfs:
        return []

    df = dfs[0].fillna("")
    headers = list(df.columns)

    docs = []
    for ridx, row in df.iterrows():
        cells = [f"{h}: {row[h]}" for h in headers]
        raw_row_text = " | ".join(cells)

        # ── ② Qwen로 한 줄 요약(선택) ───────────────────
        one_liner = None
        if llm_model and tokenizer:
            try:
                one_liner = await summarize_row_one_liner(
                    raw_row_text, llm_model, tokenizer
                )
            except Exception:
                pass

        page_content = (one_liner + "\n" if one_liner else "") + raw_row_text

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": file_path,
                    "page": page,
                    "element_type": "table",
                    "row_index": ridx,
                },
            )
        )
    return docs


async def summarize_row_one_liner(text, llm_model, tokenizer):
    """Qwen 사용: 반드시 '이 행은' 으로 시작, 40자 이내."""
    prompt = (
        "다음 텍스트는 표의 한 행입니다. "
        "한국어로 40자 이내 한 문장으로 요약하세요. "
        "반드시 '이 행은'으로 시작하세요.\n\n"
        f"{text}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = llm_model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.2,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    summary = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return summary.strip().replace("\n", " ")[:80]