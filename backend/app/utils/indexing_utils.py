# indexing_utils.py
import time
import subprocess
import uuid
import os, re, asyncio
import hashlib  # 파일 중복 체크를 위한 해시 라이브러리 추가
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import markdown
from io import StringIO
import torch
import threading

from langchain_community.document_loaders import (
    UnstructuredExcelLoader,
    TextLoader,
    PyPDFLoader,  # PyPDFLoader 임포트
    UnstructuredFileLoader,
)
import fitz
from elasticsearch import Elasticsearch
from langchain.schema import Document
from elasticsearch.helpers import bulk
import traceback
from datetime import datetime
import logging
from .table_parser import (detect_table_in_text,
                            parse_ocr_table,
                              is_same_table,
                              )
from .ocr_utils import (
        extract_text_from_file,
        extract_text_from_image,
        extract_text_from_pdf_with_ocr,
        extract_images_from_pdf_with_layout,
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
ROW_INDEX = "rag_rows_v1"
DOC_INDEX = ES_INDEX_NAME
ALIAS_READ  = "rag-idx"
ALIAS_WRITE = "rag-idx-write"
BATCH_SIZE  = 4
IMAGE_DIR = Path("app/static/document_images")  # 사용 안되면 제거 가능
os.makedirs(IMAGE_DIR, exist_ok=True)  # 사용 안되면 제거 가능

# LOADER_MAPPING: 이미지 파일 확장자 추가
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    
    # 변환 후 PDF로 처리할 파일들
    ".xlsx": ("CONVERT_TO_PDF", {}),
    ".xls": ("CONVERT_TO_PDF", {}),
    ".ppt": ("CONVERT_TO_PDF", {}),
    ".pptx": ("CONVERT_TO_PDF", {}),
    ".docx": ("CONVERT_TO_PDF", {}),
    
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    
    # 이미지 파일
    ".jpg": ("OCR_LOADER", {}),
    ".jpeg": ("OCR_LOADER", {}),
    ".png": ("OCR_LOADER", {}),
    ".bmp": ("OCR_LOADER", {}),
    ".tiff": ("OCR_LOADER", {}),
    ".tif": ("OCR_LOADER", {}),
    ".webp": ("OCR_LOADER", {}),
}
#토크나이저, 모델

shared_llm_model=None 
shared_tokenizer = None
llm_model = None
tokenizer = None


def _route_index(et: str) -> str:
    return ROW_INDEX if et == "table_row" else DOC_INDEX

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
async def load_document(file_path_to_load: str, loader_selector_ext: str, llm_model=None, tokenizer=None) -> List[Document]:
    print(
        f"load_document 호출: file_path_to_load='{file_path_to_load}', loader_selector_ext='{loader_selector_ext}'"
    )

    loader_info = LOADER_MAPPING.get(loader_selector_ext)

    # OCR 로더 처리
    if loader_info and loader_info[0] == "OCR_LOADER":
        logger.info(f"OCR 로더를 사용하여 파일 처리: {file_path_to_load}")
        return await load_document_with_ocr(file_path_to_load)
    if loader_info and loader_info[0] == "CONVERT_TO_PDF":
        logger.info(f"PDF 변환 후 처리: {file_path_to_load}")
        temp_dir = "temp_conversions"
        
        # Office 파일을 PDF로 변환
        pdf_path = await convert_office_to_pdf(file_path_to_load, temp_dir)
        
        if pdf_path:
            logger.info(f"PDF 변환 성공, Docling으로 처리: {pdf_path}")
            # 변환된 PDF를 Docling 파이프라인으로 처리
            return await load_document_with_ocr(pdf_path)
        else:
            logger.warning(f"PDF 변환 실패, 기존 방식으로 fallback: {file_path_to_load}")

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

async def extract_image_ocr_texts(pdf_path: str, layout_info, doc_id: str = None):
    image_texts = {}
    image_paths = {}
    
    if not doc_id:
        doc_id = Path(pdf_path).stem
        
    img_dir = IMAGE_DIR / doc_id
    img_dir.mkdir(parents=True, exist_ok=True)
        
    # ①  layout_info 는 리스트이므로 그대로 순회
    for idx, node in enumerate(layout_info):
        if node.get("type") != "figure":
            continue                                   # 그림(figure) 만 골라서 진행

        page_num = int(node.get("page", 1))
        x0, y0, x1, y1 = node["bbox"]
        img_data = crop_pdf_region(pdf_path, page_num, fitz.Rect(x0, y0, x1, y1))

        img_name = f"page_{page_num}_img_{idx+1}.png"
        img_path = img_dir / img_name
        with open(img_path, "wb") as f:
            f.write(img_data)

        relative = f"document_images/{doc_id}/{img_name}"
        image_paths.setdefault(page_num, []).append(relative)
    
    return image_texts, image_paths


def get_docling_converter():
    """Docling 컨버터 싱글톤 (메모리 효율)"""
    global _docling_layout_converter, _last_docling_init_time
    if not DOCLING_AVAILABLE:
        return None
    
    if _docling_layout_converter is None:
        try:
            # 레이아웃 분석만 (OCR 없이)
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = True  # 속도를 위해 OCR 비활성화
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

def split_table_rows(table_text: str, meta_base: dict) -> list[Document]:
    """
    ① 한 표 → 행 단위 Document 리스트
    ② 각 행에 table_id / row_index / n_rows 저장
    """
    rows = [r for r in table_text.splitlines() if r.strip()]
    tid = uuid.uuid4().hex[:8]

    docs = []
    for idx, row in enumerate(rows):
        doc = Document(
            page_content=row,
            metadata={
                **meta_base,
                "element_type": "table_row",
                "table_id": tid,
                "row_index": idx,
                "n_rows": len(rows),
            },
        )
        docs.append(doc)
    print(f"DEBUG MERGE ▸ table_row created = {idx}")
    return docs


def merge_table_chunks(docs: list[Document]) -> list[Document]:
    """
    1) 페이지를 넘어가면서 쪼개진 표를 다시 '완전한 표 텍스트' 로 복원
    2) split_table_rows() 에 넘겨 row chunk 로 변환
    """
    merged_tables, buf = [], []
    cur_tid = None

    for d in docs:
        if d.metadata.get("element_type") == "table":
            # 같은 표인지 확인 (헤더/ID 기준)
            same = cur_tid and is_same_table(buf[-1], d)
            if not same:
                if buf:
                    merged_tables.extend(split_table_rows("\n".join(x.page_content for x in buf),
                                                         buf[0].metadata))
                    buf.clear()
                cur_tid = uuid.uuid4().hex[:8]
            buf.append(d)
        else:
            if buf:
                merged_tables.extend(split_table_rows("\n".join(x.page_content for x in buf),
                                                     buf[0].metadata))
                buf.clear()
                cur_tid = None
            merged_tables.append(d)

    if buf:
        merged_tables.extend(split_table_rows("\n".join(x.page_content for x in buf),
                                             buf[0].metadata))
    merged_docs = merged_tables  # ← 기존 반환 리스트
    row_cnt = sum(d.metadata.get("element_type") == "table_row"
                  for d in merged_docs)
    print(f"DEBUG MERGE END ▸ table_row count = {row_cnt}")
    return merged_tables


async def load_document_with_ocr(file_path: str) -> List["Document"]:
    try:
        logger.info(f"하이브리드 방식으로 파일 처리 중: {file_path}")
        file_extension = Path(file_path).suffix.lower()

        # ── PDF 외 포맷은 기존 함수로 위임 ───────────────────────────
        if file_extension != ".pdf":
            return await load_document_with_ocr_original(file_path)

        start = time.time()
        # 1) Docling 레이아웃
        layout_info = None
        if DOCLING_AVAILABLE:
            try:
                converter = get_docling_converter()
                if converter is not None:
                    layout_result = await asyncio.to_thread(converter.convert, file_path)
                    layout_info = layout_result.document
                    logger.info("✅ 레이아웃 분석 완료")
            except Exception as e:
                logger.warning(f"Docling 레이아웃 분석 실패 → 무시: {e}")
        # 2) 텍스트 레이어 추출
        pdf_texts = extract_pdf_text_fast(file_path)
        # 3) 텍스트 없으면 기존 OCR Fallback
        if not any(t.strip() for t in pdf_texts.values()):
            logger.info("PDF 텍스트 없어서 기존 OCR 사용")
            return await load_document_with_ocr_original(file_path)
        # # 4) Hybrid 문서 생성
        docs = await create_hybrid_documents(pdf_texts, layout_info, file_path, llm_model=shared_llm_model, tokenizer=shared_tokenizer)  # layout_info → None
        logger.info(f"하이브리드 완료: {len(docs)} 청크, {time.time()-start:.2f}s")
        return docs
    
    except Exception as e:
        logger.error(f"하이브리드 실패 → 기존 방식 Fallback: {e}")
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
    llm_model: str | None,
    tokenizer=None,
) -> List["Document"]:
    """
    Docling과 OCR 결과를 조합하는 하이브리드 문서 생성
    
    기존 문제: Docling 결과가 있는 페이지를 무조건 제외 → OCR 표 감지 기회 박탈
    개선사항: 두 결과를 조합하여 최대한 많은 표 정보 추출
    """
    documents: List["Document"] = []
    file_name = Path(file_path).name
    
    doc_id = strip_uuid_prefix(file_name)
    if '.' in doc_id:
        doc_id = doc_id.rsplit('.', 1)[0]

    # === 1단계: Docling 표 추출 (기존 유지) ===
    docling_tables = []
    docling_table_pages = set()
    
    if layout_info is not None:
        tables = getattr(layout_info, "tables", [])
        for idx, tbl in enumerate(tables):
            md = getattr(tbl, "export_to_markdown", lambda: "")()
            if not md:
                continue
            
            page_num = getattr(tbl, "page", 1)
            row_docs = await markdown_to_row_chunks(
                md, page=page_num, file_path=file_path,
                llm_model=llm_model, tokenizer=tokenizer,
            )
            
            if row_docs:  # 실제로 파싱된 경우만 기록
                docling_tables.extend(row_docs)
                docling_table_pages.add(page_num)

    # === 2단계: 이미지 OCR 처리 (기존 유지) ===
    print("DEBUG ▸ image OCR 호출:", file_path)
    image_ocr_texts, image_paths = await extract_images_from_pdf_with_layout(file_path, doc_id)
    print("DEBUG ▸ image_paths keys =", list(image_paths.keys())[:5])

    # === 3단계: 각 페이지별 표 추출 전략 결정 ===
    ocr_tables = []
    
    for page_num, page_text in pdf_texts.items():
        if not page_text.strip():
            continue
        
        # 텍스트에서 표 감지 시도 (모든 페이지에서 시도)
        if detect_table_in_text(page_text):
            page_table_docs = parse_ocr_table(page_text)
            
            # Docling 결과와 비교하여 더 나은 결과 선택
            if page_num in docling_table_pages:
                # 이미 Docling에서 표를 찾은 페이지
                docling_docs_for_page = [doc for doc in docling_tables 
                                       if doc.metadata.get("page") == page_num]
                
                # OCR이 더 많은 행을 찾았거나, Docling이 실패한 경우 OCR 결과 우선
                if (len(page_table_docs) > len(docling_docs_for_page) * 1.5 or 
                    len(docling_docs_for_page) == 0):
                    print(f"페이지 {page_num}: OCR 표 결과가 더 우수함 ({len(page_table_docs)} vs {len(docling_docs_for_page)})")
                    
                    # 기존 Docling 결과 제거하고 OCR 결과 사용
                    docling_tables = [doc for doc in docling_tables 
                                    if doc.metadata.get("page") != page_num]
                    ocr_tables.extend(page_table_docs)
                else:
                    print(f"페이지 {page_num}: Docling 표 결과 유지")
            else:
                # Docling에서 표를 찾지 못한 페이지 → OCR 결과 사용
                print(f"페이지 {page_num}: Docling 놓친 표를 OCR로 발견")
                ocr_tables.extend(page_table_docs)
    
    # === 4단계: 표 Document들 병합 및 추가 ===
    all_table_docs = docling_tables + ocr_tables
    
    # 표가 있는 페이지 목록 업데이트
    actual_table_pages = set()
    for doc in all_table_docs:
        page_num = doc.metadata.get("page")
        if page_num:
            actual_table_pages.add(page_num)
    
    # 표 Document들을 페이지별로 정렬 후 병합 처리
    all_table_docs.sort(key=lambda x: (x.metadata.get("page", 0), x.metadata.get("row_index", 0)))
    merged_table_docs = merge_table_chunks(all_table_docs)
    documents.extend(merged_table_docs)
    
    # 표 페이지의 인접 페이지도 제외 (기존 로직 유지하되 실제 표 페이지 기준)
    extended_table_pages = actual_table_pages | {p + 1 for p in actual_table_pages}

    # === 5단계: 본문 + 이미지 처리 (기존 로직 유지) ===
    for page_num, page_text in pdf_texts.items():
        if page_num in extended_table_pages or not page_text.strip():
            continue
            
        combined = page_text.strip()
        page_images = []
        
        # 이미지 OCR 성공분 추가
        if page_num in image_ocr_texts:
            combined += "\n\n" + "\n".join(image_ocr_texts[page_num])
        
        # 이미지 경로 수집
        if page_num in image_paths:
            page_images = image_paths[page_num]

        # 단락 분할
        for para in re.split(r"\n\s*\n", combined):
            p = para.strip()
            if len(p) < 10:
                continue
            
            metadata = {
                "source": file_name,
                "page": page_num,
                "loaded_at": datetime.now().isoformat(),
            }
            
            if page_images:
                metadata["images"] = page_images
                metadata["has_images"] = True
                
            documents.append(
                Document(page_content=p, metadata=metadata)
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
            chk for chk in batch_chunks_input if chk.page_content and chk.page_content.strip()
        ]
        if not valid_chunks_in_batch:
            return

        chunk_texts_for_embedding = [chk.page_content for chk in valid_chunks_in_batch]
        try:
            embeddings = await asyncio.to_thread(embedding_function, chunk_texts_for_embedding)
            actions_for_bulk = []

            # ── 핵심 패치: **element_type 기준으로 인덱스 라우팅** ──
            for i, chunk_doc in enumerate(valid_chunks_in_batch):
                idx_name = _route_index(chunk_doc.metadata.get("element_type", "text"))
                page_number_to_index = chunk_doc.metadata.get("page") or 1

                source_filename = chunk_doc.metadata.get("source", "unknown_source")
                chunk_id_val    = chunk_doc.metadata.get("chunk_id", f"batch{batch_num_for_log}_{i}")
                es_doc_id       = f"{source_filename.replace('.', '_')}_{page_number_to_index}_{chunk_id_val}"

                es_source_doc = {
                    "text":        chunk_doc.page_content,
                    "embedding":   embeddings[i],
                    "source":      source_filename,
                    "page":        int(page_number_to_index),
                    "category":    category,
                    "chunk_id":    chunk_id_val,
                    "total_chunks": chunk_doc.metadata.get("total_chunks", len(chunks)),
                    "indexed_at":  datetime.now().isoformat(),
                    "element_type": chunk_doc.metadata.get("element_type", "text"),
                    # 표 파싱 실패 시 원문을 row_text 에 그대로 둠
                    "row_text":   chunk_doc.metadata.get("row_text"),
                    # 이미지 메타
                    "images":     chunk_doc.metadata.get("images", []),
                    "has_images": chunk_doc.metadata.get("has_images", False),
                }

                actions_for_bulk.append(
                    {
                        "_index": idx_name,          # ← ★★ ES_INDEX_NAME 대신 idx_name
                        "_id":    es_doc_id,
                        "_source": es_source_doc,
                    }
                )
            # ────────────────────────────────────────────────────

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
                    print(f"배치 {batch_num_for_log} 처리 중 {len(failed_items)}개 문서 인덱싱 실패.")

        except Exception as e_batch:
            print(f"배치 {batch_num_for_log} 처리 중 예외 발생: {e_batch}")
            failure_count += len(valid_chunks_in_batch)
            traceback.print_exc()

    tasks = [
        process_batch(chunks[i : i + batch_size], (i // batch_size) + 1)
        for i in range(0, len(chunks), batch_size)
    ]
    await asyncio.gather(*tasks)
    print(f"인덱싱 완료: 총 {len(chunks)} 청크 중 {success_count}개 성공, {failure_count}개 실패")
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
    es_client: Elasticsearch,
    embedding_function,
    uploaded_file_path: str,
    category: str,
    shared_llm_model=None,      # ← 추가
    shared_tokenizer=None,      # ← 추가
    *,                          # 키워드 전용
    reindex: bool = False       # reindex=True → 사본 만들지 않음
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
    documents: List = await load_document(str(saved_path), ext, shared_llm_model, shared_tokenizer)
    if not documents:
        print(f"[!] 문서 로드 실패: {saved_path.name}")
        return False

    for doc in documents:
        clean_name = strip_uuid_prefix(saved_path.name)
        doc.metadata["source"]    = clean_name        # 파일명만 저장
        doc.metadata["file_hash"] = file_hash

    # ── 3) ES 인덱싱 ─────────────────────────────────────────
    success = await index_chunks_to_elasticsearch(
        es_client, embedding_function, documents, category
    )

    if not success:
        print(f"[!] 인덱싱 실패: {saved_path.name}")
    return success

import uuid, re, markdown, pandas as pd
from io import StringIO
from langchain.schema import Document   # 쓰고 계신 Document 모델에 맞게 import

async def markdown_to_row_chunks(
    md: str,
    page: int,
    file_path: str,
    llm_model=None,
    tokenizer=None,
) -> list[Document]:
    """
    마크다운 표 ➜ 행 단위 Document 리스트.
    - table_id / row_index / n_rows  메타 추가
    - <caption> 또는 “표 3.” 형태 캡션을 찾아서 metadata["caption"] 에 저장
    """
    # ── 0) 캡션 추출 ───────────────────────────────────
    md = clean_html_tags(md)
    caption = None
    # ① HTML <caption> 태그
    m = re.search(r'<caption[^>]*>(.*?)</caption>', md, re.S | re.I)
    if m:
        caption = re.sub(r'<[^>]+>', '', m.group(1)).strip()
    # ② "표 1." · "Table 2." 같은 문장
    if not caption:
        m = re.match(r'^\s*(표|Table)\s*\d+[^|]*$', md.strip().splitlines()[0])
        if m:
            caption = m.group(0).strip()

    # ── 1) md → HTML → DataFrame ─────────────────────
    html = markdown.markdown(md, extensions=["tables"])
    dfs  = pd.read_html(StringIO(html))
    if not dfs:
        return []

    df = dfs[0].fillna("")
    headers = list(df.columns)
    n_rows  = len(df)
    tid     = uuid.uuid4().hex[:8]              # ★ table_id 한 번만 생성

    docs = []
    for ridx, row in df.iterrows():
        # ── 셀 문맥 살리기 ────────────────────────────
        cells = [f"{h}: {row[h]}" for h in headers]
        raw_row_text = " | ".join(cells)

        # ── ② 한 줄 요약 (선택) ─────────────────────
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
                    "element_type": "table_row",
                    "table_id": tid,       # ★ 같은 표라면 행마다 동일
                    "row_index": ridx,
                    "n_rows": n_rows,
                    **({"caption": caption} if caption else {}),
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
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    summary = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return summary.strip().replace("\n", " ")[:80]

UUID_PREFIX = re.compile(
    r'(?:(?:[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
    r'|[0-9a-fA-F]{32}'
    r'|[0-9a-fA-F]{12})[_-]?)+',
    re.I
)

def strip_uuid_prefix(filename: str) -> str:
    """파일명에서 UUID/해시 접두어('_' 또는 '-')를 모두 제거"""
    return UUID_PREFIX.sub('', os.path.basename(filename))

def clean_html_tags(text: str) -> str:
    """HTML 태그 제거 및 테이블 정리"""
    if not text:
        return text
    
    # 1. HTML 태그 완전 제거
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 연속된 공백/줄바꿈 정리
    text = re.sub(r'\s+', ' ', text)
    
    # 3. 앞뒤 공백 제거
    return text.strip()

def convert_office_to_pdf_sync(office_path: str, output_dir: str) -> Optional[str]:
    """LibreOffice로 Office 파일을 PDF로 변환 (Excel, PowerPoint, Word 지원)"""
    try:
        file_ext = Path(office_path).suffix.lower()
        file_stem = Path(office_path).stem
        
        print(f"Office 파일을 PDF로 변환 시도 (libreoffice): '{office_path}' -> '{output_dir}' 디렉토리로")
        print(f"파일 형식: {file_ext}")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        command = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            office_path,
        ]
        
        env = os.environ.copy()
        env["HOME"] = "/tmp"
        
        # 파일 형식에 따라 타임아웃 조정
        timeout = 180 if file_ext in ['.pptx', '.ppt'] else 120  # PPT는 시간이 더 걸릴 수 있음
        
        process = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=False, 
            timeout=timeout, 
            env=env
        )
        
        expected_pdf_filename = file_stem + ".pdf"
        converted_pdf_path = os.path.join(output_dir, expected_pdf_filename)
        
        if process.returncode == 0 and os.path.exists(converted_pdf_path):
            print(f"PDF 변환 성공 (libreoffice): '{converted_pdf_path}'")
            return converted_pdf_path
        else:
            print(f"PDF 변환 실패 (libreoffice). Return code: {process.returncode}")
            print(f"Stdout: {process.stdout.strip()}")
            print(f"Stderr: {process.stderr.strip()}")
            
            # 실패한 PDF 파일이 있다면 삭제
            if os.path.exists(converted_pdf_path):
                try:
                    os.remove(converted_pdf_path)
                    print(f"실패한 PDF 파일 삭제: {converted_pdf_path}")
                except Exception as e_rem:
                    print(f"실패한 PDF 파일 삭제 중 오류: {e_rem}")
            return None
            
    except FileNotFoundError:
        print("PDF 변환 실패: 'libreoffice' 명령어를 찾을 수 없습니다. 서버에 libreoffice가 설치되어 있는지 확인하세요.")
        return None
    except subprocess.TimeoutExpired:
        print(f"PDF 변환 시간 초과 (libreoffice): {office_path} (timeout: {timeout}초)")
        return None
    except Exception as e:
        print(f"Office -> PDF 변환 중 예외 발생 (libreoffice, {office_path}): {e}")
        traceback.print_exc()
        return None


async def convert_office_to_pdf(office_path: str, output_dir: str) -> Optional[str]:
    """Office 파일을 PDF로 비동기 변환"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, convert_office_to_pdf_sync, office_path, output_dir
    )


