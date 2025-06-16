# indexing_utils.py

import subprocess
import os, re, asyncio
import hashlib  # 파일 중복 체크를 위한 해시 라이브러리 추가
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredExcelLoader,
    TextLoader,
    PyPDFLoader,  # PyPDFLoader 임포트
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# 로깅 설정
logger = logging.getLogger(__name__)

ES_INDEX_NAME = "rag_documents_kure_v1"
IMAGE_DIR = "static/document_images"  # 사용 안되면 제거 가능
os.makedirs(IMAGE_DIR, exist_ok=True)  # 사용 안되면 제거 가능

# LOADER_MAPPING: 이미지 파일 확장자 추가
LOADER_MAPPING = {
    ".pdf": (
        PyPDFLoader,  # 페이지 처리가 더 안정적인 PyPDFLoader 사용
        {},  # PyPDFLoader는 특별한 초기화 인자 없이 파일 경로만 받음
    ),
    ".xlsx": (UnstructuredExcelLoader, {"mode": "paged"}),
    ".xls": (UnstructuredExcelLoader, {"mode": "paged"}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    # 이미지 파일 OCR 처리를 위한 확장자 추가 (OCR_LOADER로 표시)
    ".jpg": ("OCR_LOADER", {}),
    ".jpeg": ("OCR_LOADER", {}),
    ".png": ("OCR_LOADER", {}),
    ".bmp": ("OCR_LOADER", {}),
    ".tiff": ("OCR_LOADER", {}),
    ".tif": ("OCR_LOADER", {}),
    ".webp": ("OCR_LOADER", {}),
}


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


# PaddleOCR을 이용한 문서 로드 함수
async def load_document_with_ocr(file_path: str) -> List[Document]:
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
        
        logger.info(f"PaddleOCR 텍스트 추출 완료: {len(documents)} 페이지/섹션 생성")
        return documents
        
    except Exception as e:
        logger.error(f"PaddleOCR 문서 로드 중 오류 발생: {file_path}, 오류: {e}")
        traceback.print_exc()
        return []


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
            clean_text = re.sub(r'\s+', '', total_text)
            
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


# --- split_text 함수 (제공해주신 원본 코드의 adaptive 로직 유지) ---
def split_text(
    documents: List[Document],
    chunk_size: int = 768,  # 원본 기본값
    chunk_overlap: int = 128,  # 원본 기본값
    adaptive: bool = True,  # 원본 기본값
) -> List[Document]:
    if not documents:
        return []

    # 원본 코드의 adaptive 청킹 로직
    if adaptive:
        total_length = sum(len(doc.page_content) for doc in documents)
        avg_doc_length = total_length / len(documents) if documents else 0
        if avg_doc_length < 500:
            chunk_size = max(256, chunk_size // 2)
            chunk_overlap = max(64, chunk_overlap // 2)
        elif avg_doc_length > 2000:
            chunk_size = min(1024, chunk_size * 2)
            chunk_overlap = min(256, chunk_overlap * 2)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # 원본 separator (. 뒤 공백 주의)
        # keep_separator=True, # 원본은 False (기본값)
        # is_separator_regex=False, # 원본은 False (기본값)
    )
    try:
        # text_splitter.split_documents는 각 청크에 원본 Document의 메타데이터를 복사해줌
        all_chunks = text_splitter.split_documents(documents)

        # 원본 코드의 doc_title 및 enhanced_content 로직 (페이지 번호 문제와는 별개)
        # 이 부분은 필요에 따라 유지하거나, 우선 페이지 문제 해결 후 검토
        final_enhanced_chunks = []
        doc_title_from_first_chunk = ""
        for i, chunk_doc in enumerate(all_chunks):
            # 원본 코드의 제목 추출 로직
            if i == 0 and "\n" in chunk_doc.page_content:
                first_line = chunk_doc.page_content.split("\n")[0].strip()
                if len(first_line) < 100 and not first_line.endswith("."):
                    doc_title_from_first_chunk = first_line

            enhanced_content_for_chunk = chunk_doc.page_content
            if (
                doc_title_from_first_chunk
                and doc_title_from_first_chunk not in enhanced_content_for_chunk
            ):
                enhanced_content_for_chunk = (
                    f"{doc_title_from_first_chunk}\n\n{enhanced_content_for_chunk}"
                )

            # 메타데이터는 이미 chunk_doc에 복사되어 있음 (page 포함)
            chunk_doc.metadata["chunk_id"] = chunk_doc.metadata.get(
                "chunk_id", i
            )  # 기존 chunk_id 유지 시도, 없으면 순번
            chunk_doc.metadata["total_chunks"] = len(all_chunks)  # 원본 코드에 있었음

            final_enhanced_chunks.append(
                Document(
                    page_content=enhanced_content_for_chunk, metadata=chunk_doc.metadata
                )
            )

        print(f"DEBUG (split_text): 총 {len(final_enhanced_chunks)}개의 청크 생성됨.")
        # for i, chk in enumerate(final_enhanced_chunks[:3]):
        #     print(f"  Split chunk {i} metadata: {chk.metadata}") # 페이지 정보 확인
        return final_enhanced_chunks
    except Exception as e:
        print(f"텍스트 분할 중 오류 발생: {e}")
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


# --- process_and_index_file 함수 수정 (async load_document 호출) ---
async def process_and_index_file(
    es_client: Any,
    embedding_function: Any,
    uploaded_file_path: str,
    category: str,
) -> bool:
    print(f"파일 처리 시작: '{uploaded_file_path}', 카테고리: '{category}'")
    if not es_client or not embedding_function:
        print("ES 클라이언트 또는 임베딩 함수 유효하지 않음.")
        return False

    # 파일 중복 체크
    file_exists, file_hash = await check_file_exists(es_client, uploaded_file_path)
    if file_exists:
        print(f"이미 인덱싱된 파일입니다: {uploaded_file_path}")
        return True  # 중복 파일은 성공으로 간주
        
    original_filename_for_metadata = os.path.basename(uploaded_file_path)
    file_to_actually_load = uploaded_file_path
    extension_for_loader_selection = Path(uploaded_file_path).suffix.lower()
    temp_pdf_file_path = None
    temp_conversion_output_dir = None

    if extension_for_loader_selection == ".docx":
        temp_conversion_output_dir = os.path.join(
            os.path.dirname(uploaded_file_path), "temp_pdf_conversion"
        )
        os.makedirs(temp_conversion_output_dir, exist_ok=True)
        temp_pdf_file_path = await convert_docx_to_pdf(
            uploaded_file_path, temp_conversion_output_dir
        )
        if temp_pdf_file_path and os.path.exists(temp_pdf_file_path):
            file_to_actually_load = temp_pdf_file_path
            extension_for_loader_selection = ".pdf"
        else:
            print(
                f"DOCX -> PDF 변환 실패. 원본 DOCX({extension_for_loader_selection})를 직접 처리 시도."
            )
            # extension_for_loader_selection은 .docx 그대로 유지 -> load_document에서 UnstructuredFileLoader 사용

    # 함수 시그니처 변경으로 인한 수정
    documents = await load_document(file_to_actually_load, extension_for_loader_selection)
    
    if not documents:
        # ... (실패 처리 및 임시 파일 정리)
        print(f"문서 로드 실패: '{file_to_actually_load}'")
        if temp_pdf_file_path and os.path.exists(temp_pdf_file_path):
            os.remove(temp_pdf_file_path)
        if (
            temp_conversion_output_dir
            and os.path.exists(temp_conversion_output_dir)
            and not os.listdir(temp_conversion_output_dir)
        ):
            os.rmdir(temp_conversion_output_dir)
        return False
    print(
        f"문서 로드 완료: {len(documents)} 페이지/섹션 (처리 파일: '{file_to_actually_load}')"
    )

    for doc in documents:
        doc.metadata["source"] = original_filename_for_metadata
        # 파일 해시값 추가
        doc.metadata["file_hash"] = file_hash

    # 청킹: 원본 process_and_index_file의 청크 크기 결정 로직을 split_text 내부의 adaptive로 옮겼거나,
    # split_text 호출 시 chunk_size, chunk_overlap을 명시적으로 전달해야 함.
    # 여기서는 split_text의 기본값 및 adaptive 로직을 사용.
    # 만약 process_and_index_file의 원래 인자(chunk_size, chunk_overlap)를 사용하고 싶다면,
    # split_text(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap, adaptive=False) 와 같이 호출.
    # 현재는 원본 indexing_utils.py의 split_text 기본값/adaptive 로직을 따르도록 함.
    chunks = split_text(documents)  # adaptive=True가 기본으로 적용됨

    if not chunks:
        # ... (실패 처리 및 임시 파일 정리)
        print(f"텍스트 분할 실패: '{file_to_actually_load}'")
        if temp_pdf_file_path and os.path.exists(temp_pdf_file_path):
            os.remove(temp_pdf_file_path)
        if (
            temp_conversion_output_dir
            and os.path.exists(temp_conversion_output_dir)
            and not os.listdir(temp_conversion_output_dir)
        ):
            os.rmdir(temp_conversion_output_dir)
        return False
    print(f"텍스트 분할 완료: {len(chunks)} 청크 생성")

    success = await index_chunks_to_elasticsearch(
        es_client, embedding_function, chunks, category
    )

    # ... (성공/실패 로깅 및 임시 파일 정리 로직)
    if not success:
        print(f"파일 인덱싱 실패: {original_filename_for_metadata}")
    if temp_pdf_file_path and os.path.exists(temp_pdf_file_path):
        try:
            os.remove(temp_pdf_file_path)
            print(f"임시 PDF 파일 삭제 완료: '{temp_pdf_file_path}'")
        except Exception as e_rem:
            print(f"임시 PDF 파일 삭제 중 오류: {e_rem}")
    if (
        temp_conversion_output_dir
        and os.path.exists(temp_conversion_output_dir)
        and not os.listdir(temp_conversion_output_dir)
    ):
        try:
            os.rmdir(temp_conversion_output_dir)
            print(f"임시 변환 디렉토리 삭제 완료: '{temp_conversion_output_dir}'")
        except Exception as e_rem_dir:
            print(f"임시 변환 디렉토리 삭제 중 오류: {e_rem_dir}")
    return success
