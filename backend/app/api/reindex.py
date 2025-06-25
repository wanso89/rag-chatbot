"""
재인덱싱 API 모듈

모든 파일을 실제 OCR 품질로 재인덱싱하는 기능을 제공합니다.
"""
import sys
import os
import time
import logging, asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ES 및 인덱싱 관련 import
from core.embeddings import get_embedding_function
from core.elasticsearch import get_elasticsearch_client
from utils.indexing_utils import (    
    process_and_index_file,
    ES_INDEX_NAME, 
)

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOADS_DIR = Path("app/static/uploads")
BATCH_SIZE  = 10   

@router.get("/test")
async def test_api():
    """재인덱싱 API 연결 테스트"""
    return {"status": "success", "message": "재인덱싱 API 연결됨!"}


@router.post("/reindex-all-files")
async def reindex_all_files():
    """
    uploads/ 안에 존재하는 모든 파일을 다시 읽어
    Elasticsearch 인덱스를 새로 만든다.
    """
    t0 = time.time()
    logger.info("=== 재인덱싱 요청 수신 ===")

    # 1. 사전 체크
    if not UPLOADS_DIR.exists():
        raise HTTPException(500, detail=f"업로드 경로가 없습니다: {UPLOADS_DIR}")

    es = get_elasticsearch_client()
    if not es:
        raise HTTPException(500, detail="Elasticsearch 연결 실패")
    
    embed_fn = get_embedding_function()

    # 2. uploads 디렉터리 스캔
    file_list: List[Path] = [p for p in UPLOADS_DIR.iterdir() if p.is_file()]
    if not file_list:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "업로드 폴더에 파일이 없습니다.",
                "found_files": 0,
            },
        )
    logger.info(f"uploads 폴더에서 {len(file_list)}개 파일 발견")

    # 3. 기존 인덱스 전부 삭제
    try:
        es.delete_by_query(index=ES_INDEX_NAME, body={"query": {"match_all": {}}})
        logger.info("기존 인덱스 전체 삭제 완료")
    except Exception as e:
        logger.warning(f"인덱스 삭제 경고: {e}")

    # 4. 파일별 인덱싱 (batch 병렬)
    success, failed = 0, 0

    async def worker(batch: List[Path], batch_no: int):
        nonlocal success, failed
        for fp in batch:
            logger.info(f"[batch {batch_no}] 인덱싱: {fp.name}")
            try:
                ok = await process_and_index_file(
                    es_client=es,
                    embedding_function=embed_fn,
                    uploaded_file_path=str(fp),
                    category="메뉴얼",
                    reindex=True,          # ← 복사 금지
                )
                success += 1 if ok else 0
                failed  += 0 if ok else 1
            except Exception as e:
                failed += 1
                logger.error(f"{fp.name} 처리 오류: {e}")

    # batch split
    tasks = []
    for i in range(0, len(file_list), BATCH_SIZE):
        tasks.append(worker(file_list[i : i + BATCH_SIZE], (i // BATCH_SIZE) + 1))
    await asyncio.gather(*tasks)

    elapsed = round(time.time() - t0, 2)
    return {
        "status": "success" if failed == 0 else "partial_success",
        "message": f"총 {len(file_list)}개 중 {success}개 성공, {failed}개 실패",
        "total_files": len(file_list),
        "success": success,
        "failed": failed,
        "seconds": elapsed,
        "index": ES_INDEX_NAME,
    }

@router.get("/status")
async def get_reindex_status():
    """문서 수·uploads 파일 수 간단 조회"""
    es = get_elasticsearch_client()
    if not es:
        return {"status": "error", "message": "ES 연결 실패"}

    doc_cnt = es.count(index=ES_INDEX_NAME).get("count", 0)
    up_cnt  = len([p for p in UPLOADS_DIR.iterdir() if p.is_file()])
    return {
        "status": "success",
        "elasticsearch_documents": doc_cnt,
        "upload_files": up_cnt,
        "index_name": ES_INDEX_NAME,
    }

async def get_unique_files_from_uploads(uploads_dir: str) -> List[str]:
    """uploads 폴더에서 직접 파일명들을 가져옵니다."""
    files = []
    if os.path.exists(uploads_dir):
        # 모든 파일 가져오기 (확장자 제한 없음)
        files = [f for f in os.listdir(uploads_dir) 
                if os.path.isfile(os.path.join(uploads_dir, f))]
    logger.info(f"uploads 폴더에서 {len(files)}개 파일 발견")
    return files

async def get_unique_files_from_es(es_client) -> List[str]:
    """ES에서 고유 파일명들을 추출합니다."""
    try:
        # aggregation 쿼리로 고유 파일명들 추출
        query = {
            "size": 0,
            "aggs": {
                "unique_files": {
                    "terms": {
                        "field": "source",
                        "size": 1000  # 최대 1000개 파일
                    }
                }
            }
        }
        
        response = es_client.search(index=ES_INDEX_NAME, body=query)
        
        unique_files = []
        if "aggregations" in response:
            buckets = response["aggregations"]["unique_files"]["buckets"]
            unique_files = [bucket["key"] for bucket in buckets]
        
        return unique_files
    except Exception as e:
        logger.error(f"ES 검색 중 오류: {e}")
        return []

def check_existing_files(unique_files: List[str], uploads_dir: str) -> List[str]:
    """uploads 폴더에서 실제 존재하는 파일들을 확인합니다."""
    logger.info(f"서버 작업 디렉토리: {os.getcwd()}")
    logger.info(f"uploads_dir: {uploads_dir}")
    existing_files = []
    
    for filename in unique_files:
        file_path = os.path.join(uploads_dir, filename)
        if os.path.exists(file_path):
            existing_files.append(file_path)
            logger.info(f"파일 확인됨: {filename}")
        else:
            logger.warning(f"파일 없음: {filename}")
    
    return existing_files

async def delete_all_documents(es_client) -> Dict[str, Any]:
    """ES에서 모든 문서를 삭제합니다."""
    try:
        delete_query = {"query": {"match_all": {}}}
        response = es_client.delete_by_query(index=ES_INDEX_NAME, body=delete_query)
        
        return {
            "deleted": response.get("deleted", 0),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"문서 삭제 중 오류: {e}")
        return {"deleted": 0, "status": "error"}

async def reindex_files(file_paths: List[str], es_client, embedding_function) -> Dict[str, int]:
    """파일들을 순차적으로 재인덱싱합니다."""
    success_count = 0
    failed_count = 0
    
    for i, file_path in enumerate(file_paths):
        logger.info(f"재인덱싱 중... ({i+1}/{len(file_paths)}): {os.path.basename(file_path)}")
        
        # 카테고리 추정 (파일명에서)
        category = "메뉴얼"  # 기본값
        
        try:
            # 파일 재인덱싱
            result = await process_and_index_file(
                es_client=es_client,
                embedding_function=embedding_function,
                uploaded_file_path=file_path,
                category=category
            )
            
            if result:
                success_count += 1
                logger.info(f"재인덱싱 성공: {os.path.basename(file_path)}")
            else:
                failed_count += 1
                logger.error(f"재인덱싱 실패: {os.path.basename(file_path)}")
                
        except Exception as e:
            failed_count += 1
            logger.error(f"재인덱싱 중 오류 발생: {os.path.basename(file_path)} - {str(e)}")
            
    return {"success": success_count, "failed": failed_count}

