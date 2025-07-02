"""
동의어 사전 관리 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from app.utils.synonym_builder import (
    get_qwen_synonym_builder, 
    set_qwen_models_for_synonyms,
    update_synonyms_from_text
)
from app.core.elasticsearch import get_elasticsearch_client
from app.utils.indexing_utils import ES_INDEX_NAME
from app.utils.model_loader import get_llm_model_and_tokenizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/synonyms", tags=["synonyms"])

class SynonymAddRequest(BaseModel):
    term: str
    synonyms: List[str]

class SynonymResponse(BaseModel):
    success: bool  
    message: str
    data: Optional[Dict[str, Any]] = None

@router.get("/dictionary", response_model=SynonymResponse)
async def get_synonym_dictionary():
    """현재 동의어 사전 조회"""
    try:
        builder = get_qwen_synonym_builder()
        synonyms = builder.get_all_synonyms()
        
        stats = {
            "total_terms": len(synonyms),
            "total_synonyms": sum(len(syns) for syns in synonyms.values()),
            "average_synonyms_per_term": sum(len(syns) for syns in synonyms.values()) / len(synonyms) if synonyms else 0
        }
        
        return SynonymResponse(
            success=True,
            message=f"동의어 사전 조회 완료: {stats['total_terms']}개 용어",
            data={"synonyms": synonyms, "stats": stats}
        )
    except Exception as e:
        logger.error(f"동의어 사전 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add", response_model=SynonymResponse)
async def add_manual_synonyms(request: SynonymAddRequest):
    """수동으로 동의어 추가"""
    try:
        builder = get_qwen_synonym_builder()
        await builder.add_manual_synonyms(request.term, request.synonyms)
        
        return SynonymResponse(
            success=True,
            message=f"동의어 추가 완료: '{request.term}' -> {request.synonyms}"
        )
    except Exception as e:
        logger.error(f"동의어 추가 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{term}", response_model=SynonymResponse)
async def search_synonyms(term: str):
    """특정 용어의 동의어 검색"""
    try:
        builder = get_qwen_synonym_builder()
        synonyms = builder.get_synonyms(term)
        
        return SynonymResponse(
            success=True,
            message=f"'{term}' 동의어 검색 완료",
            data={"term": term, "synonyms": synonyms}
        )
    except Exception as e:
        logger.error(f"동의어 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/expand", response_model=SynonymResponse) 
async def expand_keywords(keywords: List[str]):
    """키워드 목록을 동의어로 확장"""
    try:
        builder = get_qwen_synonym_builder()
        expanded = builder.get_expanded_keywords(keywords)
        
        return SynonymResponse(
            success=True,
            message=f"{len(keywords)}개 키워드를 {len(expanded)}개로 확장",
            data={"original": keywords, "expanded": expanded}
        )
    except Exception as e:
        logger.error(f"키워드 확장 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=SynonymResponse)
async def get_synonym_stats():
    """동의어 사전 통계"""
    try:
        builder = get_qwen_synonym_builder()
        synonyms = builder.get_all_synonyms()
        
        stats = {
            "total_terms": len(synonyms),
            "total_synonyms": sum(len(syns) for syns in synonyms.values()),
            "top_terms": sorted(
                [(term, len(syns)) for term, syns in synonyms.items()], 
                key=lambda x: x[1], reverse=True
            )[:10]
        }
        
        return SynonymResponse(
            success=True,
            message="동의어 통계 조회 완료",
            data=stats
        )
    except Exception as e:
        logger.error(f"동의어 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/build-from-elasticsearch", response_model=SynonymResponse)
async def build_synonyms_from_elasticsearch():
    """ES에 인덱싱된 모든 문서를 분석해서 동의어 사전을 구축합니다."""
    try:
        logger.info("🔍 ES 기반 동의어 구축 시작")
        
        # 1. ES 클라이언트 가져오기
        es_client = get_elasticsearch_client()
        if not es_client:
            raise HTTPException(status_code=500, detail="Elasticsearch 연결 실패")
        
        # 2. 동의어 빌더 모델 확인 (이미 설정되어 있는지 확인)
        builder = get_qwen_synonym_builder()
        
        if not builder.llm_model or not builder.tokenizer:
            logger.info("🔄 Qwen 모델이 설정되지 않음, 모델 로드 중...")
            llm_model, tokenizer = get_llm_model_and_tokenizer()
            if not llm_model or not tokenizer:
                raise HTTPException(status_code=500, detail="LLM 모델 로드 실패")
            
            set_qwen_models_for_synonyms(llm_model, tokenizer)
            logger.info("✅ Qwen 모델 설정 완료")
        else:
            logger.info("✅ 기존 Qwen 모델 재사용")
        
        # 3. ES에서 모든 문서의 텍스트 가져오기
        all_texts = []
        scroll_size = 100
        processed_chunks = 0
        
        search_body = {
            "size": scroll_size,
            "_source": ["text", "source"],
            "query": {"match_all": {}}
        }
        
        logger.info(f"🔍 ES 검색 시작 - 배치 크기: {scroll_size}")
        response = es_client.search(index=ES_INDEX_NAME, body=search_body, scroll="2m")
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        
        # 첫 번째 배치 처리
        for i, hit in enumerate(hits):
            text = hit["_source"].get("text", "")
            source = hit["_source"].get("source", "unknown")
            if text and len(text.strip()) > 10:  # 의미있는 텍스트만
                all_texts.append(text)
                processed_chunks += 1
                
                # 처리된 청크의 헤드 30자 디버그 출력 (처음 5개만)
                if processed_chunks <= 5:
                    preview = text.strip()[:30].replace('\n', ' ')
                    logger.info(f"📄 청크 #{processed_chunks} ({source}): '{preview}...'")
        
        # 나머지 문서들 scroll로 가져오기
        batch_num = 1
        while len(hits) > 0:
            response = es_client.scroll(scroll_id=scroll_id, scroll="2m")
            hits = response["hits"]["hits"]
            batch_num += 1
            
            if batch_num % 5 == 0:  # 5배치마다 진행상황 출력
                logger.info(f"🔄 배치 #{batch_num} 처리 중... (현재까지 {processed_chunks}개 청크)")
            
            for hit in hits:
                text = hit["_source"].get("text", "")
                source = hit["_source"].get("source", "unknown")
                if text and len(text.strip()) > 10:
                    all_texts.append(text)
                    processed_chunks += 1
        
        # scroll 정리
        es_client.clear_scroll(scroll_id=scroll_id)
        
        logger.info(f"✅ ES 수집 완료: 총 {len(all_texts)}개 텍스트 청크, {batch_num}개 배치 처리")
        
        # 4. 모든 텍스트 사용 (샘플링 제거)
        logger.info(f"🎯 전체 텍스트 사용: {len(all_texts)}개")
        combined_text = "\n\n".join(all_texts)
        
        # 5. Qwen으로 동의어 생성
        logger.info("🤖 Qwen 모델로 동의어 생성 중...")
        await update_synonyms_from_text(combined_text)
        
        # 6. 결과 확인
        builder = get_qwen_synonym_builder()
        final_synonyms = builder.get_all_synonyms()
        terms_count = len(final_synonyms)
        total_synonyms = sum(len(syns) for syns in final_synonyms.values())
        
        logger.info(f"✅ 동의어 구축 완료: {terms_count}개 용어, {total_synonyms}개 동의어")
        
        return SynonymResponse(
            success=True,
            message=f"ES 기반 동의어 구축 완료: {terms_count}개 용어, {total_synonyms}개 동의어",
            data={
                "terms_count": terms_count,
                "total_synonyms": total_synonyms,
                "processed_texts": len(all_texts),
                "total_es_texts": len(all_texts),
                "sample_synonyms": dict(list(final_synonyms.items())[:5])  # 샘플 5개
            }
        )
        
    except Exception as e:
        logger.error(f"❌ ES 기반 동의어 구축 실패: {e}")
        raise HTTPException(status_code=500, detail=f"동의어 구축 실패: {str(e)}")
