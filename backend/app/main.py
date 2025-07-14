import sys
import os
import asyncio
import uuid
import time
import json
import traceback
import difflib  # 유사도 비교를 위한 표준 라이브러리
import glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CUDA 메모리 관리 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Depends, Request, Query, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from elasticsearch import Elasticsearch
from app.utils.indexing_utils import process_and_index_file, ES_INDEX_NAME, check_file_exists_sync, format_file_size
from fastapi.responses import FileResponse, StreamingResponse
import mimetypes  # 파일 타입 감지용
from collections import Counter  # 피드백 통계용
import re  # 정규식 사용을 위한 모듈
from pathlib import Path  # 경로 처리용
from fastapi import Request as FastAPIRequest # FastAPI의 Request를 명시적으로 임포트

# 피드백 분석 모듈 import
from app.utils.feedback_analyzer import FeedbackAnalyzer
# 파일 관리 모듈 import
from app.utils.file_manager import delete_indexed_file
from app.utils.indexing_utils import strip_uuid_prefix
# 문서 출처 및 하이라이트
from app.utils.source_preview_utils import (
    apply_highlighting,
    filter_meaningful_keywords,

)
from app.utils.search_enhancer import QueryExpander
#llm 호출
from app.utils.model_loader import get_llm_model_and_tokenizer
#분리 모듈 import
from app.core.elasticsearch import get_elasticsearch_client
from app.core.embeddings import get_embedding_function
from app.core.retriever import ElasticsearchRetriever, generate_llm_response

# 모델 임포트
import torch
import torch.nn.functional as F
from transformers import TextIteratorStreamer
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import traceback
from threading import Thread

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 파일 핸들러
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler("app.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# 스트림 핸들러
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
# traceback.print_exc() # 애플리케이션 시작 시 불필요한 traceback 제거

# 설정 상수
# LLM_MODEL_NAME = r"/home/root/ko-gemma-v1"
#EMBEDDING_MODEL_NAME = r"/home/root/KURE-v1"
# RERANKER_MODEL_NAME = r"/home/root/ko-reranker"
#LLM_MODEL_NAME = r"/home/root/Qwen3-8B"
LLM_MODEL_NAME = r"/home/root/Gukbap-Qwen2.5-7B"
EMBEDDING_MODEL_NAME = r"/home/root/kpf-sbert-v1.1"
#RERANKER_MODEL_NAME = r"/home/root/bge-reranker-large"
RERANKER_MODEL_NAME = r"/home/root/ms-marco-electra-base"
ES_HOST = "http://172.10.2.70:9200"
STATIC_DIR = "app/static"
IMAGE_DIR = os.path.join(STATIC_DIR, "document_images")
os.makedirs(IMAGE_DIR, exist_ok=True)



def check_paddle_gpu():
    try:
        import paddle
        print(f"PaddlePaddle GPU 사용 가능: {paddle.is_compiled_with_cuda()}")
        print(f"현재 디바이스: {paddle.get_device()}")
    except:
        print("PaddlePaddle 확인 불가")

# 함수 호출해서 확인
check_paddle_gpu()


# 질문 요청 모델
class QuestionRequest(BaseModel):
    question: str
    category: Optional[str] = "메뉴얼"
    history: Optional[List[Dict[str, Any]]] = []


# FastAPI 앱 초기화
app = FastAPI(title="RAG Chatbot API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")





def get_reranker_model():
    """Reranker 모델을 로드합니다."""
    print("Loading reranker model...")
    try:
        reranker = CrossEncoder(
            RERANKER_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Reranker model loaded successfully.")
        return reranker
    except Exception as e:
        print(f"Reranker 모델 로딩 중 오류 발생: {e}")
        print(f"Reranker 모델 경로: {RERANKER_MODEL_NAME}")
        print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
        traceback.print_exc()
        return None



# 향상된 리랭커 클래스 정의
class EnhancedLocalReranker:
    def __init__(self, reranker_model: Any, top_n=18):  # top_n 증가 (15 → 18)
        self.reranker = reranker_model
        self.top_n = top_n
        # 성능 최적화를 위한 캐시 추가
        self._cache = {}
        self._cache_size = 150  # 캐시 크기 증가 (100 → 150)
        self._cache_ttl = 7200  # 캐시 유효 시간 증가 (1시간 → 2시간)
        # 배치 처리 최적화
        self.batch_size = 24  # 배치 크기 증가 (16 → 24)

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        # 리랭킹 활성화 - 리랭커로 문서 재정렬
        print(f"🔄 리랭킹 시작: {len(docs)}개 문서 처리")

        # 캐시 키 생성 (쿼리와 문서 ID 조합)
        # chunk_id를 명시적으로 문자열로 변환하여 에러 방지
        query_normalized = query.lower().strip()
        cache_key = f"{query_normalized}:{','.join([str(d.metadata.get('chunk_id', i)) for i, d in enumerate(docs[:10])])}"

        # 캐시에서 결과 확인
        current_time = time.time()
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if current_time - cache_entry["timestamp"] < self._cache_ttl:
                print(f"리랭킹 캐시 적중: '{query[:30]}...'")
                return cache_entry["results"]

        # 캐시 정리 (필요시) - LRU 방식 최적화
        if len(self._cache) >= self._cache_size:
            oldest_keys = sorted(
                self._cache.keys(), 
                key=lambda k: self._cache[k]["timestamp"]
            )[:len(self._cache) // 3]  # 1/3 정도 삭제 (1/4에서 증가)
            for old_key in oldest_keys:
                del self._cache[old_key]

        try:
            # 메모리 최적화를 위한 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 상위 12개 문서 리랭킹 (원래 10개에서 상향) → 12개 그대로 유지
            docs_to_rerank = docs[:12]
            pairs = [(query_normalized, doc.page_content) for doc in docs_to_rerank]
            # 배치 처리로 성능 최적화
            scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                with torch.no_grad():
                    batch_scores = self.reranker.predict(batch_pairs)
                    scores.extend(batch_scores)

            # CrossEncoder 로짓을 소프트맥스로 변환 (Temperature Scaling 적용)
            import torch.nn.functional as F
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # Temperature scaling으로 차이 증폭 (temperature < 1 = 차이 벌리기)
            temperature = 0.3  # 낮을수록 차이가 더 벌어짐
            softmax_scores = F.softmax(scores_tensor / temperature, dim=0).tolist()


            # 정규화 로직 제거하고 소프트맥스 점수 사용
            for doc, raw_score, soft_score in zip(docs_to_rerank, scores, softmax_scores):
                doc.metadata["rerank_score"] = soft_score
                doc.metadata["raw_rerank_score"] = raw_score
                doc.metadata["relevance_score"] = soft_score
            for doc in docs_to_rerank:
                content = doc.page_content.lower()
                source = doc.metadata.get('source', '')
                
                if '사내규정모음집' in source and ('회갑' in content or '환갑' in content):
                    current_score = doc.metadata.get("rerank_score", 0.0)
                    boosted_score = current_score * 10.0  # 10배 부스팅
                    doc.metadata["rerank_score"] = boosted_score
                    doc.metadata["relevance_score"] = boosted_score
            # 리랭킹된 문서를 리랭커 점수 기준으로 정렬
            sorted_docs = sorted(
                docs_to_rerank,
                key=lambda x: x.metadata.get("rerank_score", 0.0),
                reverse=True,
            )

            # 상위 75% 퍼센타일 임계값 계산
            sorted_scores = sorted(softmax_scores, reverse=True)
            percentile_75_index = int(len(sorted_scores) * 0.5)  # 상위 50% 위치
            threshold_75 = sorted_scores[percentile_75_index] if percentile_75_index < len(sorted_scores) else 0



            # 상위 50% 퍼센타일 이상만 필터링 (RELEVANCE_SCORE 전역변수 사용 안함)
            filtered_docs = [
                doc
                for doc in sorted_docs
                if doc.metadata.get("rerank_score", 0.0) >= threshold_75
            ]



            # RELEVANCE_SCORE 관련 추가 필터링 로직 완전 제거
            # (기존 softmax_threshold, 추가 보장 로직 등 모두 삭제)

            # 나머지 문서는 추가하지 않음 (사용자 요구사항: 고품질만)
            # remaining_docs = [doc for doc in docs[12:] if doc not in docs_to_rerank]
            # sorted_docs.extend(remaining_docs)  # 이 부분 제거

            # 결과 캐싱
            result_docs = filtered_docs[:self.top_n]
            self._cache[cache_key] = {
                "results": result_docs,
                "timestamp": current_time
            }

            print(f"✅ 리랭킹 완료: {len(result_docs)}개 문서 (50% 퍼센타일 임계값: {threshold_75:.6f})")
            print("[RAW]", ", ".join(f"{s:.4f}" for s in scores[:12]))
            print("[SOFTMAX]", ", ".join(f"{s:.4f}" for s in softmax_scores[:12]))
            print(f"[SELECTED] {len(filtered_docs)}개 문서가 50% 퍼센타일 기준을 통과")
            return result_docs

        except Exception as e:
            print(f"Reranking 중 오류 발생: {e}")
            traceback.print_exc()
            return docs[:self.top_n]  # 오류 시 원본 문서 상위 n개 반환





# 검색 및 결합 함수
async def search_and_combine(
    es_client: Any,
    embedding_function: Any,
    reranker_model: Any,
    llm_model: Any,
    tokenizer: Any,
    query: str,
    category: str,
    conversation_history: List[Dict] = None,
) -> Dict[str, Any]:
    """Elasticsearch 검색, Reranking, LLM 답변 생성을 수행합니다."""

    if conversation_history is None:
        conversation_history = []

    # 대화 기록 최적화 (최근 5턴으로 제한)
    def optimize_conversation_history(
        history: List[Dict], max_turns: int = 3  # 최대 턴 수 감소 (5→3)
    ) -> List[Dict]:
        if len(history) > max_turns * 2:
            return history[-max_turns * 2 :]
        return history

    if conversation_history:
        conversation_history = optimize_conversation_history(conversation_history)

    # 기본 유효성 검사
    if query is None:
        return {"answer": "질문을 입력해주세요.", "sources": []}

    query = query.strip()
    if not query:
        return {
            "answer": "질문 내용을 입력해주세요. 공백만으로는 검색할 수 없습니다.",
            "sources": [],
        }



    start_time = time.time()
    print(f"검색 및 결합 시작: {query[:30]}...")
    
    # Redis 캐싱 적용: 동일한 쿼리의 중복 처리 방지 - 성능 대폭 개선
    from utils.cache_utils import RedisCache, CacheKeys, CACHE_TTL_SEARCH
    
    # 캐시 키 생성 (질문 + 카테고리 기반)
    cache_key = RedisCache.generate_key(
        CacheKeys.CHAT, 
        {"query": query, "category": category}
    )
    
    # 캐시에서 결과 확인
    cached_result = RedisCache.get(cache_key)
    if cached_result:
        print(f"캐시된 결과 사용: {cache_key}")
        # 캐시된 결과에 성능 메트릭 추가
        cached_result["from_cache"] = True
        cached_result["processing_time"]["cache_hit"] = round(time.time() - start_time, 3)
        
        # 캐시된 결과를 스트리밍 방식으로 반환
        async def cached_response_stream():
            # 캐시된 텍스트 응답을 작은 청크로 나누어 스트리밍
            answer_text = cached_result.get("answer", "")
            sources = cached_result.get("sources", [])
            cited_sources = cached_result.get("cited_sources", [])
            
            # 텍스트를 토큰 단위로 스트리밍 (여기서는 간단히 문자 단위로 나눔)
            chunk_size = 4  # 4자씩 스트리밍 (실제 토큰 크기와 유사하게)
            for i in range(0, len(answer_text), chunk_size):
                chunk = answer_text[i:i+chunk_size]
                yield f"data: {json.dumps({'token': chunk})}\n\n"
                await asyncio.sleep(0.01)  # 실제 스트리밍 효과를 위한 짧은 지연
            
            # 소스 정보 전송
            yield f"data: {json.dumps({'event': 'sources', 'sources': sources, 'cited_sources': cited_sources})}\n\n"
            
            # 캐시 사용 정보 전송 (클라이언트에서 캐시 사용 여부 표시 가능)
            yield f"data: {json.dumps({'event': 'cache_info', 'from_cache': True})}\n\n"
            
            # 스트림 종료 이벤트
            yield f"data: {json.dumps({'event': 'eos', 'message': 'Stream ended (from cache).'})}\n\n"
        
        # 캐시된 응답을 스트리밍 형태로 반환
        return StreamingResponse(
            cached_response_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        
        # 기존 코드 (한 번에 반환)
        # return cached_result

    try:
        # 메모리 최적화를 위한 토치 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. 검색  - 검색 결과 수 최적화
        retrieval_start = time.time()
        # ElasticsearchRetriever - 2단계 검색 방식으로 개선 (Qwen 대안쿼리 제거)
        retriever = ElasticsearchRetriever(
            es_client=es_client,
            index_name=ES_INDEX_NAME,
            embedding_function=embedding_function,
            category=category,
            k=15,
            llm_model=None,      # Qwen 대안쿼리 비활성화
            tokenizer=None       # Qwen 대안쿼리 비활성화
        )

        docs = await retriever.async_get_relevant_documents(query)
        retrieval_time = time.time() - retrieval_start
        print(f"Retrieval time: {retrieval_time:.2f}s, Found {len(docs)} docs from ES.")
        print(f"ES 검색 성능 분석: 검색 시간 {retrieval_time:.2f}초, 문서 수 {len(docs)}개")

        # 검색 결과가 없을 경우 조기 반환
        if not docs:
            return {
                "answer": "검색된 관련 문서가 없습니다. 다른 질문을 시도해 보세요.",
                "sources": [],
            }

        # 메모리 최적화를 위한 토치 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Reranking (최적화 - 비동기 처리)
        rerank_start = time.time()
        # EnhancedLocalReranker는 top_n=18 (성능 최적화 설정)
        #reranker = EnhancedLocalReranker(reranker_model, top_n=18)

        try:
            #reranked_docs = reranker.rerank(query, docs) 현재 리랭킹 비활성화 -> except로 빠지게하였음 
            rerank_time = time.time() - rerank_start
            print(f"Reranking time: {rerank_time:.2f}s, Reranked to {len(reranked_docs)} docs.")
            print(f"리랭킹 성능 분석: 리랭킹 시간 {rerank_time:.2f}초, 문서 수 {len(reranked_docs)}개")
        except Exception as rerank_error:
            #print(f"Reranking 중 오류 발생, 원본 문서 사용: {rerank_error}")
            #traceback.print_exc()
            # 리랭킹 실패 시 원본 문서 사용
            reranked_docs = docs[:15]  # 상위 15개만 사용
            rerank_time = time.time() - rerank_start
            print(f"docs rank : {rerank_time:.2f}s")

        # 최종 토큰 수 제한
        # 컨텍스트 크기를 최대 8,192 토큰으로 제한
        max_tokens = 8192
        token_count = 0
        final_docs = []

        for doc in reranked_docs:
            if not doc.page_content:
                continue

            # 입력에 대해 예상 토큰 수 계산 (한국어 토큰화는 복잡하므로 근사치 사용)
            approx_tokens = len(doc.page_content) / 3
            if token_count + approx_tokens > max_tokens:
                break

            token_count += approx_tokens
            final_docs.append(doc)
        for i, doc in enumerate(docs[:15]):
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 0)
            score = doc.metadata.get('relevance_score', 0)
            print(f"  {i+1}. {source} p.{page} (score: {score:.3f})")
        # LLM 입력 형식으로 변환
        # 리랭킹 결과 문서들을 하나의 컨텍스트로 결합
        context_chunks = []
        source_metadata = []

        for i, doc in enumerate(final_docs):
            # 소스 정보 URL 인코딩
            source_path = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", i)
            element_type = doc.metadata.get("element_type", "text")

            # 텍스트에서 불필요한 공백과 개행 정리
            chunk_text = doc.page_content.strip()
            chunk_text = " ".join(chunk_text.split())

            # 파일명에서 UUID 제거 (UUID_파일명.확장자 형식 가정)
            clean_filename = strip_uuid_prefix(source_path)
            source_label = f"[{clean_filename} p.{page_num}]" if page_num > 1 else f"[{clean_filename}]"
            context_chunks.append(f"{source_label} {chunk_text}")
            print(f"📄 Context chunk {i+1}: {source_label}")
            # 표와 이미지 정보 추가
            metadata_item = {
                "path": source_path,
                "display_name": clean_filename,  # 화면 표시용 정제된 파일명 추가
                "page": page_num,
                "chunk_id": chunk_id,
                "score": doc.metadata.get("relevance_score", 0),
                "element_type": element_type,
            }
            
            # 표 관련 정보 추가
            if element_type == "table_row":
                metadata_item.update({
                    "table_id": doc.metadata.get("table_id"),
                    "table_caption": doc.metadata.get("table_caption"),
                    "row_index": doc.metadata.get("row_index"),
                    "n_rows": doc.metadata.get("n_rows"),
                })
            
            # 이미지 관련 정보 추가 (개선)
            has_images = doc.metadata.get("has_images", False)
            images = doc.metadata.get("images", [])
            image_captions = doc.metadata.get("image_captions", [])
            
            if has_images and images:
                # 프론트엔드에서 사용하기 쉽게 이미지 정보 구조화
                processed_images = []
                for idx, img_path in enumerate(images):
                    img_info = {
                        "path": img_path,
                        "url": f"/api/image-viewer/{img_path}",
                        "caption": "",
                        "ocr_text": "",
                        "ai_description": ""
                    }
                    
                    # 캡션 정보 추가
                    if image_captions and idx < len(image_captions):
                        caption_info = image_captions[idx]
                        if isinstance(caption_info, dict):
                            img_info["caption"] = caption_info.get("ai_caption", "")
                            img_info["ocr_text"] = caption_info.get("ocr_text", "")
                            img_info["ai_description"] = caption_info.get("ai_caption", "")
                        elif isinstance(caption_info, str):
                            img_info["caption"] = caption_info
                    
                    # 기본 캡션 설정
                    if not img_info["caption"]:
                        img_info["caption"] = f"페이지 {page_num} 이미지 {idx + 1}"
                    
                    processed_images.append(img_info)
                
                metadata_item.update({
                    "has_images": True,
                    "images": images,
                    "image_captions": image_captions,
                    "processed_images": processed_images,
                    "image_count": len(processed_images)
                })
            else:
                metadata_item.update({
                    "has_images": False,
                    "images": [],
                    "processed_images": [],
                    "image_count": 0
                })
            
            source_metadata.append(metadata_item)

        full_context = "\n\n".join(context_chunks)
        print(f"Combined context length: {len(full_context)} characters.")
        # LLM이 프롬프트 텍스트를 생성하는단계 
        '''요약이나 제목 생성을 쓸때도 사용하면 여러가지 답변형식이 가능하나 현재는 안쓰고 있음 
        -> history 기반으로 답변을 생성하나 통계기반 답변으로 답변에 노이즈가 낄 수 있고 추후 개선 필요함
        -> 멀티턴 방식으로 진행하기 위해 히스토리를 살리되 각각의 채팅세션으로 캐시나 히스토리를 분리해야함
        '''
        llm_start = time.time()
        answer = await generate_llm_response(
            tokenizer,
            query,
            final_docs,
            0.1,
            conversation_history
        )
        prompt_text = answer["prompt_text"]
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
#--------------- 검색 끝 답변 시작----------------------------
        #인공지능 모델 파라미터
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=2048,  #답변 길이
                temperature=0.2, # 답변 자유도 높아지면 할루시네이션 발생
                do_sample=True,
                repetition_penalty=1.1,  # 반복 억제를 위한 페널티 추가
                eos_token_id=tokenizer.eos_token_id,  # 토큰으로 끝나면 자동 종료
                pad_token_id=tokenizer.eos_token_id
            )

        answer = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # 응답이 None인 경우 대체 응답 사용 (방어 코드)
        if answer is None:
            print("LLM 응답이 None입니다. 대체 응답을 사용합니다.")
            answer = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 다시 질문해 주세요."
        
        # 응답 정제 - Qwen의 ChatML 포맷 처리
        def clean_response(resp):
            """응답 텍스트에서 불필요한 시스템 메시지, 사용자 메시지 등을 제거합니다."""
            if not resp:
                return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
                
            # 1. 시스템 메시지 제거
            if resp.startswith("system\n"):
                parts = resp.split("user\n")
                if len(parts) > 1:
                    resp = parts[1]
                    parts = resp.split("assistant\n")
                    if len(parts) > 1:
                        resp = parts[1].strip()
                        return resp
            
            # 2. assistant 접두사 제거
            if "assistant\n" in resp:
                parts = resp.split("assistant\n")
                if len(parts) > 1:
                    resp = parts[-1].strip()
                    return resp
            
            # 3. 그 외의 경우
            # 불필요한 태그 제거
            for tag in ["system\n", "user\n", "assistant\n", "system:", "user:", "assistant:", "system", "user", "assistant"]:
                if resp.startswith(tag):
                    resp = resp[len(tag):].strip()
                    
            return resp.strip()


    #---------------------- 답변 끝 출처 시작 -----------------------
    # 응답 정제 적용
        cleaned_answer = clean_response(answer)
        
        # 검색해서 히트된 모든 문서를 출처로 표시 (간단하고 직관적)
        cited_sources = []
            
        # 문서와 이미지를 쌍으로 묶어서 출처 생성 (2n개 방식)
        print(f"문서-이미지 쌍 생성 시작: 총 {len(source_metadata)}개 문서")
        cited_sources = []
        qualified_sources = []
        
        # 스코어 임계값 설정 제거 - 직접 인용된 문서만 출처로 포함
        print("스코어 임계값 설정 없이 직접 인용된 문서만 출처로 포함합니다.")
        
        # 초기화
        for i, meta in enumerate(source_metadata):
            meta["is_cited"] = False  # 초기값 설정
        
        # 메타데이터에 is_cited 설정 - 초기화
        cited_sources = []
        qualified_sources = []
        print("직접 인용 여부에 따라 출처를 선별합니다.")
        
        # 유효한 응답이 있는 경우 추가적으로 인용 기반 출처 선별
        if cleaned_answer and isinstance(cleaned_answer, str) and cleaned_answer.strip():
            print(f"LLM 인용 기반 출처 재선별 시작: 총 {len(source_metadata)}개 문서")
            directly_cited_refs = set()
            
            # 1. LLM 응답에서 직접 인용된 파일명 추출 (예: [파일명 p.페이지])
            import re
            citation_pattern = r'\[([^[\]]+?)(?:\s+p\.(\d+))?\]'
            cited_files = re.findall(citation_pattern, cleaned_answer)

            for file_part, page_part in cited_files:
                clean_file = file_part.strip()
                if clean_file:
                    if page_part:  # 페이지 번호가 있으면
                        directly_cited_refs.add(f"{clean_file.lower()}_p{page_part}")
                        print(f"🎯 응답에서 직접 인용된 파일: '{clean_file}' p.{page_part}")
                    else:  # 페이지 번호가 없으면
                        directly_cited_refs.add(clean_file.lower())
                        print(f"🎯 응답에서 직접 인용된 파일: '{clean_file}' (페이지 없음)")

            # 2. 직접 인용된 문서들만 선택
            new_qualified_sources = []

            if directly_cited_refs:  # 직접 인용이 있는 경우에만
                for i, meta in enumerate(source_metadata):
                    display_name = meta.get('display_name', 'unknown')
                    page_num = meta.get('page', '')
                    score = meta.get("score", 0)
                    has_images = meta.get("has_images", False)
                    images = meta.get("images", [])
                    
                    # 정확한 매칭 (파일명 + 페이지 번호)
                    is_directly_cited = False
                    for cited_ref in directly_cited_refs:
                        if '_p' in cited_ref:  # 페이지 번호 포함된 참조
                            file_name, page_info = cited_ref.split('_p')
                            expected_page = int(page_info)
                            if file_name in display_name.lower() and page_num == expected_page:
                                is_directly_cited = True
                                print(f"  🎯 정확한 매칭: '{display_name}' p.{page_num}")
                                break
                        else:  # 파일명만 있는 참조 (페이지 번호 없음)
                            if cited_ref in display_name.lower():
                                is_directly_cited = True
                                print(f"  🎯 파일명 매칭: '{display_name}'")
                                break
                    
                    if is_directly_cited:
                        new_qualified_sources.append({
                            'meta': meta,
                            'score': max(score, 0.95),
                            'index': i,
                            'directly_cited': True,
                            'reason': 'direct_citation'
                        })
                        print(f"  ✅ 직접 인용 문서: {display_name} (스코어: {score:.3f})")
                        
                        # 이미지가 있는 경우 별도의 항목으로 추가
                        if has_images and images:
                            for img_index, img_path in enumerate(images):
                                new_qualified_sources.append({
                                    'meta': {
                                        'document_id': meta.get('document_id'),
                                        'path': img_path,
                                        'display_name': f"{display_name} - 이미지 {img_index + 1}",
                                        'page': page_num,
                                        'chunk_id': meta.get('chunk_id', i),
                                        'score': max(score, 0.95),
                                        'has_images': True,
                                        'images': [img_path]
                                    },
                                    'score': max(score, 0.95),
                                    'index': i,
                                    'directly_cited': False,
                                    'reason': 'related_image'
                                })
                                print(f"  ✅ 관련 이미지 추가: {display_name} - 이미지 {img_index + 1} (스코어: {max(score, 0.95):.3f})")

            # 직접 인용이 없으면 출처를 추가하지 않음
            if not new_qualified_sources:
                print("  ⚠️ 직접 인용 매칭 실패, 출처 추가 없음")

            # 정렬: 직접 인용 > 스코어 순
            new_qualified_sources.sort(key=lambda x: (x.get('directly_cited', False), x['score']), reverse=True)
            final_sources = new_qualified_sources

            for source in final_sources:
                reason = source.get('reason', 'unknown')
                score = source['score']
                display_name = source['meta'].get('display_name', 'unknown')
                print(f"  ✅ 최종 출처: {display_name} (이유: {reason}, 스코어: {score:.3f})")

            # 메타데이터에 is_cited 설정 업데이트
            cited_sources = []
            document_sources = [source for source in final_sources if source['meta'].get('element_type', 'text') != 'image'][:2]
            image_sources = [source for source in final_sources if source['meta'].get('element_type', 'text') == 'image'][:2]
            final_limited_sources = document_sources + image_sources
            cited_indices = {source['index'] for source in final_limited_sources if 'index' in source}
            for i, meta in enumerate(source_metadata):
                if i in cited_indices:
                    meta["is_cited"] = True
                    cited_sources.append(meta)
                else:
                    meta["is_cited"] = False

            print(f"최종 선별된 출처: {len(cited_sources)}개 (LLM 직접 인용, 출처 2개 및 이미지 2개로 제한)")
            for source in final_sources:
                meta = source['meta']
                element_type = meta.get("element_type", "text")
                type_desc = "표" if element_type in ["table_row", "table"] else "이미지" if meta.get("has_images") else "텍스트"
                reason = source.get('reason', 'unknown')
                existing_display_name = meta.get('display_name', 'unknown')
                page_num = meta.get("page", "")

                if page_num and 'p.' not in existing_display_name:
                    display_name = f"{existing_display_name} p.{page_num}"
                else:
                    display_name = existing_display_name
                print(f"  📑 {display_name} - 스코어: {source['score']:.3f} ({type_desc}, {reason})")
            llm_time = time.time() - llm_start
            print(f"LLM generation time: {llm_time:.2f}s")
            print(f"응답에 포함된 출처 수: {len(cited_sources)}")
            print(f"LLM 생성 성능 분석: 생성 시간 {llm_time:.2f}초")

            # 최종 응답 생성
            final_result = {
                "answer": cleaned_answer,  # 정제된 응답 사용
                "sources": source_metadata,
                "cited_sources": cited_sources,
                "processing_time": {
                    "retrieval": round(retrieval_time, 2),
                    "reranking": round(rerank_time, 2),
                    "llm_generation": round(llm_time, 2),
                    "total": round(time.time() - start_time, 2),
                },
            }
        # 결과 캐싱 (재사용을 위해)
        try:
            RedisCache.set(cache_key, final_result, CACHE_TTL_SEARCH)
            print(f"응답 결과 캐싱 완료: {cache_key}")
        except Exception as cache_error:
            print(f"캐싱 중 오류 발생 (무시됨): {cache_error}")
        
        return final_result

    except Exception as e:
        print(f"검색 및 응답 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "answer": f"요청을 처리하는 중 오류가 발생했습니다: {str(e)}",
            "sources": [],
            "cited_sources": [],  # 빈 cited_sources 추가
            "error": str(e),  # 오류 정보 추가
            "processing_time": {  # 일관된 구조 유지를 위한 처리 시간 정보 추가
                "total": round(time.time() - start_time, 2),
                "retrieval": 0,
                "enhancement": 0,
                "reranking": 0,
                "llm_generation": 0
            }
        }


# SQLCoder 모델 로드 함수 추가
def get_sqlcoder_model():
    """SQLCoder 모델을 로드합니다."""
    print("SQLCoder 모델 로딩 중...")
    try:
        from utils.sqlcoder_utils import load_sqlcoder_model
        model, tokenizer = load_sqlcoder_model()
        
        if model is None or tokenizer is None:
            print("SQLCoder 모델 로드 실패")
            return None, None
            
        print("SQLCoder 모델 로드 성공")
        return model, tokenizer
    except Exception as e:
        print(f"SQLCoder 모델 로딩 중 오류 발생: {e}")
        traceback.print_exc()
        return None, None

# 전역 변수 선언 (초기화는 startup에서)
es_client = None
embedding_function = None
llm_model = None
tokenizer = None
reranker_model = None
sqlcoder_model = None
sqlcoder_tokenizer = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 초기화"""
    global es_client, embedding_function, llm_model, tokenizer, reranker_model, sqlcoder_model, sqlcoder_tokenizer
    
    print("🚀 서버 시작 - 모델 초기화 중...")
    
    # 모델 초기화 (한 번만 실행됨)
    es_client = get_elasticsearch_client()
    embedding_function = get_embedding_function()
    llm_model, tokenizer = get_llm_model_and_tokenizer()
    reranker_model = get_reranker_model()
    if reranker_model is None:
        print("⚠️ 리랭커 모델 로드 실패, 검색 속도가 느려질 수 있습니다.")
    sqlcoder_model, sqlcoder_tokenizer = get_sqlcoder_model()
    
    # indexing_utils에 모델 전달 (캡션 생성용)
    from app.utils.indexing_utils import set_shared_models
    from app.utils.synonym_builder import set_qwen_models_for_synonyms
    set_shared_models(llm_model, tokenizer)
    
    # 동의어 빌더에 Qwen 모델 설정 (자동 동의어 사전 구축용)
    set_qwen_models_for_synonyms(llm_model, tokenizer)
    print("✅ Qwen 동의어 빌더 연결 완료")
    
    print("✅ 모든 모델 초기화 완료!")



class FeedbackRequest(BaseModel):
    messageId: str
    feedbackType: str
    rating: int
    content: str


# 피드백 저장 디렉토리 설정
FEEDBACK_DIR = "app/feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.json")


# 피드백 저장 함수
def save_feedback(feedback_data: Dict[str, Any]):
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedbacks = json.load(f)
        else:
            feedbacks = []
        feedbacks.append(feedback_data)
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedbacks, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"피드백 저장 중 오류 발생: {e}")
        return False


class ConversationRequest(BaseModel):
    userId: str  # 사용자 식별자 (임시로 문자열 사용, 실제로는 인증 기반 ID)
    conversationId: str  # 대화 식별자
    messages: List[Dict[str, Any]]  # 대화 메시지 목록


class ConversationLoadRequest(BaseModel):
    userId: str
    conversationId: str


# 대화 저장 디렉토리 설정
CONVERSATION_DIR = "app/conversations"
os.makedirs(CONVERSATION_DIR, exist_ok=True)


# 대화 저장 함수
def save_conversation(
    user_id: str, conversation_id: str, messages: List[Dict[str, Any]]
):
    try:
        file_path = os.path.join(CONVERSATION_DIR, f"{user_id}_{conversation_id}.json")
        conversation_data = {
            "userId": user_id,
            "conversationId": conversation_id,
            "messages": messages,
            "timestamp": datetime.now().isoformat(),
        }
        with open(file_path, "w") as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"대화 저장 중 오류 발생: {e}")
        return False


# 대화 불러오기 함수
def load_conversation(user_id: str, conversation_id: str):
    try:
        file_path = os.path.join(CONVERSATION_DIR, f"{user_id}_{conversation_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"대화 불러오기 중 오류 발생: {e}")
        return None


# 대화 저장 엔드포인트
@app.post("/api/conversations/save")
async def save_conversation_endpoint(request: ConversationRequest = Body(...)):
    try:
        success = save_conversation(
            request.userId, request.conversationId, request.messages
        )
        if success:
            return {"status": "success", "message": "대화가 저장되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="대화 저장에 실패했습니다.")
    except Exception as e:
        print(f"대화 저장 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"대화 저장 중 오류 발생: {str(e)}")


# 대화 불러오기 엔드포인트
@app.post("/api/conversations/load")
async def load_conversation_endpoint(request: ConversationLoadRequest = Body(...)):
    try:
        conversation = load_conversation(request.userId, request.conversationId)
        if conversation:
            return {"status": "success", "conversation": conversation}
        else:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    except Exception as e:
        print(f"대화 불러오기 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"대화 불러오기 중 오류 발생: {str(e)}"
        )


class UserSettingsRequest(BaseModel):
    userId: str  # 사용자 식별자 (임시로 문자열 사용, 실제로는 인증 기반 ID)
    settings: Dict[str, Any]  # 사용자 설정 데이터


class UserSettingsLoadRequest(BaseModel):
    userId: str


# 사용자 설정 저장 디렉토리 설정
SETTINGS_DIR = "app/settings"
os.makedirs(SETTINGS_DIR, exist_ok=True)


# 사용자 설정 저장 함수
def save_user_settings(user_id: str, settings: Dict[str, Any]):
    try:
        file_path = os.path.join(SETTINGS_DIR, f"{user_id}_settings.json")
        settings_data = {
            "userId": user_id,
            "settings": settings,
            "timestamp": datetime.now().isoformat(),
        }
        with open(file_path, "w") as f:
            json.dump(settings_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"사용자 설정 저장 중 오류 발생: {e}")
        return False


# 사용자 설정 불러오기 함수
def load_user_settings(user_id: str):
    try:
        file_path = os.path.join(SETTINGS_DIR, f"{user_id}_settings.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"사용자 설정 불러오기 중 오류 발생: {e}")
        return None


# 사용자 설정 저장 엔드포인트
@app.post("/api/settings/save")
async def save_user_settings_endpoint(request: UserSettingsRequest = Body(...)):
    try:
        success = save_user_settings(request.userId, request.settings)
        if success:
            return {"status": "success", "message": "사용자 설정이 저장되었습니다."}
        else:
            raise HTTPException(
                status_code=500, detail="사용자 설정 저장에 실패했습니다."
            )
    except Exception as e:
        print(f"사용자 설정 저장 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"사용자 설정 저장 중 오류 발생: {str(e)}"
        )


# 사용자 설정 불러오기 엔드포인트
@app.post("/api/settings/load")
async def load_user_settings_endpoint(request: UserSettingsLoadRequest = Body(...)):
    try:
        settings_data = load_user_settings(request.userId)
        if settings_data:
            return {"status": "success", "settings": settings_data["settings"]}
        else:
            return {
                "status": "not_found",
                "message": "사용자 설정을 찾을 수 없습니다.",
                "settings": {},
            }  # 파일이 없어도 에러 대신 빈 설정 반환
    except Exception as e:
        print(f"사용자 설정 불러오기 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"사용자 설정 불러오기 중 오류 발생: {str(e)}",
            "settings": {},
        }  # 500 대신 에러 메시지 반환


class SourcePreviewRequest(BaseModel):
    path: str
    page: int  # 페이지 정보는 여전히 유용할 수 있음 (UI 표시용)
    chunk_id: str  #str로 고정
    keywords: Optional[List[str]] = None  # 프론트엔드에서 전달하는 하이라이트 키워드 (선택)
    answer_text: Optional[str] = None  # 챗봇 응답 전체 텍스트 (선택)


# 참고 문서 미리보기 엔드포인트
@app.post("/api/source-preview")
async def source_preview_endpoint(request: SourcePreviewRequest = Body(...)):
    try:
        # 요청 검증
        if not request.path:
            return {
                "status": "error",
                "message": "요청에 파일 경로가 없습니다",
                "content": None,
            }
        
        
        # 1. 해당 페이지의 모든 chunk들 가져오기 (전체 페이지 재구성용)
        try:
            all_chunks_query = {
                "size": 100,  # 한 페이지에 충분한 chunk 수
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"source": request.path}},
                            {"term": {"page": request.page}}
                        ]
                    }
                },
                "sort": [
                    {"chunk_id": {"order": "asc"}},  # chunk_id 순서대로 정렬
                    {"row_index": {"order": "asc", "missing": "_last"}}  # 테이블 행 순서
                ]
            }
            
            response = es_client.search(index=ES_INDEX_NAME, body=all_chunks_query)
            all_chunks = response.get("hits", {}).get("hits", [])
            
            if not all_chunks:
                return {
                    "status": "error",
                    "message": "해당 페이지의 문서를 찾을 수 없습니다.",
                    "content": None
                }
            
            print(f"📄 찾은 총 chunk 수: {len(all_chunks)}")
            
            # 2. 전체 페이지 내용 재구성 및 Hit chunk 정보 수집
            page_content_parts = []
            hit_chunks_info = []  # 실제 Hit된 chunk 정보
            
            # 검색된 chunk_id와 일치하는 chunk 찾기
            target_chunk_id = str(request.chunk_id) if request.chunk_id else None
            
            for chunk_hit in all_chunks:
                chunk_source = chunk_hit["_source"]
                chunk_text = chunk_source.get("text", "")
                chunk_id = str(chunk_source.get("chunk_id", ""))
                element_type = chunk_source.get("element_type", "text")
                bm25_score  = chunk_hit.get("_score") or 0.0
                faiss_score = chunk_source.get("faiss_score") or 0.0
                relevance   = 0.7 * bm25_score + 0.3 * faiss_score

                chunk_metadata = chunk_source.get("metadata", {}) or {}
                chunk_metadata.update({                          # ✅ 기존 키 보존 + 새 키 추가
                    "bm25_score": bm25_score,
                    "faiss_score": faiss_score,
                    "relevance_score": relevance
                })
                # 현재 chunk가 검색된 Hit인지 확인
                is_hit_chunk = (target_chunk_id and chunk_id == target_chunk_id)
                
                if is_hit_chunk:
                    # Hit chunk 정보 저장
                    hit_info = {
                        "content": chunk_text,
                        "element_type": element_type,
                        "chunk_id": chunk_id,
                        "table_id": chunk_source.get("table_id"),
                        "row_no": chunk_source.get("row_no"),
                        "row_index": chunk_source.get("row_index"),
                        "metadata": chunk_metadata
                    }
                    hit_chunks_info.append(hit_info)
                    print(f"✅ Hit chunk 발견: {element_type}, chunk_id: {chunk_id}")
                if hit_chunks_info:
                    # 페이지에서 가장 높은 relevance_score 선택 (다른 방식 원하면 변경 가능)
                    page_metadata = hit_chunks_info[0]["metadata"].copy()
                    page_metadata["relevance_score"] = max(
                        h["metadata"].get("relevance_score", 0.0) for h in hit_chunks_info
                    )
                else:
                    page_metadata = {"relevance_score": 0.0}
                # 모든 chunk는 일반 텍스트로 추가 (마킹 없이)
                page_content_parts.append(chunk_text)
            
            # 3. 전체 페이지 내용 결합
            full_page_content = "\n\n".join(page_content_parts)
            
            # 4. apply_highlighting 함수를 사용한 하이라이트 적용
            if request.answer_text and len(hit_chunks_info) > 0:
                # Hit된 chunk의 내용을 기반으로 키워드 추출
                hit_keywords = []
                query_expander = QueryExpander()
                for hit_info in hit_chunks_info:
                    chunk_keywords = query_expander.extract_keywords(hit_info["content"])
                    hit_keywords.extend(chunk_keywords)
                
                # 중복 제거
                unique_hit_keywords = list(set(hit_keywords))
                
                # 답변 텍스트에서도 키워드 추출
                answer_keywords = query_expander.extract_keywords(request.answer_text)
                
                # 공통 키워드 찾기 (유연한 매칭으로 개선)
                common_keywords = []
                
                # 1. 정확한 일치 (기존 로직)
                exact_matches = [k for k in unique_hit_keywords if k.lower() in [ak.lower() for ak in answer_keywords]]
                common_keywords.extend(exact_matches)
                
                # 2. 부분 문자열 매칭 (새 로직 추가)
                for hit_keyword in unique_hit_keywords:
                    if hit_keyword.lower() not in [k.lower() for k in common_keywords]:  # 중복 방지
                        for answer_keyword in answer_keywords:
                            # 3글자 이상의 의미있는 부분 문자열 매칭
                            if (len(hit_keyword) >= 3 and len(answer_keyword) >= 3 and
                                (hit_keyword.lower() in answer_keyword.lower() or 
                                 answer_keyword.lower() in hit_keyword.lower())):
                                common_keywords.append(hit_keyword)
                                break
                
                # 3. Hit 키워드가 충분하지 않으면 답변 키워드도 사용
                if len(common_keywords) < 3:
                    for answer_keyword in answer_keywords[:5]:  # 상위 5개만
                        if (len(answer_keyword) >= 3 and 
                            answer_keyword.lower() not in [k.lower() for k in common_keywords]):
                            common_keywords.append(answer_keyword)
                            if len(common_keywords) >= 5:  # 최대 5개로 제한
                                break
                
                # 중복 제거 및 정리
                common_keywords = list(dict.fromkeys(common_keywords))  # 순서 유지하며 중복 제거
                
                # apply_highlighting 함수 사용
                if common_keywords:
                    full_page_content, _ = apply_highlighting(
                        content=full_page_content,
                        keywords=common_keywords,
                        original_query=request.answer_text,
                        metadata=page_metadata
                    )
            elif request.keywords:
                # 직접 제공된 키워드가 있으면 사용
                full_page_content, _ = apply_highlighting(
                    content=full_page_content,
                    keywords=request.keywords,
                    metadata=page_metadata
                )
            
            # 5. 성공 응답 반환
            return {
                "status": "success",
                "message": f"Hit chunk {len(hit_chunks_info)}개가 포함된 페이지를 성공적으로 찾았습니다.",
                "content": full_page_content,
                "keywords": common_keywords if 'common_keywords' in locals() else [],
                "hit_info": {
                    "total_chunks": len(all_chunks),
                    "hit_chunks": len(hit_chunks_info),
                    "hit_details": hit_chunks_info
                },
                "source_metadata": {
                    "filename": os.path.basename(request.path),
                    "page": request.page,
                    "target_chunk_id": target_chunk_id
                }
            }
            
        except Exception as search_error:
            print(f"Elasticsearch 검색 오류: {search_error}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"문서 검색 중 오류 발생: {str(search_error)}",
                "content": None
            }
        
    except Exception as e:
        print(f"소스 미리보기 처리 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"소스 미리보기 처리 중 오류 발생: {str(e)}", 
            "content": None
        }


@app.get("/api/indexed-files")
async def get_indexed_files():
    """Elasticsearch에 인덱싱된 고유한 파일명 목록을 반환합니다."""
    if not es_client:
        raise HTTPException(status_code=503, detail="Elasticsearch is not connected")

    try:
        # Elasticsearch Terms Aggregation 쿼리
        # source 필드의 고유한 값을 모두 가져오기 위해 size를 충분히 크게 설정
        query = {
            "size": 0,  # 실제 문서는 가져오지 않음
            "aggs": {
                "unique_sources": {
                    "terms": {
                        "field": "source",  # source 필드 기준
                        "size": 10000,  # 충분히 큰 값 (예상되는 고유 파일 수 이상)
                    }
                }
            },
        }

        response = es_client.search(index=ES_INDEX_NAME, body=query)
        # Aggregation 결과에서 파일명 추출
        buckets = (
            response.get("aggregations", {})
            .get("unique_sources", {})
            .get("buckets", [])
        )
        file_list = [
            bucket.get("key") for bucket in buckets if bucket.get("key")
        ]  # key가 파일명

        print(
            f"Indexed file list requested. Found {len(file_list)} unique files."
        )  # 로그 추가
        return {"status": "success", "files": file_list}

    except Exception as e:
        print(f"인덱싱된 파일 목록 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"파일 목록 조회 중 오류 발생: {str(e)}"
        )

# 엑셀 파일 미리보기 엔드포인트
@app.get("/preview/excel/")
async def preview_excel(file_path: str):
    """
    엑셀 파일 미리보기를 제공하는 API 엔드포인트
    - file_path: 서버상의 엑셀 파일 경로
    - 엑셀 파일(.xlsx, .xls)만 처리하며, 확장자가 맞지 않으면 400 에러 반환
    """
    # 1. 확장자 검사
    if not file_path.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="엑셀 파일(.xlsx, .xls)만 지원됩니다.")
    
    try:
        # 2. excel_to_html 함수 호출
        from app.utils.source_preview_utils import excel_to_html
        result = excel_to_html(file_path)
        
        # 3. 결과 JSON으로 반환
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"엑셀 파일 처리 중 오류 발생: {str(e)}")


# --- API 엔드포인트 추가 끝 ---


class StatsRequest(BaseModel):
    userId: str  # 사용자 식별자
    action: str  # 수행한 행동 (예: question, feedback, view_source)
    details: Dict[str, Any] = {}  # 추가 세부 정보 (예: 카테고리, 피드백 유형)


class StatsQueryRequest(BaseModel):
    userId: str = ""  # 특정 사용자 조회 (빈 문자열이면 전체 조회)
    startDate: str = ""  # 시작 날짜 (형식: YYYY-MM-DD)
    endDate: str = ""  # 종료 날짜 (형식: YYYY-MM-DD)


# 통계 저장 디렉토리 설정
STATS_DIR = "app/stats"
os.makedirs(STATS_DIR, exist_ok=True)
STATS_FILE = os.path.join(STATS_DIR, "stats.json")


# 통계 저장 함수
def save_stat(stat_data: Dict[str, Any]):
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats_list = json.load(f)
        else:
            stats_list = []
        stats_list.append(stat_data)
        with open(STATS_FILE, "w") as f:
            json.dump(stats_list, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"통계 저장 중 오류 발생: {e}")
        return False


# 통계 조회 함수
def query_stats(user_id: str = "", start_date: str = "", end_date: str = ""):
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats_list = json.load(f)
        else:
            return []

        filtered_stats = stats_list
        if user_id:
            filtered_stats = [s for s in filtered_stats if s["userId"] == user_id]
        if start_date:
            filtered_stats = [s for s in filtered_stats if s["timestamp"] >= start_date]
        if end_date:
            filtered_stats = [
                s
                for s in filtered_stats
                if s["timestamp"] <= end_date + "T23:59:59.999999"
            ]
        return filtered_stats
    except Exception as e:
        print(f"통계 조회 중 오류 발생: {e}")
        return []


# 통계 저장 엔드포인트
@app.post("/api/stats/save")
async def save_stats_endpoint(request: StatsRequest = Body(...)):
    try:
        stat_data = {
            "userId": request.userId,
            "action": request.action,
            "details": request.details,
            "timestamp": datetime.now().isoformat(),
        }
        success = save_stat(stat_data)
        if success:
            return {"status": "success", "message": "통계가 저장되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="통계 저장에 실패했습니다.")
    except Exception as e:
        print(f"통계 저장 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"통계 저장 중 오류 발생: {str(e)}")


# 통계 조회 엔드포인트
@app.post("/api/stats/query")
async def query_stats_endpoint(request: StatsQueryRequest = Body(...)):
    try:
        stats = query_stats(request.userId, request.startDate, request.endDate)
        return {"status": "success", "stats": stats}
    except Exception as e:
        print(f"통계 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류 발생: {str(e)}")


UPLOAD_DIR = Path(STATIC_DIR) / "uploads"
IMAGE_DIR_PATH = Path(STATIC_DIR) / "document_images"

@app.get("/api/image-viewer/{filename:path}")
async def get_image_for_viewer(filename: str):
    """문서 이미지를 반환합니다."""
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 이미지 경로는 app/static/document_images/ 하위
    abs_path = IMAGE_DIR_PATH / filename
    if not abs_path.is_file():
        # UUID 접두사가 있는 경우도 고려하여 탐색
        pattern = str(IMAGE_DIR_PATH / f"*_{filename}")
        matches = glob.glob(pattern)
        if not matches:
            raise HTTPException(status_code=404, detail=f"Image not found at {abs_path}")
        abs_path = Path(matches[0])

    mime, _ = mimetypes.guess_type(abs_path)
    return FileResponse(abs_path, media_type=mime or "image/jpeg")


@app.get("/api/document-images/{doc_id}")
async def get_document_images(doc_id: str):
    """특정 문서의 모든 이미지 목록을 반환합니다."""
    try:
        # 문서 ID로 이미지 디렉토리 찾기
        img_dir = IMAGE_DIR_PATH / doc_id
        
        if not img_dir.exists():
            return {"status": "success", "images": [], "message": "해당 문서의 이미지가 없습니다."}
        
        # 이미지 파일 목록 수집
        image_files = []
        for img_file in img_dir.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                # 페이지 번호 추출 (파일명에서)
                page_match = re.search(r'page_(\d+)', img_file.name)
                page_num = int(page_match.group(1)) if page_match else 1
                
                relative_path = f"document_images/{doc_id}/{img_file.name}"
                image_files.append({
                    "filename": img_file.name,
                    "path": relative_path,
                    "url": f"/api/image-viewer/{relative_path}",
                    "page": page_num,
                    "size": img_file.stat().st_size
                })
        
        # 페이지 번호로 정렬
        image_files.sort(key=lambda x: x["page"])
        
        return {
            "status": "success", 
            "images": image_files, 
            "count": len(image_files),
            "message": f"{len(image_files)}개의 이미지를 찾았습니다."
        }
        
    except Exception as e:
        logger.error(f"문서 이미지 목록 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 목록 조회 중 오류: {str(e)}")


@app.post("/api/source-images")
async def get_source_images(request: dict = Body(...)):
    """출처 문서와 연관된 이미지들을 반환합니다."""
    try:
        source_path = request.get("source_path", "")
        page = request.get("page", 1)
        
        if not source_path:
            return {"status": "error", "message": "source_path가 필요합니다.", "images": []}
        
        # 파일명에서 UUID 제거하여 문서 ID 생성
        clean_filename = strip_uuid_prefix(source_path)
        doc_id = clean_filename.rsplit('.', 1)[0] if '.' in clean_filename else clean_filename
        
        # Elasticsearch에서 해당 문서의 이미지 정보 검색
        query = {
            "size": 50,
            "_source": ["images", "image_captions", "has_images", "page"],
            "query": {
                "bool": {
                    "must": [
                        {"term": {"source": source_path}},
                        {"term": {"has_images": True}}
                    ]
                }
            }
        }
        
        response = es_client.search(index=ES_INDEX_NAME, body=query)
        hits = response.get("hits", {}).get("hits", [])
        
        all_images = []
        
        for hit in hits:
            source_data = hit["_source"]
            images = source_data.get("images", [])
            image_captions = source_data.get("image_captions", [])
            page_num = source_data.get("page", 1)
            
            # 특정 페이지만 필터링 (페이지 지정된 경우)
            if page and page_num != page:
                continue
            
            for idx, img_path in enumerate(images):
                img_info = {
                    "path": img_path,
                    "url": f"/api/image-viewer/{img_path}",
                    "page": page_num,
                    "caption": "",
                    "ocr_text": "",
                    "ai_description": ""
                }
                
                # 캡션 정보 추가
                if image_captions and idx < len(image_captions):
                    caption_info = image_captions[idx]
                    if isinstance(caption_info, dict):
                        img_info["caption"] = caption_info.get("ai_caption", "")
                        img_info["ocr_text"] = caption_info.get("ocr_text", "")
                        img_info["ai_description"] = caption_info.get("ai_caption", "")
                    elif isinstance(caption_info, str):
                        img_info["caption"] = caption_info
                
                # 기본 캡션 설정
                if not img_info["caption"]:
                    img_info["caption"] = f"페이지 {page_num} 이미지 {idx + 1}"
                
                all_images.append(img_info)
        
        # 페이지 번호로 정렬
        all_images.sort(key=lambda x: x["page"])
        
        return {
            "status": "success",
            "images": all_images,
            "count": len(all_images),
            "document_id": doc_id
        }
        
    except Exception as e:
        logger.error(f"출처 이미지 검색 중 오류: {e}")
        return {"status": "error", "message": f"이미지 검색 중 오류: {str(e)}", "images": []}


@app.get("/api/file-viewer/{filename:path}")
async def get_file_for_viewer(filename: str):
    # UUID가 포함된 전체 파일명을 사용한다고 가정
    # 보안: filename에 ../ 등이 포함되어 상위 디렉토리 접근 시도 방지
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    abs_path = UPLOAD_DIR / filename
    if not abs_path.is_file():
        # 2️⃣  실패하면 “UUID_*” 파일을 자동 탐색
        pattern = str(UPLOAD_DIR / f"*_{filename}")
        matches = glob.glob(pattern)
        if not matches:
            raise HTTPException(404, "File not found")
        abs_path = Path(matches[0])                # 첫 번째 매칭 사용

    mime, _ = mimetypes.guess_type(abs_path)
    return FileResponse(abs_path, media_type=mime or "application/octet-stream")


# 파일 삭제 엔드포인트
@app.delete("/api/delete-file")
async def delete_file(filename: str):
    """특정 파일을 Elasticsearch와 디스크에서 삭제합니다."""
    if not es_client:
        raise HTTPException(status_code=503, detail="Elasticsearch is not connected")
    
    if not filename:
        raise HTTPException(status_code=400, detail="Filename parameter is required")

    # 보안: filename에 ../ 등이 포함되어 상위 디렉토리 접근 시도 방지
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    try:
        print(f"파일 삭제 요청: {filename}")
        
        # file_manager.py의 delete_indexed_file 함수 호출
        result = delete_indexed_file(
            es_client=es_client,
            filename=filename,
            index_name=ES_INDEX_NAME,
            uploads_dir=os.path.join(STATIC_DIR, "uploads")
        )
        
        if result["status"] == "success":
            print(f"파일 삭제 성공: {filename}")
            return {"status": "success", "message": result["message"]}
        else:
            print(f"파일 삭제 실패: {filename}, 이유: {result['message']}")
            return {"status": result["status"], "message": result["message"]}
    
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"파일 삭제 중 오류가 발생했습니다: {str(e)}")


# 파일 전체 삭제 엔드포인트 추가
@app.delete("/api/delete-all-files")
async def delete_all_files():
    """
    인덱싱된 모든 파일을 삭제합니다.
    파일 시스템에서 파일을 삭제하고 Elasticsearch에서 관련 문서를 모두 제거합니다.
    """
    start_time = time.time()
    logger.info("모든 파일 삭제 요청 수신")

    es_client = get_elasticsearch_client()
    if not es_client:
        logger.error("Elasticsearch 클라이언트 초기화 실패. 모든 파일 삭제 작업을 중단합니다.")
        raise HTTPException(status_code=500, detail="Elasticsearch 연결 실패")

    uploads_dir = os.path.join(STATIC_DIR, "uploads")
    deleted_files_count = 0
    failed_files_es = []
    failed_files_fs = []

    try:
        # 1. Elasticsearch에서 모든 문서 삭제
        logger.info(f"'{ES_INDEX_NAME}' 인덱스에서 모든 문서 삭제 시작...")
        try:
            # match_all 쿼리를 사용하여 모든 문서 삭제
            delete_response = es_client.delete_by_query(
                index=ES_INDEX_NAME,
                body={
                    "query": {
                        "match_all": {}
                    }
                },
                refresh=True,  # 즉시 인덱스 갱신
                wait_for_completion=True  # 작업 완료까지 대기
            )
            
            es_deleted_count = delete_response.get("deleted", 0)
            logger.info(f"Elasticsearch에서 총 {es_deleted_count}개 문서 삭제 완료")
            
            if es_deleted_count == 0:
                logger.warning("Elasticsearch에서 삭제할 문서가 없습니다.")
        except Exception as es_error:
            logger.error(f"Elasticsearch 문서 삭제 중 오류: {es_error}")
            traceback.print_exc()
            failed_files_es.append("ALL_DOCUMENTS")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Elasticsearch 문서 삭제 중 오류: {str(es_error)}"}
            )

        # 2. 파일 시스템에서 모든 파일 삭제
        logger.info(f"파일 시스템에서 모든 파일 삭제 시작 (경로: {uploads_dir})...")
        try:
            files = os.listdir(uploads_dir)
            logger.info(f"삭제할 파일 {len(files)}개 발견")
            
            for filename in files:
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files_count += 1
                        logger.info(f"파일 삭제 성공: {filename}")
                    except Exception as file_error:
                        failed_files_fs.append(filename)
                        logger.error(f"파일 삭제 실패 ({filename}): {file_error}")
            
            logger.info(f"파일 시스템에서 총 {deleted_files_count}개 파일 삭제 완료")
        except Exception as fs_error:
            logger.error(f"파일 시스템 접근 중 오류: {fs_error}")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"파일 시스템 접근 중 오류: {str(fs_error)}"}
            )

        # 3. 결과 반환
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        
        if not failed_files_fs and not failed_files_es:
            logger.info(f"모든 파일 삭제 성공 (총 {deleted_files_count}개, 소요시간: {execution_time}초)")
            return JSONResponse(
                content={
                    "status": "success",
                    "message": f"모든 파일이 성공적으로 삭제되었습니다. (총 {deleted_files_count}개)",
                    "deleted_count": deleted_files_count,
                    "es_deleted_count": es_deleted_count,
                    "execution_time": execution_time
                }
            )
        else:
            failed_count = len(failed_files_fs)
            logger.warning(f"일부 파일 삭제 실패 (성공: {deleted_files_count}개, 실패: {failed_count}개)")
            return JSONResponse(
                content={
                    "status": "partial_success",
                    "message": f"{deleted_files_count}개 파일 삭제 성공, {failed_count}개 파일 삭제 실패",
                    "deleted_count": deleted_files_count,
                    "failed_count": failed_count,
                    "failed_files": failed_files_fs[:10],  # 최대 10개까지만 표시
                    "execution_time": execution_time
                }
            )
            
    except Exception as e:
        logger.error(f"전체 파일 삭제 중 오류 발생: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"전체 파일 삭제 중 오류 발생: {str(e)}"}
        )


# 파일 업로드 및 인덱싱 엔드포인트
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),  # 다중 파일 지원
    category: str = Form("메뉴얼"),  # 기본값을 메뉴얼로 설정
):
    results = []
    start_time = time.time()  # 전체 처리 시작 시간

    logger.info(f"파일 업로드 요청 수신: {len(files)}개 파일, 카테고리: {category}")

    # 지원하는 이미지 확장자 목록 (OCR 처리 가능)
    ocr_supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    # 메모리 정리 - 초기 상태
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"업로드 처리 시작 - 초기 GPU 메모리 사용량: {initial_memory:.2f} MB")

    for file_index, file in enumerate(files):
        logger.info(f"[{file_index+1}/{len(files)}] 파일 처리 중: {file.filename}, 카테고리: {category}")
        
        # 파일 확장자 확인
        file_extension = Path(file.filename).suffix.lower()
        is_ocr_candidate = file_extension in ocr_supported_extensions or file_extension == '.pdf'
        
        # 임시 파일 저장 - 한글 파일명 인코딩 문제 해결
        unique_id = uuid.uuid4()
        
        # 파일명 인코딩 안전하게 처리
        safe_filename = file.filename
        if safe_filename:
            try:
                # UTF-8로 인코딩된 파일명을 안전하게 디코딩
                if isinstance(safe_filename, bytes):
                    safe_filename = safe_filename.decode('utf-8', errors='replace')
                elif isinstance(safe_filename, str):
                    # 이미 문자열인 경우 Latin-1로 인코딩 후 UTF-8로 디코딩 시도
                    try:
                        safe_filename = safe_filename.encode('latin1').decode('utf-8')
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        # 디코딩 실패 시 원본 사용
                        pass
                        
                # 파일명에서 위험한 문자 제거
                safe_filename = re.sub(r'[<>:"/\\|?*]', '_', safe_filename)
                logger.info(f"원본 파일명: {file.filename} -> 안전한 파일명: {safe_filename}")
                
            except Exception as encoding_error:
                logger.warning(f"파일명 인코딩 처리 실패: {encoding_error}, 원본 파일명 사용")
                safe_filename = file.filename
        else:
            safe_filename = "unnamed_file"
            
        file_path = f"app/static/uploads/{unique_id}_{safe_filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # 파일 정보 및 초기 상태
            file_size = os.path.getsize(file_path)
            file_result = {
                "filename": file.filename,
                "unique_id": str(unique_id),
                "status": "processing",
                "message": f"파일 '{file.filename}' 처리 중...",
                "size": file_size,
                "start_time": time.time(),
                "progress": 0,  # 진행률 추가
                "file_info": {
                    "size_formatted": format_file_size(file_size),
                    "extension": file_extension,
                    "index": file_index + 1,
                    "total": len(files),
                    "ocr_supported": is_ocr_candidate  # OCR 지원 여부 표시
                }
            }

            # 파일 처리 및 인덱싱
            try:
                logger.info(f"파일 인덱싱 시작: {file.filename}")
                
                if is_ocr_candidate:
                    logger.info(f"OCR 지원 파일 감지: {file.filename} - OCR 처리가 시도될 수 있습니다.")
                
                # 파일 중복 체크
                file_exists, file_hash = check_file_exists_sync(es_client, file_path)

                # 해시값 저장
                file_result["file_hash"] = file_hash[:8] + "..." if file_hash else None

                if file_exists:
                    # 중복 파일인 경우
                    file_result.update({
                            "status": "skipped",
                            "message": f"파일 '{file.filename}'은(는) 이미 인덱싱되어 있습니다.",
                        "processing_time": round(time.time() - file_result["start_time"], 2),
                            "duplicate": True,
                        "progress": 100
                    })
                else:
                    # 새 파일 처리 - 메모리 관리 강화
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        pre_process_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                        logger.info(f"파일 처리 전 GPU 메모리: {pre_process_memory:.2f} MB")

                    # 파일 처리 진행률 업데이트 (실제로는 비동기 처리가 필요할 수 있음)
                    file_result["progress"] = 30
                    
                    processing_start = time.time()
                    if is_ocr_candidate:
                        logger.info(f"OCR 처리 시작: {file.filename}")
                    
                    success = process_and_index_file(
                    es_client=es_client,
                    embedding_function=embedding_function,
                    uploaded_file_path=file_path,
                    category=category,
                    shared_llm_model=llm_model,
                    shared_tokenizer=tokenizer,
                    reindex=False  # 새 업로드는 해시 접두어 붙여서 저장
                )
                    
                    processing_time = time.time() - processing_start
                    logger.info(f"파일 처리 소요 시간: {processing_time:.2f}초")

                    # 메모리 사용량 확인
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        post_process_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                        memory_used = post_process_memory - pre_process_memory
                        logger.info(f"파일 처리 후 GPU 메모리: {post_process_memory:.2f} MB (변화: {memory_used:.2f} MB)")

                    if success:
                        logger.info(f"파일 인덱싱 성공: {file.filename}")
                        # 성공 메시지에 OCR 정보 포함
                        if is_ocr_candidate:
                            success_message = f"파일 '{file.filename}' 인덱싱 완료 (OCR 처리 적용)"
                        else:
                            success_message = f"파일 '{file.filename}' 인덱싱 완료"
                            
                        file_result.update({
                                "status": "success",
                                "message": success_message,
                            "processing_time": round(time.time() - file_result["start_time"], 2),
                            "ocr_processed": is_ocr_candidate,
                            "progress": 100
                        })
                    else:
                        logger.error(f"파일 인덱싱 실패: {file.filename}")
                        file_result.update({
                                "status": "error",
                                "message": "파일 인덱싱 실패",
                            "processing_time": round(time.time() - file_result["start_time"], 2),
                            "progress": 100
                        })
            except Exception as e:
                logger.error(f"파일 처리 중 오류 발생: {file.filename}, 오류: {str(e)}")
                traceback.print_exc()
                file_result.update({
                        "status": "error",
                        "message": f"오류 발생: {str(e)}",
                    "processing_time": round(time.time() - file_result["start_time"], 2),
                        "error_details": str(e),
                    "progress": 100
                })

            results.append(file_result)
        except Exception as e:
            logger.error(f"파일 저장 중 오류 발생: {file.filename}, 오류: {str(e)}")
            traceback.print_exc()
            results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"파일 저장 중 오류 발생: {str(e)}",
                    "error_details": str(e),
                "progress": 100
            })

    # 최종 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"전체 처리 완료 - 최종 GPU 메모리 사용량: {final_memory:.2f} MB")

    # 전체 요약 통계 추가
    total_time = round(time.time() - start_time, 2)
    ocr_files_count = sum(1 for r in results if r.get("file_info", {}).get("ocr_supported", False))
    ocr_processed_count = sum(1 for r in results if r.get("ocr_processed", False) and r["status"] == "success")
    
    summary = {
        "total_files": len(results),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
        "skipped_count": sum(1 for r in results if r["status"] == "skipped"),
        "ocr_supported_count": ocr_files_count,
        "ocr_processed_count": ocr_processed_count,
        "total_size": sum(r["size"] for r in results),
        "total_size_formatted": format_file_size(sum(r["size"] for r in results)),
        "total_processing_time": total_time,
        "average_file_time": round(total_time / len(results), 2) if results else 0
    }

    return JSONResponse(
        content={
            "status": "success" if summary["error_count"] == 0 else "partial_success",
            "message": f"{summary['total_files']}개 파일 처리 완료. {summary['success_count']}개 성공, {summary['error_count']}개 실패, {summary['skipped_count']}개 건너뜀 (OCR 처리: {summary['ocr_processed_count']}개)",
            "results": results,
            "summary": summary,
        }
    )

@app.post("/api/chat")
async def chat(fastapi_request: FastAPIRequest, request: QuestionRequest = Body(...)):
    logger.info(f"Received chat request: '{request.question}', Category: '{request.category}', History items: {len(request.history) if request.history else 0}")
    request_start_time = time.time()

    # 의존성 확인
    global es_client, embedding_function, reranker_model, llm_model, tokenizer

    if not all([es_client, embedding_function, reranker_model, llm_model, tokenizer]):
        logger.error("Critical components (ES, models, tokenizer) not initialized.")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "챗봇 시스템이 준비되지 않았습니다. 관리자에게 문의하세요."}
        )

    try:
        result = await search_and_combine(
            es_client=es_client,
            embedding_function=embedding_function,
            reranker_model=reranker_model,
            llm_model=llm_model,
            tokenizer=tokenizer,
            query=request.question,
            category=request.category,
            conversation_history=request.history
        )
        
        # 🚀 result가 StreamingResponse인 경우 그대로 반환 (캐시된 경우)
        if isinstance(result, StreamingResponse):
            logger.info("Returning cached streaming response")
            return result
        
        # 📝 일반 응답인 경우 스트리밍으로 변환
        logger.info(f"Converting result to streaming response. Processing time: {result.get('processing_time', {}).get('total', 0)}s")
        
        async def result_to_stream():
            answer = result.get("answer", "응답을 생성할 수 없습니다.")
            sources = result.get("sources", [])
            cited_sources = result.get("cited_sources", [])
            
            # 텍스트를 청크로 나누어 스트리밍
            chunk_size = 5  # 5자씩
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                yield f"data: {json.dumps({'token': chunk})}\n\n"
                await asyncio.sleep(0.01)  # 스트리밍 효과
            
            # 소스 정보 전송
            yield f"data: {json.dumps({'event': 'sources', 'sources': sources, 'cited_sources': cited_sources})}\n\n"
            
            # 처리 시간 정보 (선택적)
            if result.get("from_cache"):
                yield f"data: {json.dumps({'event': 'cache_info', 'from_cache': True})}\n\n"
            
            # 스트림 종료
            yield f"data: {json.dumps({'event': 'eos', 'message': 'Stream ended successfully.'})}\n\n"
            
            total_time = time.time() - request_start_time
            logger.info(f"Total request processing time: {total_time:.4f} seconds")
        
        # StreamingResponse 반환
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        
        return StreamingResponse(result_to_stream(), media_type="text/event-stream", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in chat endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "챗봇 응답 처리 중 심각한 오류가 발생했습니다.", "error_details": str(e)}
        )


# 카테고리 목록 조회 엔드포인트
@app.get("/api/categories")
async def get_categories():
    try:
        query = {
            "size": 0,
            "aggs": {"categories": {"terms": {"field": "category", "size": 100}}},
        }

        result = es_client.search(index=ES_INDEX_NAME, body=query)
        categories = [
            bucket["key"] for bucket in result["aggregations"]["categories"]["buckets"]
        ]

        # 카테고리가 없으면 기본값 추가
        if not categories:
            categories = ["메뉴얼", "장애보고서"]

        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"카테고리 조회 실패: {str(e)}")


# 대화 제목 자동 생성을 위한 모델 추가
class GenerateTitleRequest(BaseModel):
    messages: List[Dict[str, Any]]  # 대화 메시지 목록

# 대화 제목 생성 API 엔드포인트
@app.post("/api/generate-title")
async def generate_title_endpoint(request: GenerateTitleRequest = Body(...)):
    try:
        if not request.messages or len(request.messages) == 0:
            return {"title": "새 대화"}

        # 첫 번째 사용자 메시지 추출
        user_messages = [msg for msg in request.messages if msg.get("role") == "user"]
        if not user_messages:
            return {"title": "새 대화"}

        first_user_message = user_messages[0].get("content", "")
        if not first_user_message:
            return {"title": "새 대화"}

        # 간단한 제목 생성 로직 (LLM 사용 없이)
        title = first_user_message[:20]  # 첫 20자 추출

        # 마침표, 물음표, 느낌표로 끝나는 경우 처리
        punctuation_marks = [".", "?", "!", ",", ";", ":", "...", "…"]
        for mark in punctuation_marks:
            if title.endswith(mark):
                title = title[:-len(mark)]
                break

        # 조사로 끝나는 경우 처리
        korean_particles = ["이", "가", "을", "를", "은", "는", "에", "의", "로", "와", "과"]
        for particle in korean_particles:
            if title.endswith(particle):
                title = title[:-len(particle)]
                break

        # 너무 짧은 경우 적당한 접미사 추가
        if len(title) < 5:
            title += "에 대한 대화"
        elif "?" in first_user_message or "질문" in first_user_message:
            title += "에 대한 질문"

        # 50자 이상이면 줄임
        if len(title) > 50:
            title = title[:47] + "..."

        return {"title": title.strip()}
    except Exception as e:
        print(f"대화 제목 생성 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"대화 제목 생성 중 오류 발생: {str(e)}")


# 기본 라우트
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API 서버가 실행 중입니다."}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    if exc.status_code == 405:
        print(
            f"405 Method Not Allowed: {request.method} 요청이 {request.url}로 수신됨",
            flush=True,
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# 피드백 저장 엔드포인트
@app.post("/api/feedback")
async def save_feedback_endpoint(request: FeedbackRequest = Body(...)):
    try:
        feedback_data = {
            "messageId": request.messageId,
            "feedbackType": request.feedbackType,
            "rating": request.rating,
            "content": request.content,
            "timestamp": datetime.now().isoformat(),
        }
        success = save_feedback(feedback_data)
        if success:
            return {"status": "success", "message": "피드백이 저장되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="피드백 저장에 실패했습니다.")
    except Exception as e:
        print(f"피드백 저장 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"피드백 저장 중 오류 발생: {str(e)}"
        )


# 피드백 분석 및 검색 품질 개선 API 추가
@app.get("/api/feedback/stats")
async def get_feedback_statistics():
    """
    피드백 데이터 통계를 반환하는 API 엔드포인트
    """
    try:
        analyzer = FeedbackAnalyzer()
        stats = analyzer.get_feedback_stats()

        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        print(f"피드백 통계 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"피드백 통계 조회 중 오류 발생: {str(e)}"
        )

@app.get("/api/feedback/trends")
async def get_feedback_trends(days: int = 30):
    """
    최근 N일간의 피드백 추세를 분석하는 API 엔드포인트
    """
    try:
        analyzer = FeedbackAnalyzer()
        trends = analyzer.analyze_feedback_trends(days=days)

        return {
            "status": "success",
            "trends": trends
        }
    except Exception as e:
        print(f"피드백 추세 분석 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"피드백 추세 분석 중 오류 발생: {str(e)}"
        )

@app.get("/api/feedback/documents")
async def get_document_quality_scores():
    """
    피드백을 기반으로 한 문서 품질 점수를 반환하는 API 엔드포인트
    """
    try:
        analyzer = FeedbackAnalyzer()
        scores = analyzer.calculate_document_quality_scores()

        # 단순 출력용으로 점수 정렬
        sorted_scores = sorted(
            [{"document": doc, "score": score} for doc, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "status": "success",
            "document_scores": sorted_scores,
            "count": len(sorted_scores)
        }
    except Exception as e:
        print(f"문서 품질 점수 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"문서 품질 점수 조회 중 오류 발생: {str(e)}"
        )

@app.get("/api/feedback/frequently-asked")
async def get_frequently_asked_questions(min_count: int = 3):
    """
    자주 묻는 질문 패턴을 반환하는 API 엔드포인트
    """
    try:
        analyzer = FeedbackAnalyzer()
        questions = analyzer.extract_frequent_questions(min_count=min_count)

        return {
            "status": "success",
            "questions": questions,
            "count": len(questions)
        }
    except Exception as e:
        print(f"자주 묻는 질문 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"자주 묻는 질문 조회 중 오류 발생: {str(e)}"
        )

# SQL 관련 모델 정의
class SQLQueryRequest(BaseModel):
    question: str
    
class SQLAndLLMRequest(BaseModel):
    question: str

# SQL 관련 엔드포인트 추가
@app.get("/api/db-schema")
async def get_db_schema():
    """데이터베이스 스키마 정보를 반환합니다."""
    try:
        from utils.get_mariadb_schema import get_schema_for_sqlcoder, test_db_connection
        
        # 먼저 DB 연결 테스트
        if not test_db_connection():
            print("DB 연결 테스트 실패")
            return {
                "status": "error", 
                "schema": "# 데이터베이스 연결 오류\n\n데이터베이스에 연결할 수 없습니다. 관리자에게 문의하세요."
            }
        
        # SQLCoder 형식으로 스키마 가져오기
        schema = get_schema_for_sqlcoder()
        
        # 오류 메시지가 반환된 경우 (ERROR로 시작하는 문자열)
        if isinstance(schema, str) and schema.startswith("ERROR:"):
            print(f"DB 스키마 조회 오류: {schema}")
            # 오류가 발생했지만 200 OK와 함께 오류 메시지 전달
            return {
                "status": "error", 
                "schema": "# 데이터베이스 연결 오류\n\n데이터베이스에 연결할 수 없습니다. 관리자에게 문의하세요.",
                "error": schema
            }
            
        return {"status": "success", "schema": schema}
    except Exception as e:
        print(f"DB 스키마 조회 중 오류 발생: {str(e)}")
        traceback.print_exc()
        # 500 에러가 아닌 200 OK 응답으로 변경하여 클라이언트 오류 처리 개선
        return {
            "status": "error",
            "schema": "# 데이터베이스 스키마 로딩 오류\n\n시스템 오류로 스키마를 불러올 수 없습니다. 관리자에게 문의하세요.",
            "error": str(e)
        }

@app.post("/api/sql-query")
async def process_sql_query(request: SQLQueryRequest = Body(...)):
    """자연어 질문을 SQL로 변환하고 실행 결과를 반환합니다."""
    try:
        # SQLCoder 유틸 사용
        from utils.sqlcoder_utils import generate_sql_from_question, run_sql_query
        
        # SQL 생성
        print(f"자연어 질문: {request.question}")
        sql = generate_sql_from_question(request.question)
        
        if not sql:
            return {
                "sql": "SELECT 1;",
                "results": "⚠️ SQL을 생성할 수 없습니다. 질문을 더 구체적으로 작성해주세요."
            }
        
        # SQL 실행
        results = run_sql_query(sql)
        
        # 결과가 에러인 경우
        if isinstance(results, dict) and "error" in results:
            error_msg = results["error"]
            return {
                "sql": sql,
                "results": f"❌ SQL 실행 오류: {error_msg}"
            }
        
        # 결과가 문자열인 경우 (이미 포맷팅된 메시지일 수 있음)
        if isinstance(results, str):
            return {
                "sql": sql,
                "results": results
            }
        
        # 결과가 비어있는 경우
        if not results or len(results) == 0:
            return {
                "sql": sql,
                "results": "⚠️ 쿼리 결과가 없습니다."
            }
        
        # 테이블 형태로 결과 변환
        try:
            # 결과를 마크다운 테이블로 변환
            headers = list(results[0].keys())
            markdown_table = "| " + " | ".join(headers) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            for row in results:
                row_values = []
                for header in headers:
                    value = row[header]
                    # None 값 처리
                    if value is None:
                        value = "NULL"
                    # 긴 문자열 처리
                    elif isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    row_values.append(str(value))
                markdown_table += "| " + " | ".join(row_values) + " |\n"
                
            return {
                "sql": sql,
                "results": markdown_table
            }
        except Exception as e:
            print(f"결과 포맷팅 오류: {str(e)}")
            # 결과를 문자열로 변환하여 안전하게 반환
            return {
                "sql": sql,
                "results": f"결과 처리 중 오류가 발생했습니다: {str(e)}\n\n원본 결과: {str(results)}"
            }
            
    except Exception as e:
        print(f"SQL 쿼리 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SQL 쿼리 처리 중 오류 발생: {str(e)}")

@app.post("/api/sql-and-llm")
async def process_sql_and_llm(request: SQLAndLLMRequest = Body(...)):
    """자연어 질문을 SQL로 변환 실행하고, LLM으로 설명을 추가합니다. 스트리밍 방식으로 응답합니다."""
    try:
        # SQLCoder 유틸 사용
        from utils.sqlcoder_utils import generate_sql_from_question, run_sql_query
        
        # SQL 생성 및 실행
        sql_query = generate_sql_from_question(request.question)
        
        # 쿼리가 오류를 포함하고 있는 경우
        if sql_query.startswith("-- SQL 생성 중 오류 발생"):
            return JSONResponse(content={
                "status": "error",
                "sql": sql_query,
                "results": "❌ SQL 생성에 실패했습니다",
                "explanation": f"SQL 생성 중 오류가 발생했습니다: {sql_query}"
            })
        
        # SQL 쿼리 실행 (run_sql_query가 SQL 및 마크다운 형식 결과 반환)
        sql_result = run_sql_query(sql_query)
        
        # SQL 실행 중 오류가 발생했는지 확인
        has_error = isinstance(sql_result, dict) and sql_result.get('error') is not None
        is_empty = False  # 결과가 비어있는지 여부
        
        if has_error:
            # 오류 메시지 가져오기
            error_message = sql_result.get('error', "알 수 없는 오류")
            results_markdown = f"❌ SQL 실행 오류: {error_message}"
            
            # SQL 오류가 복잡한 경우 단순화된 메시지 생성
            explanation = f"SQL 쿼리 실행 중 오류가 발생했습니다. 쿼리 문법이나 존재하지 않는 테이블/컬럼을 참조했을 수 있습니다."
            
            return JSONResponse(content={
                "status": "error",
                "sql": sql_query,
                "results": results_markdown,
                "explanation": explanation
            })
        elif isinstance(sql_result, str) and "레코드를 찾을 수 없습니다" in sql_result:
            # 결과가 없는 경우
            results_markdown = "⚠️ 조건에 맞는 데이터가 없습니다."
            is_empty = True
        else:
            # 정상 실행 결과인 경우 마크다운 테이블 생성
            results_markdown = sql_result
        
        print(f"[SQL+LLM] LLM 모델로 향상된 응답 생성 시작 - 질문: '{request.question}'")
        
        # LLM을 통한 설명 생성을 스트리밍 방식으로 변경
        # 프롬프트 구성
        explanation_prompt = f"""# SQL 쿼리 결과 분석 및 응답 생성
## 질문
{request.question}

## SQL 쿼리
```sql
{sql_query}
```

## 쿼리 결과
```
{results_markdown}
```

## 지시사항
1. 위 SQL 쿼리와 결과를 바탕으로 질문에 대한 답변을 생성해주세요.
2. 간결하게 요점만 설명하되, 핵심 내용을 누락하지 마세요.
3. 가능하다면 데이터의 특징이나 패턴을 언급하세요.
4. 수치가 있다면 중요한 수치를 언급하고 그 의미를 설명하세요.
5. 질문에 직접적인 답변을 하는 형식으로 작성하세요.
6. SQL 코드나 기술적 설명은 포함하지 마세요.
7. 결과가 비어있다면 그 이유와 가능한 원인을 설명하세요.
"""

        print(f"[SQL+LLM] 향상된 프롬프트 길이: {len(explanation_prompt)} 문자")
        
        # TextIteratorStreamer 설정
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 모델 입력 준비
        try:
            inputs = tokenizer(explanation_prompt, return_tensors="pt", padding=False, truncation=False).to(llm_model.device)
        except Exception as e:
            print(f"[SQL+LLM] 프롬프트 토큰화 오류: {e}")
            return JSONResponse(content={
                "status": "error",
                "sql": sql_query,
                "results": results_markdown,
                "explanation": f"설명 생성을 위한 프롬프트 처리 중 오류 발생: {str(e)}"
            })
        
        # 생성 파라미터 설정
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            do_sample=False  # 결정적 생성
        )
        
        # 별도 스레드에서 모델 생성 실행
        thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 스트리밍 응답을 위한 변수
        accumulated_text = ""
        
        # 비동기 제너레이터 정의
        async def stream_generator():
            nonlocal accumulated_text
            
            # 먼저 SQL 및 결과 정보 전송
            yield f"data: {json.dumps({'event': 'sql', 'sql': sql_query, 'results': results_markdown})}\n\n"
            
            # LLM 응답 스트리밍
            for new_text in streamer:
                if new_text:
                    accumulated_text += new_text
                    yield f"data: {json.dumps({'token': new_text})}\n\n"
                await asyncio.sleep(0.001)  # 다른 비동기 작업 실행 기회 부여
            
            # 중복 제거 로직 적용
            try:
                cleaned_text = deduplicate_markdown_sections_py(accumulated_text)
                
                # 응답이 비었거나 너무 짧은 경우 등의 후처리
                if not cleaned_text.strip() or len(cleaned_text.strip()) < 10 or "오류" in cleaned_text:
                    if is_empty:
                        cleaned_text = "조건에 맞는 데이터가 없습니다. 검색 조건을 변경해 보시거나, 다른 질문을 시도해보세요."
                    elif "오류" in accumulated_text and not cleaned_text.strip():
                        cleaned_text = accumulated_text  # 원본 오류 메시지 사용
                    else:
                        cleaned_text = "SQL 쿼리 결과를 기반으로 한 설명입니다. 위 테이블에서 자세한 정보를 확인하세요."
                
                # 최종 정리된 응답 전송 (필요시)
                if cleaned_text != accumulated_text:
                    yield f"data: {json.dumps({'event': 'cleaned_explanation', 'explanation': cleaned_text})}\n\n"
            except Exception as clean_error:
                print(f"[SQL+LLM] 응답 정리 중 오류: {clean_error}")
                # 오류 시 원본 텍스트 사용
            
            # 스트림 종료 알림
            yield f"data: {json.dumps({'event': 'eos'})}\n\n"
            
            # 스레드 종료 대기
            if thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print("[SQL+LLM] 생성 스레드가 정상적으로 종료되지 않았습니다.")
        
        # StreamingResponse 반환
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=headers)
        
    except Exception as e:
        print(f"SQL+LLM 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "sql": "-- SQL 생성 실패",
            "results": "❌ 오류 발생",
            "explanation": f"처리 중 오류가 발생했습니다: {str(e)}"
        })

# 소스 미리보기에 필요한 유틸리티 함수
def extract_keywords_from_text(text, max_keywords=12):
    """텍스트에서 주요 키워드 추출"""
    try:
        # 한글, 영문, 숫자 단어 추출 (특수문자 및 공백 기준 분리)
        words = re.findall(r'[가-힣a-zA-Z0-9]{2,}', text)
        
        # 불용어 제거 (한국어 기준)
        stopwords = {
            '이', '그', '저', '것', '수', '등', '및', '에서', '에게', '으로', '로', '을', '를',
            '이다', '있다', '하다', '이런', '저런', '그런', '어떤', '무슨', '어떻게', '왜',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of',
            '있는', '없는', '경우', '때문', '위해', '통해', '따라', '의해', '의한', '때는',
            '있습니다', '없습니다', '합니다', '입니다', '됩니다', '관련', '때문에', '위하여',
            '만약', '그러나', '하지만', '또한', '그리고', '따라서', '이러한', '그러한','다른','그럼', 
            '이것', '그것', '저것', '무엇', '어디', '언제', '누구', '받을', '있니', '는', '?', '!'
        }
        
        # 중복 제거 및 단어 개수 세기
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            if len(word) >= 2 and word_lower not in stopwords:
                # 특수 가중치 적용 - 특정 길이 범위의 단어에 가중치 부여
                weight = 1.0
                if 3 <= len(word) <= 8:  # 보통 의미있는 단어 길이 범위
                    weight = 1.5
                if word[0].isupper() and len(word) > 1 and not word.isupper():  # 대문자로 시작하는 단어 (고유명사 가능성)
                    weight = 2.0
                
                word_counts[word] = word_counts.get(word, 0) + weight
        
        # 빈도수 기준 상위 키워드 추출 (가중치 적용)
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 중복 단어 제거 (소문자화하여 비교)
        unique_keywords = []
        added_lower = set()
        
        for kw, _ in keywords:
            kw_lower = kw.lower()
            if kw_lower not in added_lower and len(kw) >= 2:
                unique_keywords.append(kw)
                added_lower.add(kw_lower)
            
            if len(unique_keywords) >= max_keywords:
                break
        
        return unique_keywords
    except Exception as e:
        print(f"키워드 추출 중 오류: {e}")
        return []

def format_content(content):
    """내용 서식 개선"""
    if not content:
        return ""
    
    # 여러 개의 연속 공백을 하나로 압축
    formatted = re.sub(r'\s+', ' ', content).strip()
    
    # 문장 구분자 후 개행 추가 (문단 구분)
    formatted = re.sub(r'([.!?])\s+', r'\1\n\n', formatted)
    
    # 중복 개행 제거 (3개 이상 → 2개)
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    
    return formatted

def highlight_keywords(content, keywords, answer_text=None):
    """내용에서 문장 단위 하이라이트 - 답변과 일치하는 문장 강조"""
    if not content:
        return content, []
        
    # 최종 결과와 관련 문단 저장 변수
    highlighted_paragraphs = []
    relevant_paragraphs = []
    
    # 답변 텍스트가 있으면 문장 단위 매칭 적용
    if answer_text and isinstance(answer_text, str) and len(answer_text.strip()) > 10:
        print(f"응답 텍스트 기반 하이라이트 시작 - 응답 길이: {len(answer_text)}")
        
        # 1. 콘텐츠를 문단으로 분리
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', content)
        
        # 2. 응답 텍스트에서 핵심 키워드 추출
        answer_keywords = extract_keywords_from_text(answer_text, max_keywords=15)
        
        # 3. 각 문단의 관련성 점수 계산 및 하이라이트 적용
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph or len(paragraph) < 15:
                continue
            
            # 문단에서 키워드 추출
            paragraph_keywords = extract_keywords_from_text(paragraph, max_keywords=10)
            
            # 키워드 일치 점수 계산
            matching_keywords = [k for k in paragraph_keywords if k.lower() in [ak.lower() for ak in answer_keywords]]
            keyword_score = len(matching_keywords) / max(1, len(paragraph_keywords))
            
            # 직접 텍스트 매칭 검사 (정확한 인용구 확인)
            direct_match = False
            if len(paragraph) > 30:
                for i in range(len(paragraph) - 30):
                    snippet = paragraph[i:i+30]
                    if snippet in answer_text:
                        direct_match = True
                        break
            
            # 관련성 높은 문단 선택 (키워드 일치 또는 직접 매칭)
            is_relevant = keyword_score > 0.5 or direct_match
            
            # 하이라이트 적용 및 관련 문단 표시
            if is_relevant:
                # 키워드 볼드 처리와 함께 문단 전체에 배경색 적용
                highlighted_paragraph = paragraph
                for keyword in matching_keywords:
                    try:
                        # 정규식 특수문자 이스케이프
                        escaped_keyword = re.escape(keyword)
                        # 키워드에 볼드체 적용
                        highlighted_paragraph = re.sub(
                            f'\\b{escaped_keyword}\\b', 
                            f'**{keyword}**', 
                            highlighted_paragraph, 
                            flags=re.IGNORECASE
                        )
                    except Exception as e:
                        print(f"하이라이트 오류: {e}")
                
                highlighted_paragraphs.append(highlighted_paragraph)
                relevant_paragraphs.append({
                    'text': paragraph,
                    'relevance': 'high' if direct_match else 'medium',
                    'matching_keywords': matching_keywords
                })
            else:
                # 관련성 낮은 문단은 그대로 추가
                highlighted_paragraphs.append(paragraph)
        
        # 하이라이트된 내용을 다시 결합
        highlighted_content = "\n\n".join(highlighted_paragraphs)
        
        # 하이라이트에 사용된 키워드 목록 (중복 제거)
        all_matching_keywords = []
        for para in relevant_paragraphs:
            all_matching_keywords.extend(para.get('matching_keywords', []))
        unique_keywords = list(set(all_matching_keywords))
        
        return highlighted_content, unique_keywords
    
    # 답변 텍스트가 없는 경우, 기본 키워드 하이라이트만 적용
    elif keywords and isinstance(keywords, list):
        for keyword in keywords:
            try:
                # 정규식 특수문자 이스케이프
                escaped_keyword = re.escape(keyword)
                # 키워드에 볼드체 적용
                content = re.sub(
                    f'\\b{escaped_keyword}\\b', 
                    f'**{keyword}**', 
                    content, 
                    flags=re.IGNORECASE
                )
            except Exception as e:
                print(f"하이라이트 오류: {e}")
        return content, keywords
    
    # 키워드나 답변 텍스트가 없을 경우 원본 콘텐츠 반환
    return content, []

# 상수 선언
ES_INDEX_NAME = "rag_documents_kure_v1"

# 직접 실행을 위한 코드 추가
if __name__ == "__main__":
    import uvicorn
    print("Qwen2.5-7B 모델로 서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 기타 라우터 등록
from stats.dashboard_api import router as dashboard_router
from api.reindex import router as reindex_router
from app.api.synonyms import router as synonyms_router
from app.api.source_images import router as source_images_router
app.include_router(dashboard_router, prefix="", tags=["dashboard"])
app.include_router(reindex_router, prefix="/api/reindex", tags=["reindex"])
app.include_router(synonyms_router, prefix="", tags=["synonyms"])
app.include_router(source_images_router, prefix="/api", tags=["source_images"])

# 개선된 중복 제거 함수 (Python 버전) - 마크다운 섹션 기반 구조적 중복 제거
def deduplicate_markdown_sections_py(markdown_text: str) -> str:
    if not markdown_text or not isinstance(markdown_text, str):
        return ""
    
    # 1. 마크다운 헤더(##, ### 등)로 섹션 분리
    section_pattern = re.compile(r'(^|\n)(#+ [^\n]+)(?=\n)', re.MULTILINE)
    sections = []
    last_idx = 0
    for match in section_pattern.finditer(markdown_text):
        start = match.start(2)
        if last_idx < start:
            # 헤더 없는 앞부분 블록
            block = markdown_text[last_idx:match.start(1)].strip()
            if block:
                sections.append((None, block))
        header = match.group(2).strip()
        last_idx = start
        # 다음 헤더 전까지가 본문
    # 마지막 헤더 이후 블록
    if last_idx < len(markdown_text):
        block = markdown_text[last_idx:].strip()
        if block:
            # 헤더 추출
            header_match = re.match(r'^(#+ [^\n]+)\n', block)
            if header_match:
                header = header_match.group(1).strip()
                body = block[len(header):].strip()
                sections.append((header, body))
            else:
                sections.append((None, block))

    # 2. 블록별로 중복(유사) 여부 판단
    def is_similar(a, b, threshold=0.8):
        # difflib의 SequenceMatcher로 유사도 측정
        return difflib.SequenceMatcher(None, a, b).ratio() >= threshold

    unique_blocks = []
    seen_bodies = []
    for header, body in sections:
        # 질문 섹션은 무조건 스킵
        if header and ("질문" in header or "question" in header.lower()):
            continue
        # 본문 정규화
        norm_body = re.sub(r'\s+', ' ', body.strip().lower())
        # 이미 유사한 본문이 있는지 확인
        is_dup = False
        for prev_body in seen_bodies:
            if is_similar(norm_body, prev_body):
                is_dup = True
                break
        if not is_dup:
            seen_bodies.append(norm_body)
            unique_blocks.append((header, body))

    # 3. 최종 조합 (헤더 없는 블록은 그대로, 헤더 있는 블록은 헤더+본문)
    result_lines = []
    for header, body in unique_blocks:
        if header:
            result_lines.append(header)
        result_lines.append(body)
    result = '\n\n'.join(result_lines)
    # 연속 빈 줄 정리
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()

@app.get("/api/test-reindex")
async def test_reindex():
    return {"status": "success", "message": "Reindex API works!"}

@app.post("/api/quick-reindex")
async def quick_reindex():
    from api.reindex import reindex_all_files
    return reindex_all_files()

# 중복 UUID 파일명 정리 엔드포인트
@app.post("/api/clean-duplicate-uuid")
async def clean_duplicate_uuid():
    """중복 UUID가 붙은 잘못된 파일명들을 정리합니다."""
    start_time = time.time()
    logger.info("중복 UUID 파일명 정리 작업 시작")
    
    # 정리할 디렉토리들
    directories_to_clean = [
        os.path.join(STATIC_DIR, "uploads"),
        "backend/temp_conversions"
    ]
    
    # 중복 UUID 패턴: uuid1_uuid2_파일명.확장자
    duplicate_uuid_pattern = re.compile(r'^([a-f0-9]{8,12})_([a-f0-9-]{8,36})_(.+)$', re.IGNORECASE)
    
    cleaned_files = []
    errors = []
    total_files_checked = 0
    
    try:
        for directory in directories_to_clean:
            if not os.path.exists(directory):
                logger.info(f"디렉토리 존재하지 않음: {directory}")
                continue
                
            logger.info(f"디렉토리 스캔 중: {directory}")
            files = os.listdir(directory)
            total_files_checked += len(files)
            
            for filename in files:
                file_path = os.path.join(directory, filename)
                
                # 파일인지 확인
                if not os.path.isfile(file_path):
                    continue
                
                # 중복 UUID 패턴 매칭
                match = duplicate_uuid_pattern.match(filename)
                if match:
                    uuid1, uuid2, clean_filename = match.groups()
                    
                    # 새 파일명: 첫 번째 UUID만 유지
                    new_filename = f"{uuid1}_{clean_filename}"
                    new_file_path = os.path.join(directory, new_filename)
                    
                    # 파일명 변경
                    try:
                        # 동일한 이름의 파일이 이미 존재하는지 확인
                        if os.path.exists(new_file_path):
                            logger.warning(f"타겟 파일이 이미 존재함: {new_filename}")
                            # 기존 파일 삭제 (중복 파일이므로)
                            os.remove(file_path)
                            cleaned_files.append({
                                "original": filename,
                                "action": "deleted_duplicate",
                                "reason": f"동일한 파일({new_filename})이 이미 존재하여 중복 파일 삭제"
                            })
                        else:
                            # 파일명 변경
                            os.rename(file_path, new_file_path)
                            cleaned_files.append({
                                "original": filename,
                                "new": new_filename,
                                "action": "renamed",
                                "removed_uuid": uuid2
                            })
                        
                        logger.info(f"파일 정리 완료: {filename}")
                        
                    except Exception as e:
                        error_msg = f"파일 정리 실패 ({filename}): {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
        
        # 결과 반환
        execution_time = round(time.time() - start_time, 2)
        
        if cleaned_files:
            logger.info(f"중복 UUID 정리 완료: {len(cleaned_files)}개 파일 처리")
            return {
                "status": "success",
                "message": f"{len(cleaned_files)}개 파일의 중복 UUID를 정리했습니다.",
                "cleaned_files": cleaned_files,
                "errors": errors,
                "stats": {
                    "total_files_checked": total_files_checked,
                    "cleaned_count": len(cleaned_files),
                    "error_count": len(errors),
                    "execution_time": execution_time
                }
            }
        else:
            logger.info("중복 UUID가 붙은 파일이 없습니다.")
            return {
                "status": "success",
                "message": "중복 UUID가 붙은 파일이 없습니다. 모든 파일명이 정상입니다.",
                "cleaned_files": [],
                "errors": errors,
                "stats": {
                    "total_files_checked": total_files_checked,
                    "cleaned_count": 0,
                    "error_count": len(errors),
                    "execution_time": execution_time
                }
            }
            
    except Exception as e:
        logger.error(f"중복 UUID 정리 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"중복 UUID 정리 중 오류 발생: {str(e)}",
            "cleaned_files": cleaned_files,
            "errors": errors + [str(e)]
        }


def verify_and_fix_citations(answer: str, provided_docs: List[Document]) -> str:
    """
    LLM 응답에서 잘못된 출처 인용을 감지하고 수정합니다.
    할루시네이션 방지를 위한 핵심 함수입니다.
    """
    if not answer or not provided_docs:
        return answer
    
    import re
    from app.utils.indexing_utils import strip_uuid_prefix
    
    # 제공된 문서들의 파일명 목록 구성
    provided_files = {}
    for doc in provided_docs:
        source_path = doc.metadata.get("source", "")
        clean_filename = strip_uuid_prefix(source_path)
        page_num = doc.metadata.get("page", 1)
        
        key = f"{clean_filename} p.{page_num}" if page_num > 1 else clean_filename
        provided_files[key] = doc
    
    # 답변에서 인용된 파일명들 추출
    citation_pattern = r'\[([^[\]]+?)\]'
    cited_files = re.findall(citation_pattern, answer)
    
    print(f"🔍 출처 검증 - 제공된 문서: {list(provided_files.keys())}")
    print(f"🔍 출처 검증 - 인용된 문서: {cited_files}")
    
    # 잘못된 인용 감지 및 수정
    corrected_answer = answer
    
    for cited_file in cited_files:
        # 정확한 매칭 확인
        if cited_file not in provided_files:
            print(f"⚠️ 잘못된 출처 인용 감지: [{cited_file}]")
            
            # 가장 관련성 높은 문서로 교체
            best_match = None
            best_score = 0
            
            for provided_key, doc in provided_files.items():
                # 내용 기반 매칭 점수 계산
                doc_content = doc.page_content.lower()
                
                # 답변의 해당 부분과 문서 내용 간의 유사도 확인
                # 간단한 키워드 매칭으로 관련성 판단
                answer_keywords = re.findall(r'[가-힣a-zA-Z0-9]{3,}', answer.lower())
                matching_keywords = [kw for kw in answer_keywords if kw in doc_content]
                
                if matching_keywords:
                    score = len(matching_keywords) / len(answer_keywords) if answer_keywords else 0
                    if score > best_score:
                        best_score = score
                        best_match = provided_key
            
            if best_match:
                print(f"✅ 출처 수정: [{cited_file}] → [{best_match}] (관련성: {best_score:.2f})")
                corrected_answer = corrected_answer.replace(f"[{cited_file}]", f"[{best_match}]")
            else:
                print(f"❌ 적절한 대체 출처를 찾을 수 없음: [{cited_file}]")
                # 첫 번째 문서로 대체
                if provided_files:
                    first_doc = list(provided_files.keys())[0]
                    print(f"🔄 첫 번째 문서로 대체: [{cited_file}] → [{first_doc}]")
                    corrected_answer = corrected_answer.replace(f"[{cited_file}]", f"[{first_doc}]")
    
    return corrected_answer
