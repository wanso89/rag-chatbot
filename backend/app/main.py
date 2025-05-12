import os
import asyncio
import uuid
import time
import hashlib
import json
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from elasticsearch import Elasticsearch
from app.utils.indexing_utils import process_and_index_file, ES_INDEX_NAME, check_file_exists, format_file_size
from fastapi.responses import FileResponse, StreamingResponse
import mimetypes  # 파일 타입 감지용
from collections import Counter  # 피드백 통계용
import re  # 정규식 사용을 위한 모듈

# 검색 개선 모듈 import
from app.utils.search_enhancer import EnhancedSearchPipeline
# 피드백 분석 모듈 import
from app.utils.feedback_analyzer import FeedbackAnalyzer, SearchQualityOptimizer

# 모델 임포트
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import traceback

traceback.print_exc()

# 설정 상수
# LLM_MODEL_NAME = r"/home/root/ko-gemma-v1"
# EMBEDDING_MODEL_NAME = r"/home/root/KURE-v1"
# RERANKER_MODEL_NAME = r"/home/root/ko-reranker"
LLM_MODEL_NAME = r"/home/root/Qwen3-8B"
EMBEDDING_MODEL_NAME = r"/home/root/kpf-sbert-v1.1"
RERANKER_MODEL_NAME = r"/home/root/ko-reranker"
ES_HOST = "http://172.10.2.70:9200"
STATIC_DIR = "app/static"
IMAGE_DIR = os.path.join(STATIC_DIR, "document_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# 키워드 추출 함수 - 전역으로 정의
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
            '만약', '그러나', '하지만', '또한', '그리고', '따라서', '이러한', '그러한', 
            '이것', '그것', '저것', '무엇', '어디', '언제', '누구'
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
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 모델 로드 함수들
def get_elasticsearch_client():
    """Elasticsearch 클라이언트를 로드합니다."""
    print("Initializing Elasticsearch client...")
    try:
        client = Elasticsearch(
            ES_HOST,
            request_timeout=60,
            retry_on_timeout=True,
            max_retries=3,
            verify_certs=False,
        )

        if not client.ping():
            print("Elasticsearch 서버에 연결할 수 없습니다. 서버 상태를 확인하세요.")
            return None

        print("Elasticsearch client connected successfully.")

        # 인덱스 존재 확인 및 생성
        if not client.indices.exists(index=ES_INDEX_NAME):
            INDEX_SETTINGS = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "30s",
                    "analysis": {
                        "analyzer": {
                            "korean": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "filter": ["lowercase", "nori_part_of_speech"],
                            }
                        }
                    },
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "korean",
                            "search_analyzer": "korean",
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,
                                "ef_construction": 100,
                            },
                        },
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "category": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "total_chunks": {"type": "integer"},
                        "indexed_at": {"type": "date"},
                        "image_path": {"type": "keyword"},
                    }
                },
            }

            try:
                client.indices.create(index=ES_INDEX_NAME, body=INDEX_SETTINGS)
                print(f"Elasticsearch 인덱스 '{ES_INDEX_NAME}' 생성 완료.")
            except Exception as e:
                print(f"Elasticsearch 인덱스 생성 실패: {e}")
                return None

        return client
    except Exception as e:
        print(f"Elasticsearch 클라이언트 초기화 중 오류 발생: {e}")
        traceback.print_exc()
        return None


def get_embedding_function():
    """임베딩 모델 함수를 로드합니다."""
    print("Loading embedding model...")
    try:
        # LangchainEmbeddingFunction 클래스 정의
        class LangchainEmbeddingFunction:
            def __init__(self, model_name: str):
                # HuggingFaceEmbeddings 직접 사용
                self.model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    },
                )

            def __call__(self, texts: list[str]) -> List[List[float]]:
                # embed_documents 메서드 사용
                return self.model.embed_documents(texts)

        embedding_func = LangchainEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

        # 테스트 임베딩 실행
        try:
            test_result = embedding_func(["테스트"])
            print(
                f"Embedding model loaded successfully. Vector dimension: {len(test_result[0])}"
            )
            return embedding_func
        except Exception as e:
            print(f"임베딩 모델 테스트 중 오류 발생: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"임베딩 모델 로딩 중 오류 발생: {e}")
        traceback.print_exc()
        return None


def get_llm_model_and_tokenizer():
    print("Loading LLM model and tokenizer...")
    try:
        torch.cuda.empty_cache()  # 메모리 정리

        # 토크나이저 로드 최적화: 병렬 처리 옵션 활성화
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            use_fast=True,  # 빠른 토크나이저 사용
            padding_side="left",  # 왼쪽 패딩 (생성 모델에 적합)
            use_auth_token=None,  # 인증 토큰 불필요 시 명시적으로 None
            trust_remote_code=True,  # 원격 코드 신뢰 (일부 모델에 필요)
        )

        # 특수 토큰 설정 - Qwen3 모델용 업데이트
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Qwen3 모델 특수 토큰 확인 및 설정
        if "qwen" in LLM_MODEL_NAME.lower():
            tokenizer.padding_side = "left"
            # Qwen 모델은 이미 필요한 특수 토큰이 설정되어 있음

        # 양자화 설정 고급 옵션 (4비트 정밀도 + offload)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # CPU offload 활성화
        )

        # 모델 로드 최적화 설정
        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
            "device_map": "auto",  # 자동 장치 맵핑
            "revision": "main",
            "low_cpu_mem_usage": True,  # 낮은 CPU 메모리 사용
            "attn_implementation": "flash_attention_2",  # Flash Attention 2 사용 (지원 시)
            "use_cache": True,  # KV 캐시 활성화
            "trust_remote_code": True,  # 원격 코드 신뢰
        }

        # 모델 로드 시도
        try:
            # 먼저 Flash Attention으로 로드 시도
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                **model_kwargs
            )
            print("LLM model loaded successfully with Flash Attention.")
        except Exception as flash_att_error:
            print(f"Flash Attention 로드 실패, 표준 방식으로 재시도: {flash_att_error}")
            # Flash Attention 실패 시 일반 방식으로 로드
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                **model_kwargs
            )
            print("LLM model loaded successfully with standard attention.")

        # 모델 최적화 설정 (추론 전용)
        model.eval()  # 평가 모드 설정

        # 모델 메모리 사용 정보 출력 (옵션)
        if torch.cuda.is_available():
            print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        return model, tokenizer
    except Exception as e:
        print(f"LLM 모델 또는 토크나이저 로딩 중 오류 발생: {e}")
        traceback.print_exc()
        return None, None


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
        traceback.print_exc()
        return None


# ElasticsearchRetriever 클래스 정의
class ElasticsearchRetriever:
    def __init__(self, es_client: Any, embedding_function: Any, category: str, k=25):
        self.es_client = es_client
        self.index_name = ES_INDEX_NAME
        self.embedding_function = embedding_function
        self.k = k
        self.category = category
        # 성능 최적화를 위한 캐시 추가
        self._cache = {}
        self._cache_size = 150  # 최대 캐시 항목 수 증가 (100 → 150)
        self._cache_ttl = 7200  # 캐시 유효 시간 증가 (3600 → 7200초, 2시간)
        # 피드백 기반 검색 최적화 도구 초기화
        self.search_optimizer = SearchQualityOptimizer()

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not self.es_client or not self.embedding_function:
            print("Elasticsearch 클라이언트 또는 임베딩 함수가 초기화되지 않았습니다.")
            return []

        # 캐시 키 생성 (쿼리와 카테고리 조합)
        query_normalized = query.lower().strip()
        cache_key = f"{query_normalized}:{self.category}"

        # 캐시에서 결과 확인
        current_time = time.time()
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if current_time - cache_entry["timestamp"] < self._cache_ttl:
                print(f"캐시에서 검색 결과 반환: '{query[:30]}...'")
                return cache_entry["results"]

        # 캐시 정리 (필요시) - LRU 방식 최적화
        if len(self._cache) >= self._cache_size:
            # 가장 오래된 항목부터 삭제 (LRU)
            oldest_keys = sorted(
                self._cache.keys(), 
                key=lambda k: self._cache[k]["timestamp"]
            )[:len(self._cache) // 3]  # 1/3 정도 삭제 (기존 1/4에서 증가)
            for old_key in oldest_keys:
                del self._cache[old_key]

        try:
            # 임베딩 생성
            query_embedding = self.embedding_function([query_normalized])[0]

            # 하이브리드 쿼리 구성 (BM25 + 벡터 검색) - 가중치 최적화
            hybrid_query = {
                "size": self.k,
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "should": [
                            # 1. 정확한 문구 검색 (가중치 상향)
                            {
                                "match_phrase": {
                                    "text": {"query": query, "boost": 3.5, "slop": 3}  # 3.0 → 3.5
                                }
                            },
                            # 2. BM25 키워드 검색 (가중치 상향)
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 2.5,  # 2.0 → 2.5
                                        "operator": "OR",
                                        "minimum_should_match": "60%",  # 50%에서 60%로 상향
                                    }
                                }
                            },
                            # 3. 벡터 검색 (가중치 상향)
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding},
                                    },
                                    "boost": 2.2,  # 가중치 상향 (1.5 → 2.2)
                                }
                            },
                        ],
                        "filter": [{"term": {"category": self.category}}],
                        "minimum_should_match": 1,
                    }
                },
            }

            # 피드백 기반 쿼리 최적화 적용
            try:
                optimized_query = self.search_optimizer.apply_optimizations_to_query(
                    query_normalized, hybrid_query
                )
                if optimized_query != hybrid_query:
                    print(f"피드백 기반 쿼리 최적화 적용됨: '{query[:30]}...'")
                    hybrid_query = optimized_query
            except Exception as optimize_error:
                print(f"쿼리 최적화 적용 중 오류: {optimize_error}")
                # 최적화 오류 시 원본 쿼리 사용

            # 검색 실행 (타임아웃 설정)
            response = self.es_client.search(
                index=self.index_name,
                body=hybrid_query,
                request_timeout=30  # 30초 타임아웃
            )

            # 결과 처리
            docs = []
            for hit in response["hits"]["hits"]:
                # 메타데이터 추출 (embedding 필드 제외)
                metadata = {}
                for k, v in hit["_source"].items():
                    if k != "text" and k != "embedding":
                        metadata[k] = v

                # 점수 정규화 (0~1 사이로)
                raw_score = hit["_score"]
                # Elasticsearch 스코어는 범위가 다양하므로 정규화
                normalized_score = min(
                    max(raw_score / 10, 0), 1
                )  # 10으로 나누어 0~1 범위로 조정

                metadata["relevance_score"] = raw_score  # 원본 점수 유지
                metadata["source"] = hit["_source"].get("source", "unknown")
                metadata["page"] = hit["_source"].get("page", 1)

                # chunk_id가 정수형이면 문자열로 변환 (type 오류 방지)
                chunk_id = hit["_source"].get("chunk_id")
                if chunk_id is not None:
                    metadata["chunk_id"] = str(chunk_id)  # 명시적 문자열 변환

                # Document 객체 생성
                docs.append(
                    Document(
                        page_content=hit["_source"].get("text", ""), metadata=metadata
                    )
                )

            print(f"검색 완료: {len(docs)} 문서 검색됨")

            # 결과 캐싱
            self._cache[cache_key] = {
                "results": docs,
                "timestamp": current_time,
            }

            return docs
        except Exception as e:
            print(f"Elasticsearch 검색 중 오류 발생: {e}")
            traceback.print_exc()
            return []


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
        if not docs or not self.reranker:
            return []

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
                # torch CUDA 설정으로 성능 최적화
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    batch_scores = self.reranker.predict(batch_pairs)
                    scores.extend(batch_scores)

            # 메타데이터에 점수 추가 및 정규화
            for doc, score in zip(docs_to_rerank, scores):
                # 점수 범위를 0~1로 정규화 (-1~1 범위에서)
                normalized_score = min(max((score + 1) / 2, 0), 1)
                doc.metadata["rerank_score"] = float(normalized_score)
                doc.metadata["raw_rerank_score"] = float(score)  # 원본 점수도 저장

            # 리랭킹된 문서와 나머지 문서 결합
            sorted_docs = sorted(
                docs_to_rerank,
                key=lambda x: x.metadata.get("rerank_score", 0.0),
                reverse=True,
            )

            # 나머지 문서 추가 (이미 포함된 문서 제외)
            remaining_docs = [doc for doc in docs[12:] if doc not in docs_to_rerank]
            sorted_docs.extend(remaining_docs)

            # 임계값 필터링 - 점수가 낮은 문서 제외 (임계값 하향으로 더 많은 문서 포함)
            threshold = 0.52  # 임계값 하향 (0.6 → 0.52)
            filtered_docs = [
                doc
                for doc in sorted_docs
                if doc.metadata.get("rerank_score", 0.0) >= threshold
            ]

            # 필터링 결과가 최소 개수 미만이면 상위 문서 추가
            min_docs = 3  # 최소 3개 문서 보장
            if len(filtered_docs) < min_docs and sorted_docs:
                additional_docs = [
                    doc for doc in sorted_docs 
                    if doc not in filtered_docs
                ][:min_docs - len(filtered_docs)]
                filtered_docs.extend(additional_docs)

            # 결과 캐싱
            result_docs = filtered_docs[:self.top_n]
            self._cache[cache_key] = {
                "results": result_docs,
                "timestamp": current_time
            }

            return result_docs

        except Exception as e:
            print(f"Reranking 중 오류 발생: {e}")
            traceback.print_exc()
            return docs


# LLM 답변 생성 함수
async def generate_llm_response(
    llm_model: Any,
    tokenizer: Any,
    question: str,
    top_docs: List[Document],
    temperature: float = 0.2,
    conversation_history=None,
) -> str:
    if not llm_model or not tokenizer:
        return "LLM 모델 또는 토크나이저가 로드되지 않았습니다."
    if not top_docs:
        return "관련 문서를 찾지 못해 답변을 생성할 수 없습니다."

    # 대화 기록이 없으면 빈 리스트로 초기화
    if conversation_history is None:
        conversation_history = []

    # 문서 컨텍스트 준비 - 최적화: 상위 7개 문서 사용 (기존 5개에서 증가)
    context_parts = []
    for i, doc in enumerate(top_docs[:7]):  # 상위 7개 문서 사용 (증가)
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[문서 {i+1}] (출처: {source}, 페이지: {page})\n{doc.page_content.strip()}"
        )

    combined_context = "\n\n".join(context_parts)

    # 이전 대화 기록 포맷팅 - 최적화: 최근 3개 턴으로 제한 (기존 5개에서 감소)
    conversation_context = ""
    if conversation_history:
        conversation_parts = []
        # 대화 기록의 최대 길이 제한 (최근 3개 턴)
        recent_history = (
            conversation_history[-3:]
            if len(conversation_history) > 3
            else conversation_history
        )

        for msg in recent_history:
            role = "사용자" if msg["role"] == "user" else "시스템"
            conversation_parts.append(f"{role}: {msg['content']}")

        conversation_context = "대화 기록:\n" + "\n".join(conversation_parts) + "\n\n"

    # 간결하고 명확한 프롬프트 - 최적화: 프롬프트 단순화
    prompt = f"""<|im_start|>system
당신은 사용자 질문에 대해 주어진 참고 문서를 기반으로 답변하는 한국어 AI 어시스턴트입니다.
다음 지침을 매우 엄격히 따라주세요:

1. 반드시 한국어로만 답변하세요. 영어로 절대 응답하지 마세요.
2. 영어로 먼저 답변한 후 한국어로 번역하는 방식은 절대 사용하지 마세요. 처음부터 한국어로만 답변하세요.
3. 답변은 반드시 제공된 '참고 문서' 섹션의 내용에 근거해야 합니다.
4. 문서에 질문과 관련된 정보가 없다면, "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 명확히 답변하세요.
5. 답변은 명확하고 간결하게 작성해주세요.
6. HTML 태그는 <b>, <ul>, <li>만 사용할 수 있습니다.

특별 주의사항:
- 어떤 상황에서도 영어로 응답하지 마세요.
- 어떤 상황에서도 영어와 한국어를 함께 사용하지 마세요.
- 첫 문장부터 반드시 한국어로만 작성하세요.
<|im_end|>

<|im_start|>user
{conversation_context}
현재 질문: {question}

참고 문서:
{combined_context}
<|im_end|>

<|im_start|>assistant
한국어 답변: """

    try:
        # 메모리 최적화를 위한 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):  # Mixed precision 활성화
            # 입력 인코딩 및 CUDA 장치로 이동
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

            # 생성 파라미터 최적화 - 안정성 및 속도 개선
            outputs = await asyncio.wait_for(
                asyncio.to_thread(
                    llm_model.generate,
                    **inputs,
                    max_new_tokens=512,  # 약간 증가 (500 → 512)
                    repetition_penalty=1.2,
                    temperature=0.01,  # 더 낮은 온도로 설정 - 더 결정적인 응답 (0.05 → 0.01)
                    do_sample=False,  # 결정적인 디코딩 사용
                    top_p=0.95,
                    top_k=40,
                    pad_token_id=tokenizer.eos_token_id,  # 패딩 토큰 명시적 설정
                    num_beams=1,  # 빔 서치 없이 빠른 생성
                ),
                timeout=60.0,  # 타임아웃 60초로 증가 (기존 45초)
            )

            # 효율적인 텍스트 디코딩 (skip_special_tokens=True)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 응답 추출 로직 개선 - 영어 출력 제거 및 한국어 응답 최적화
            import re  # 상단에서 import가 있으므로 여기서는 불필요하지만, 명시적으로 추가
            
            if "<|im_start|>" in generated_text:
                # <|im_start|>assistant 이후 텍스트 추출
                parts = generated_text.split("<|im_start|>assistant")
                if len(parts) > 1:
                    answer = parts[1].strip()
                    
                    # 영어 텍스트 제거 로직 - 강화된 버전
                    # 1. "한국어 답변:" 패턴으로 시작하는지 확인
                    korean_prefix_match = re.search(r'한국어\s*답변\s*:', answer, re.IGNORECASE)
                    if korean_prefix_match:
                        # 해당 패턴 이후의 텍스트만 사용
                        answer = answer[korean_prefix_match.end():].strip()
                    
                    # 2. 한글이 처음 등장하는 위치 찾기
                    korean_match = re.search(r'[가-힣]', answer)
                    if korean_match:
                        korean_start_idx = korean_match.start()
                        
                        # 첫 한글 글자 전에 100자 이상의 텍스트가 있으면 영어 응답일 가능성이 높음
                        if korean_start_idx > 100:
                            # 첫 한글 직전에 마침표, 줄바꿈, 공백 등이 있는지 찾기
                            sentence_end = max([
                                answer[:korean_start_idx].rfind('. '),
                                answer[:korean_start_idx].rfind('.\n'),
                                answer[:korean_start_idx].rfind('? '),
                                answer[:korean_start_idx].rfind('?\n'),
                                answer[:korean_start_idx].rfind('! '),
                                answer[:korean_start_idx].rfind('!\n'),
                                answer[:korean_start_idx].rfind('\n\n')
                            ])
                            
                            if sentence_end > 0:
                                # 영어 응답 종료 후 한글 응답 시작 지점 발견
                                answer = answer[sentence_end+1:].strip()
                                print(f"영어 응답이 감지되어 제거됨. 한글 응답만 유지함.")
                    
                    # 3. 응답 시작 부분에 있는 '답변:', '한국어 답변:' 등 패턴 정리
                    answer = re.sub(r'^(답변|응답|결과|한국어\s*답변)\s*:', '', answer).strip()
                    
                    # 언어 확인 - 한글이 없으면 경고 로그 출력
                    if not re.search(r'[가-힣]', answer):
                        print("WARNING: 응답에 한글이 포함되어 있지 않습니다. 혹시 모델이 지시를 따르지 않는 것 같습니다.")
                else:
                    answer = generated_text
            elif "assistant" in generated_text:
                # assistant 이후 텍스트만 추출
                answer = generated_text.split("assistant")[-1].strip()
                
                # 위와 동일한 영어 텍스트 제거 로직 적용
                korean_prefix_match = re.search(r'한국어\s*답변\s*:', answer, re.IGNORECASE)
                if korean_prefix_match:
                    answer = answer[korean_prefix_match.end():].strip()
                
                korean_match = re.search(r'[가-힣]', answer)
                if korean_match:
                    korean_start_idx = korean_match.start()
                    if korean_start_idx > 100:
                        sentence_end = max([
                            answer[:korean_start_idx].rfind('. '),
                            answer[:korean_start_idx].rfind('.\n'),
                            answer[:korean_start_idx].rfind('? '),
                            answer[:korean_start_idx].rfind('?\n'),
                            answer[:korean_start_idx].rfind('! '),
                            answer[:korean_start_idx].rfind('!\n'),
                            answer[:korean_start_idx].rfind('\n\n')
                        ])
                        
                        if sentence_end > 0:
                            answer = answer[sentence_end+1:].strip()
                            print(f"영어 응답이 감지되어 제거됨. 한글 응답만 유지함.")
                
                # 응답 시작 부분의 '답변:' 등 제거
                answer = re.sub(r'^(답변|응답|결과|한국어\s*답변)\s*:', '', answer).strip()
            else:
                # 기타 경우 - 프롬프트에 응답 추가 부분 제거
                prompt_parts = prompt.split("<|im_start|>assistant")[0]
                if prompt_parts in generated_text:
                    answer = generated_text.replace(prompt_parts, "").strip()
                else:
                    answer = generated_text

            # 빈 응답 처리
            if not answer.strip():
                answer = "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다. 다시 질문해 주세요."
                
            return answer  # 수정: 들여쓰기 제거, 항상 응답 반환
    except asyncio.TimeoutError:
        print(f"LLM 답변 생성 시간 초과 (질문: {question[:50]}...)")
        return "답변 생성 시간이 초과되었습니다. 더 짧은 질문으로 다시 시도해 주세요."
    except Exception as e:
        print(f"LLM 답변 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return "답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요."


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
        print("--- DEBUG: Query is empty after strip, returning early. ---")
        return {
            "answer": "질문 내용을 입력해주세요. 공백만으로는 검색할 수 없습니다.",
            "sources": [],
        }

    print(f"--- DEBUG: search_and_combine ---")
    print(f"Processed query: '{query}'")

    start_time = time.time()

    # 검색 개선 파이프라인 초기화
    search_pipeline = EnhancedSearchPipeline()

    try:
        # 메모리 최적화를 위한 토치 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. 검색 (비동기 처리)
        retrieval_start = time.time()
        # ElasticsearchRetriever는 k=25 (성능 최적화 설정)
        retriever = ElasticsearchRetriever(
            es_client, embedding_function, category=category, k=25
        )

        docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - retrieval_start
        print(f"Retrieval time: {retrieval_time:.2f}s, Found {len(docs)} docs from ES.")

        # 검색 결과가 없을 경우 조기 반환
        if not docs:
            print("--- DEBUG: No documents found from retrieval. ---")
            return {
                "answer": "검색된 관련 문서가 없습니다. 다른 질문을 시도해 보세요.",
                "sources": [],
            }

        # 메모리 최적화를 위한 토치 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1.5. 검색 개선 파이프라인 적용 (쿼리 확장, 점수 보정, 컨텍스트 리랭킹)
        enhance_start = time.time()
        # 검색 개선 파이프라인 실행
        try:
            query_info, enhanced_docs = search_pipeline.process(query, docs)

            if enhanced_docs:
                docs = enhanced_docs
                print(f"Enhanced search with query variants: {', '.join(query_info['variants'])}")
            else:
                print("Enhanced search returned no results, using original docs")
        except Exception as enhance_error:
            print(f"검색 개선 적용 중 오류 발생: {enhance_error}")
            traceback.print_exc()
            # 개선 실패 시 원본 문서 사용

        enhance_time = time.time() - enhance_start
        print(f"Search enhancement time: {enhance_time:.2f}s")

        # 2. Reranking (최적화 - 비동기 처리)
        rerank_start = time.time()
        # EnhancedLocalReranker는 top_n=18 (성능 최적화 설정)
        reranker = EnhancedLocalReranker(reranker_model, top_n=18)

        try:
            reranked_docs = reranker.rerank(query, docs)
            rerank_time = time.time() - rerank_start
            print(f"Reranking time: {rerank_time:.2f}s, Reranked to {len(reranked_docs)} docs.")
        except Exception as rerank_error:
            print(f"Reranking 중 오류 발생, 원본 문서 사용: {rerank_error}")
            traceback.print_exc()
            # 리랭킹 실패 시 원본 문서 사용
            reranked_docs = docs[:15]  # 상위 15개만 사용

        # 최종 토큰 수 제한
        # 컨텍스트 크기를 최대 3,500 토큰으로 제한
        max_tokens = 3500
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

        # LLM 입력 형식으로 변환
        # 리랭킹 결과 문서들을 하나의 컨텍스트로 결합
        context_chunks = []
        source_metadata = []

        for i, doc in enumerate(final_docs):
            # 소스 정보 URL 인코딩
            source_path = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", i)

            # 텍스트에서 불필요한 공백과 개행 정리
            chunk_text = doc.page_content.strip()
            chunk_text = " ".join(chunk_text.split())

            # 파일명에서 UUID 제거 (UUID_파일명.확장자 형식 가정)
            clean_filename = os.path.basename(source_path)
            # UUID_ 패턴 감지 (UUID는 일반적으로 8-4-4-4-12 형식의 16진수 문자)
            if '_' in clean_filename:
                uuid_parts = clean_filename.split('_', 1)
                if len(uuid_parts) > 1 and len(uuid_parts[0]) >= 8:  # UUID로 추정되는 부분이 있으면 제거
                    clean_filename = uuid_parts[1]

            if chunk_text:
                context_chunks.append(f"[문서 {i+1}] {chunk_text}")
                source_metadata.append({
                        "path": source_path,
                    "display_name": clean_filename,  # 화면 표시용 정제된 파일명 추가
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "score": doc.metadata.get("relevance_score", 0),
                })

        full_context = "\n\n".join(context_chunks)
        print(f"Combined context length: {len(full_context)} characters.")

        # LLM으로 답변 생성
        llm_start = time.time()
        answer = await generate_llm_response(
            llm_model, tokenizer, query, final_docs, 0.2, conversation_history
        )
        
        # 응답이 None인 경우 대체 응답 사용 (방어 코드)
        if answer is None:
            print("LLM 응답이 None입니다. 대체 응답을 사용합니다.")
            answer = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 다시 질문해 주세요."
        
        # 소스 텍스트가 LLM 출력에 직접 인용된 경우를 확인
        cited_sources = []
        
        try:
            # 답변에서 키워드 추출 (함수 이름 수정: extract_keywords → extract_keywords_from_text)
            answer_keywords = extract_keywords_from_text(answer)
            print(f"답변에서 추출된 키워드: {answer_keywords}")
            
            # 응답 시간 측정
            llm_time = time.time() - llm_start
            print(f"LLM time: {llm_time:.2f}s")
            
            # 인용 감지 로직
            for i, doc in enumerate(final_docs):
                cited = False
                meta = source_metadata[i] if i < len(source_metadata) else {}
                source_text = context_chunks[i] if i < len(context_chunks) else ""

                # 1. 기존 방식: 연속된 텍스트 일치 여부 확인 (최소 30자)
                if len(source_text) > 50:
                    for j in range(0, len(source_text) - 30, 10):
                        snippet = source_text[j:j+30]
                        if snippet in answer:
                            cited = True
                            break

                # 2. 개선된 방식: 키워드 기반 매칭 (기존 방식으로 감지되지 않은 경우)
                if not cited and source_text:
                    # 소스에서 키워드 추출 (함수 이름 수정)
                    source_keywords = extract_keywords_from_text(source_text)
                    # 키워드 일치율 계산
                    if source_keywords:
                        matches = [k for k in source_keywords if k in answer]
                        match_ratio = len(matches) / len(source_keywords)
                        # 키워드의 30% 이상이 응답에 포함되어 있으면 인용으로 간주
                        if match_ratio > 0.3:
                            cited = True
                
                # 메타데이터에 인용 여부 저장
                if meta:
                    meta["is_cited"] = cited
                    if cited:
                        cited_sources.append(meta)
        except Exception as e:
            print(f"인용 처리 중 오류 발생: {e}")
            print("유효한 응답이 없어 인용 처리를 건너뜁니다.")
            # 모든 메타데이터에 is_cited = False 설정
            for meta in source_metadata:
                meta["is_cited"] = False

        llm_time = time.time() - llm_start
        print(f"LLM generation time: {llm_time:.2f}s")
        print(f"응답에 포함된 출처 수: {len(cited_sources)}")

        # 최종 응답 생성
        return {
            "answer": answer,
            "sources": source_metadata,
            "cited_sources": cited_sources,
            "processing_time": {
                "retrieval": round(retrieval_time, 2),
                "enhancement": round(enhance_time, 2),
                "reranking": round(rerank_time, 2),
                "llm_generation": round(llm_time, 2),
                "total": round(time.time() - start_time, 2),
            },
        }

    except Exception as e:
        print(f"검색 및 응답 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "answer": f"요청을 처리하는 중 오류가 발생했습니다: {str(e)}",
            "sources": [],
            "cited_sources": [],  # 빈 cited_sources 추가
        }


# 모델 초기화
es_client = get_elasticsearch_client()
embedding_function = get_embedding_function()
llm_model, tokenizer = get_llm_model_and_tokenizer()
reranker_model = get_reranker_model()


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
    chunk_id: Any  # 문자열 또는 정수일 수 있으므로 Any (ES 저장 방식에 따라)
    keywords: Optional[List[str]] = None  # 프론트엔드에서 전달하는 하이라이트 키워드 (선택)
    answer_text: Optional[str] = None  # 챗봇 응답 전체 텍스트 (선택)


# 참고 문서 미리보기 엔드포인트
@app.post("/api/source-preview")
async def source_preview_endpoint(request: SourcePreviewRequest = Body(...)):
    try:
        print(
            f"Source preview 요청: path={request.path}, page={request.page}, chunk_id={request.chunk_id}"
        )

        # chunk_id 전처리 - 명시적 문자열 변환
        chunk_id_query = str(request.chunk_id)  # 모든 경우 문자열로 처리

        # 다양한 쿼리 시도 (우선순위에 따라)
        search_attempts = []

        # 1. source, page, chunk_id(문자열)로 검색 (1순위)
        search_attempts.append({
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"source": request.path}},
                            {"term": {"page": request.page}},
                        {"term": {"chunk_id": chunk_id_query}}
                        ]
                    }
                }
        })

        # 2. chunk_id가 정수로 저장되었을 가능성 (2순위)
        try:
            chunk_id_int = int(request.chunk_id)
            search_attempts.append({
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"source": request.path}},
                                {"term": {"page": request.page}},
                            {"term": {"chunk_id": chunk_id_int}}
                            ]
                        }
                    }
            })
        except (ValueError, TypeError):
            # 정수 변환 불가 시 이 시도는 건너뜀
            pass

        # 3. source와 page로만 검색 (3순위)
        search_attempts.append({
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"source": request.path}},
                        {"term": {"page": request.page}}
                        ]
                    }
                },
                "size": 1,
            "sort": [{"chunk_id": "asc"}]  # 첫 번째 청크 선택
        })

        # 4. source_path만으로 검색 (4순위 - 마지막 시도)
        search_attempts.append({
            "query": {
                "term": {"source": request.path}}
            ,
            "size": 1
        })
        
        # 문서 콘텐츠 포맷팅 함수
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
            """내용에서 문장 단위 하이라이트 - 답변과 일치하는 문장 강조 (개선된 버전)"""
            if not content:
                return content
                
            # 답변 텍스트가 있으면 문장 단위 매칭 적용 (개선된 버전)
            if answer_text and isinstance(answer_text, str) and len(answer_text.strip()) > 10:
                print(f"[개선된] 응답 텍스트 기반 하이라이트 시작 - 응답 길이: {len(answer_text)}")
                
                # 문장 단위로 더 정확하게 분리 (콜론 포함)
                # 1. 응답 텍스트 문장 분리 - 다양한 구분자와 콜론 처리 개선
                answer_text = answer_text.replace(' : ', ': ')  # 콜론 정규화
                raw_answer_sentences = []
                
                # 주요 구분점 먼저 처리 (예: "첫째,", "1.", "방법은 다음과 같습니다:" 등)
                segments = re.split(r'(다음과\s*같습니다\s*:|\d+\.\s|\s*첫째,|\s*둘째,|\s*셋째,|\s*넷째,|\s*다섯째,)', answer_text)
                
                for i in range(len(segments)):
                    segment = segments[i].strip()
                    if not segment:
                        continue
                        
                    # 구분점이면 다음 부분에 붙여서 처리
                    if re.match(r'(다음과\s*같습니다\s*:|\d+\.\s|\s*첫째,|\s*둘째,|\s*셋째,|\s*넷째,|\s*다섯째,)', segment):
                        if i + 1 < len(segments):
                            raw_answer_sentences.append(segment + " " + segments[i+1].strip())
                    # 일반 텍스트면 문장 단위로 분리
                    elif i > 0 and re.match(r'(다음과\s*같습니다\s*:|\d+\.\s|\s*첫째,|\s*둘째,|\s*셋째,|\s*넷째,|\s*다섯째,)', segments[i-1].strip()):
                        continue  # 이미 앞 단계에서 처리됨
                    else:
                        # 일반 문장 분리
                        sub_sentences = re.split(r'([.!?]\s+|\n\s*\n)', segment)
                        for sub in sub_sentences:
                            if sub.strip():
                                raw_answer_sentences.append(sub.strip())
                
                # 최종 응답 문장 정제 (중복 제거, 최소 길이 필터)
                answer_sentences = []
                for sent in raw_answer_sentences:
                    sent = sent.strip()
                    # 유효한 문장만 추가 (길이 조건 완화)
                    if sent and len(sent) > 4 and sent not in answer_sentences:
                        answer_sentences.append(sent)
                
                # 2. 원본 콘텐츠 문장 분리 개선
                raw_content_lines = []
                
                # 줄바꿈 기준 분리 후 처리
                paragraphs = re.split(r'\n\s*\n', content)
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                        
                    # 문단 내 문장 분리
                    lines = re.split(r'([.!?]\s+)', para)
                    current_line = ""
                    
                    for i in range(len(lines)):
                        line = lines[i].strip()
                        if not line:
                            continue
                            
                        if re.match(r'[.!?]\s+', line):
                            # 문장 종결자 - 이전 문장과 합쳐서 저장
                            raw_content_lines.append((current_line + line).strip())
                            current_line = ""
                        else:
                            # 콜론 포함 문장은 별도 처리
                            if ':' in line and len(line) > 10:
                                colon_parts = line.split(':', 1)
                                if colon_parts[0].strip() and colon_parts[1].strip():
                                    raw_content_lines.append(line.strip())
                                    # 현재 문장 초기화하지 않고 계속 유지
                                    current_line = line
                                else:
                                    current_line = line
                            else:
                                current_line = line
                    
                    # 마지막 문장 처리
                    if current_line:
                        raw_content_lines.append(current_line.strip())
                
                # 빈 줄이나 너무 짧은 줄 제거
                content_lines = [line for line in raw_content_lines if line and len(line.strip()) > 4]
                
                # 디버그 로그
                print(f"[개선된] 분리된 응답 문장 수: {len(answer_sentences)}")
                for i, sent in enumerate(answer_sentences[:5]):  # 처음 5개만 출력
                    print(f"  응답 문장 {i+1}: {sent[:50]}...")
                print(f"[개선된] 분리된 원본 문장 수: {len(content_lines)}")
                
                # 결과 저장용 배열 초기화
                result = []
                highlight_count = 0
                max_highlights = 15  # 최대 하이라이트 수 증가
                
                # 매칭된 원본 문장 기록 (중복 하이라이트 방지)
                highlighted_lines = set()
                
                # 3. 답변의 각 문장마다 매칭되는 원본 문장 찾기
                for ans_sent in answer_sentences:
                    ans_norm = re.sub(r'\s+', ' ', ans_sent.lower())
                    
                    # 응답 문장에서 불용어 제거하고 핵심 단어 추출
                    ans_words = set(re.findall(r'\b\w+\b', ans_norm))
                    stopwords = {'그', '이', '저', '것', '수', '를', '을', '에', '에서', '와', '과', '은', '는', '이다', '있다', '하다', '다음', '같습니다', '방법', '경우', '의', '및', '또는', '후', '전', '중', '내', '외', '상', '하', '좌', '우'}
                    ans_keywords = ans_words - stopwords
                    
                    # 매칭 결과 저장 변수
                    best_match_score = 0
                    best_match_line_idx = -1
                    
                    # 모든 원본 문장 검사
                    for line_idx, line in enumerate(content_lines):
                        # 이미 하이라이트된 라인은 건너뜀
                        if line_idx in highlighted_lines:
                            continue
                            
                        line_norm = re.sub(r'\s+', ' ', line.lower())
                        
                        # 1. 정확히 일치하는 경우 (포함 관계)
                        if ans_norm in line_norm or line_norm in ans_norm:
                            best_match_score = 1.0
                            best_match_line_idx = line_idx
                            break
                            
                        # 2. 단어 유사도 계산
                        line_words = set(re.findall(r'\b\w+\b', line_norm))
                        if ans_keywords and line_words:
                            # 응답 키워드가 원본 문장에 얼마나 포함되는지
                            if len(ans_keywords) > 0:  # 분모가 0이 아닌지 확인
                                match_score = len(ans_keywords.intersection(line_words)) / len(ans_keywords)
                                
                                # 더 나은 매치 발견 시 업데이트
                                if match_score > best_match_score and match_score >= 0.6:  # 임계값 60%로 조정
                                    best_match_score = match_score
                                    best_match_line_idx = line_idx
                    
                    # 발견된 최적 매치 적용
                    if best_match_line_idx >= 0 and best_match_score >= 0.6 and highlight_count < max_highlights:
                        highlighted_lines.add(best_match_line_idx)
                        highlight_count += 1
                        
                        # 최적 매치 로그 출력
                        matched_line = content_lines[best_match_line_idx]
                        log_line = matched_line[:70] + ("..." if len(matched_line) > 70 else "")
                        log_ans = ans_sent[:50] + ("..." if len(ans_sent) > 50 else "")
                        print(f"매치 발견: '{log_line}' ↔ '{log_ans}', 점수: {best_match_score:.2f}")
                
                # 4. 결과 생성 - 하이라이트 적용하여 원본 텍스트 재구성
                current_text = content
                
                # 하이라이트할 원본 문장 목록 (인덱스로 정렬)
                sorted_highlights = sorted(highlighted_lines)
                
                # 각 하이라이트 대상 문장에 대해 마크다운 볼드 적용
                for line_idx in sorted_highlights:
                    if line_idx < len(content_lines):
                        line_to_highlight = content_lines[line_idx]
                        
                        # 정규식 특수문자 이스케이프 처리
                        pattern_text = re.escape(line_to_highlight)
                        
                        # 원본 텍스트에서 해당 문장을 찾아 볼드 처리
                        try:
                            pattern = re.compile(f"(?<!\*){pattern_text}(?!\*)", re.DOTALL)
                            current_text = pattern.sub(f"**{line_to_highlight}**", current_text)
                        except re.error as e:
                            print(f"정규식 오류: {e}, 문장: {line_to_highlight[:30]}...")
                
                print(f"[개선된] 하이라이트 완료 - 총 {len(sorted_highlights)}개 문장 하이라이트")
                return current_text
            
            # 기존 키워드 기반 하이라이트 로직 (백업용)
            elif keywords and isinstance(keywords, list) and any(keywords):
                # 중복, 공백 제거
                unique_keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
                if not unique_keywords:
                    return content
                    
                print(f"키워드 기반 하이라이트 시작 - {len(unique_keywords)}개 키워드")
                
                # 내용을 문장으로 분리
                content_lines = re.split(r'([.!?]\s+|\n\s*\n)', content)
                result = []
                
                # 각 콘텐츠 줄에 대해 처리
                i = 0
                highlight_count = 0
                while i < len(content_lines):
                    line = content_lines[i]
                    if not line.strip():
                        result.append(line)
                        i += 1
                        continue
                        
                    # 문장 종결자가 다음에 있는 경우
                    if i + 1 < len(content_lines) and re.match(r'[.!?]\s+', content_lines[i+1]):
                        complete_line = line + content_lines[i+1]
                        i += 2
                    else:
                        complete_line = line
                        i += 1
                    
                    # 키워드와 매칭 여부 확인
                    should_highlight = False
                    matching_keywords = []
                    
                    for keyword in unique_keywords:
                        if keyword.lower() in complete_line.lower():
                            should_highlight = True
                            matching_keywords.append(keyword)
                    
                    # 하이라이트 적용
                    if should_highlight:
                        result.append(f"**{complete_line}**")
                        highlight_count += 1
                        log_sent = complete_line[:70] + ("..." if len(complete_line) > 70 else "")
                        print(f"하이라이트된 문장: '{log_sent}', 매칭 키워드: {matching_keywords}")
                    else:
                        result.append(complete_line)
                
                # 결과 합치기
                result_text = "".join(result)
                print(f"하이라이트 완료 - 총 {highlight_count}개 문장 하이라이트")
                return result_text
            
            # 아무 매칭 조건이 없으면 원본 반환
            return content

        # 순차적으로 검색 시도
        for i, search_body in enumerate(search_attempts):
            # 기본 _source 필드 추가
            if "_source" not in search_body:
                search_body["_source"] = ["text", "source", "page", "chunk_id"]

            # 크기 지정이 없으면 기본값 설정
            if "size" not in search_body:
                search_body["size"] = 1

            response = es_client.search(index=ES_INDEX_NAME, body=search_body)
            hits = response["hits"]["hits"]

            if hits:
                hit = hits[0]["_source"]
                result_msg = f"검색 성공 (시도 {i+1}/{len(search_attempts)})"
                
                # 원본 텍스트 내용 가져오기
                content = hit.get("text", "내용을 찾을 수 없습니다.")
                
                # 내용 서식 개선
                formatted_content = format_content(content)
                
                # 프론트엔드에서 키워드를 전달한 경우 우선 사용, 없으면 자동 추출
                if request.keywords and isinstance(request.keywords, list) and any(request.keywords):
                    keywords = [str(k) for k in request.keywords if isinstance(k, str)]
                    print(f"프론트엔드 전달 키워드 사용: {keywords}")
                else:
                    keywords = extract_keywords_from_text(formatted_content)
                
                # 로그 추가
                print(f"추출된 키워드: {keywords}")
                
                # 유효한 키워드인지 확인
                valid_keywords = [k for k in keywords if k and isinstance(k, str) and len(k.strip()) > 0]
                
                # 프론트엔드에서 전달한 답변 텍스트 사용
                answer = request.answer_text if request.answer_text else None
                if answer:
                    print(f"프론트엔드에서 전달한 답변 텍스트 사용: {answer[:100]}...")
                
                # 키워드가 유효하지 않으면 로그
                if len(valid_keywords) < len(keywords):
                    print(f"유효하지 않은 키워드 제거됨: {set(keywords) - set(valid_keywords)}")
                
                # 키워드 하이라이트 적용
                highlighted_content = highlight_keywords(formatted_content, valid_keywords, answer_text=answer)

                # 모든 필드가 문자열로 변환되도록 보장
                result = {
                    "status": "success",
                    "content": highlighted_content,  # 하이라이트 적용된 내용
                    "original_content": formatted_content,  # 원본 포맷팅된 내용
                    "source": os.path.basename(hit.get("source", "N/A")),
                    "page": hit.get("page", 0),
                    "chunk_id": str(hit.get("chunk_id", "N/A")),  # 명시적 문자열 변환
                    "message": result_msg,
                    "keywords": valid_keywords  # 추출된 키워드
                }

                # 첫 번째 시도가 아니라면 알림 메시지 추가
                if i > 0:
                    result["message"] = f"원본 chunk_id로 찾지 못해 대체 문서를 표시합니다. ({result_msg})"

                return result

        # 모든 시도 실패
        return {
            "status": "not_found",
            "message": "해당 문서 내용을 찾을 수 없습니다.",
            "content": "",
            "original_content": "",
            "source": os.path.basename(request.path),
            "page": request.page,
            "chunk_id": str(request.chunk_id),  # 명시적 문자열 변환
            "keywords": []
        }
    except Exception as e:
        print(f"참고 문서 미리보기 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"참고 문서 미리보기 중 오류 발생: {str(e)}",
            "content": "",
            "original_content": "",
            "source": os.path.basename(request.path),
            "page": request.page,
            "chunk_id": str(request.chunk_id),  # 명시적 문자열 변환
            "keywords": []
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


# 서버 시작 시 인덱스 확인
@app.on_event("startup")
async def startup_event():
    if not all([es_client, embedding_function, llm_model, tokenizer, reranker_model]):
        print("필수 리소스 로딩에 실패했습니다. 서버 로그를 확인하세요.")


@app.get("/api/file-viewer/{filename}")
async def get_file_for_viewer(filename: str):
    # UUID가 포함된 전체 파일명을 사용한다고 가정
    # 보안: filename에 ../ 등이 포함되어 상위 디렉토리 접근 시도 방지
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 파일이 저장된 실제 경로 (main.py 기준 상대 경로 또는 절대 경로)
    # /api/upload에서 저장한 경로와 동일해야 함
    file_path = os.path.join(STATIC_DIR, "uploads", filename)

    print(f"File view request for: {filename}, Path: {file_path}")

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    # 파일 타입(MIME) 추측
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # 기본값

    # Content-Disposition을 inline으로 설정하여 브라우저에서 바로 열도록 시도
    # (브라우저 설정에 따라 다운로드될 수도 있음)
    # 실제 파일명을 표시하려면 cleanFilename 로직을 여기서도 사용 가능
    # clean_name = filename[filename.find('_')+1:] if '_' in filename else filename
    # headers = {'Content-Disposition': f'inline; filename="{clean_name}"'}

    # FileResponse 사용하여 파일 내용 반환
    # return FileResponse(path=file_path, media_type=mime_type, headers=headers)
    # 간단히 파일 내용만 반환 (브라우저가 타입에 맞게 처리)
    return FileResponse(path=file_path, media_type=mime_type)


# 파일 업로드 및 인덱싱 엔드포인트
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),  # 다중 파일 지원
    category: str = Form("메뉴얼"),  # 기본값을 메뉴얼로 설정
):
    results = []
    start_time = time.time()  # 전체 처리 시작 시간

    print(f"파일 업로드 요청 수신: {len(files)}개 파일, 카테고리: {category}")

    # 메모리 정리 - 초기 상태
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"업로드 처리 시작 - 초기 GPU 메모리 사용량: {initial_memory:.2f} MB")

    for file_index, file in enumerate(files):
        print(f"[{file_index+1}/{len(files)}] 파일 처리 중: {file.filename}, 카테고리: {category}")

        # 임시 파일 저장
        unique_id = uuid.uuid4()
        file_path = f"app/static/uploads/{unique_id}_{file.filename}"
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
                    "extension": os.path.splitext(file.filename)[1].lower(),
                    "index": file_index + 1,
                    "total": len(files)
                }
            }

            # 파일 처리 및 인덱싱
            try:
                print(f"파일 인덱싱 시작: {file.filename}")
                # 파일 중복 체크
                file_exists, file_hash = await check_file_exists(es_client, file_path)

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
                        print(f"파일 처리 전 GPU 메모리: {pre_process_memory:.2f} MB")

                    # 파일 처리 진행률 업데이트 (실제로는 비동기 처리가 필요할 수 있음)
                    file_result["progress"] = 30

                    success = await process_and_index_file(
                        es_client, embedding_function, file_path, category
                    )

                    # 메모리 사용량 확인
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        post_process_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                        memory_used = post_process_memory - pre_process_memory
                        print(f"파일 처리 후 GPU 메모리: {post_process_memory:.2f} MB (변화: {memory_used:.2f} MB)")

                    if success:
                        print(f"파일 인덱싱 성공: {file.filename}")
                        file_result.update({
                                "status": "success",
                                "message": f"파일 '{file.filename}' 인덱싱 완료",
                            "processing_time": round(time.time() - file_result["start_time"], 2),
                            "progress": 100
                        })
                    else:
                        print(f"파일 인덱싱 실패: {file.filename}")
                        file_result.update({
                                "status": "error",
                                "message": "파일 인덱싱 실패",
                            "processing_time": round(time.time() - file_result["start_time"], 2),
                            "progress": 100
                        })
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file.filename}, 오류: {str(e)}")
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
            print(f"파일 저장 중 오류 발생: {file.filename}, 오류: {str(e)}")
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
        print(f"전체 처리 완료 - 최종 GPU 메모리 사용량: {final_memory:.2f} MB")

    # 전체 요약 통계 추가
    total_time = round(time.time() - start_time, 2)
    summary = {
        "total_files": len(results),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
        "skipped_count": sum(1 for r in results if r["status"] == "skipped"),
        "total_size": sum(r["size"] for r in results),
        "total_size_formatted": format_file_size(sum(r["size"] for r in results)),
        "total_processing_time": total_time,
        "average_file_time": round(total_time / len(results), 2) if results else 0
    }

    return JSONResponse(
        content={
            "status": "success" if summary["error_count"] == 0 else "partial_success",
            "message": f"{summary['total_files']}개 파일 처리 완료. {summary['success_count']}개 성공, {summary['error_count']}개 실패, {summary['skipped_count']}개 건너뜀",
            "results": results,
            "summary": summary,
        }
    )

# 질문-응답 엔드포인트
@app.post("/api/chat")
async def chat(request: QuestionRequest = Body(...)):
    try:
        result = await search_and_combine(
            es_client,
            embedding_function,
            reranker_model,
            llm_model,
            tokenizer,
            query=request.question,
            category=request.category,
            conversation_history=request.history,
        )

        return {
            "bot_response": result["answer"],
            "sources": result["sources"],
            "cited_sources": result["cited_sources"],  # 인용된 출처 정보 추가
            "questionContext": request.question,
        }

    except Exception as e:
        print(f"챗봇 응답 생성 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"챗봇 응답 생성 중 오류 발생: {str(e)}"
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
