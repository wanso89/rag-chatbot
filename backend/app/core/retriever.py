#core/retriever.py

import os
import time
import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from elasticsearch import Elasticsearch
import torch
from langchain.schema import Document
from app.utils.feedback_analyzer import SearchQualityOptimizer
from app.utils.indexing_utils import ES_INDEX_NAME
from app.utils.qwen3_prompts import create_query_optimization_prompt, create_chat_messages
import logging


logger = logging.getLogger(__name__)

ALIAS_READ  = "rag-idx"
NUM_CANDID  = 200                # K-NN 후보군
DEFAULT_K = 30

@dataclass
class OptimizedQuery:
    original: str
    primary: str
    alternatives: List[str]
    keywords: List[str]
    confidence: float


class ElasticsearchRetriever:
    
    def __init__(
        self,
        es_client:    Elasticsearch,
        embedding_function: Any,
        category: str,
        k: int = DEFAULT_K,
        llm_model=None,
        tokenizer=None,
        index_name: str | None = None,      # ← 매개변수는 그대로 두되
    ):
        self.es_client         = es_client
        self.embedding_function= embedding_function
        self.k                 = k
        self.category          = category
        self.llm_model         = llm_model
        self.tokenizer         = tokenizer
        self.index_name        = index_name or ALIAS_READ
        
        # 캐시 설정
        self._cache = {}
        self._cache_size = 150
        self._cache_ttl = 7200
        
        # 검색 최적화기
        self.search_optimizer = SearchQualityOptimizer()
        
        # 쿼리 최적화기
        if llm_model and tokenizer:
            self.llm_model = llm_model
            self.tokenizer = tokenizer

        self.query_optimizer_enabled = bool(llm_model and tokenizer)
        
        if self.query_optimizer_enabled:
            print("✅ 쿼리 최적화 기능 활성화")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """동기 검색 함수 - 원래 방식으로 복구"""
        # 일단 원래대로 두고, 호출하는 쪽에서 async로 바꾸거나
        # 이 함수를 호출하는 곳을 async가 아닌 환경에서 호출하도록 수정
        return asyncio.run(self.async_get_relevant_documents(query))

    async def async_get_relevant_documents(self, query: str) -> List[Document]:
        """비동기 검색 함수"""
        
        if not self.es_client or not self.embedding_function:
            return []

        print(f"🔍 검색 시작: {query[:30]}...")

        # 쿼리 최적화
        search_queries = [query]
        if self.query_optimizer_enabled:
            optimized_queries = await self._optimize_query_with_qwen(query)
            if optimized_queries and len(optimized_queries) > 1:
                search_queries = optimized_queries
                print(f"📝 구조화된 쿼리들: {search_queries}")
            else:
                print("⚠️ 쿼리 최적화 실패, 원본 사용")
                search_queries = [query]
        
        # 다중 쿼리 검색
        all_docs = []
        for i, search_query in enumerate(search_queries):
            boost_factor = 1.0 if i == 0 else 0.7
            docs = await self._search_single_query(search_query, boost_factor)
            all_docs.extend(docs)
        
        # 결과 정리
        if len(search_queries) > 1:
            unique_docs = self._deduplicate_results(all_docs)
            unique_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
            result = unique_docs[:self.k]
        else:
            result = all_docs[:self.k]
        
        print(f"✅ 검색 완료: {len(result)}개 문서")
        return result
    
    async def _optimize_query_with_qwen(self, query: str) -> List[str]:
        """Qwen 쿼리 최적화 - 구조화된 분해 방식"""
        prompt = create_query_optimization_prompt(query, self.category)

        # 1) 프롬프트 토크나이즈
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1000,
        )

        # 2) 모델 디바이스 확인 → 입력도 맞춰서 이동
        device = next(self.llm_model.parameters()).device   # ex) cuda:0
        enc = {k: v.to(device) for k, v in enc.items()}     # 중요!

        # 3) 생성
        with torch.no_grad():
            out_ids = self.llm_model.generate(
                **enc,
                max_new_tokens=120,
                temperature=0.1,
                do_sample=True,
                top_p=0.8,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]

        # 4) 디코딩(입력 길이 이후만)
        gen_ids = out_ids[enc["input_ids"].shape[1]:].cpu().tolist()
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return self._parse_qwen_response(response, query)
    
    def _parse_qwen_response(self, response: str, original_query: str) -> List[str]:
        """Qwen 응답 파싱 - 더 다양한 쿼리 생성"""
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                return [original_query]

            data      = json.loads(json_match.group(0))
            keywords  = [k.strip() for k in data.get("keywords", []) if k.strip()]
            if not keywords:
                return [original_query]

            # ────────── 1. 대안 쿼리 만들기 ──────────
            alts = []

            # (a) 키워드 AND 결합  → 가장 높은 boost 를 주고 싶으므로 첫 번째
            alts.append(" AND ".join(keywords[:3]))       # ndt4 AND ceph AND 설치

            # (b) 두 핵심 키워드만 공백 결합
            alts.append(" ".join(keywords[:2]))           # ndt4 ceph

            # ────────── 2. 최종 순서 결정 ──────────
            # ① 자연어를 뒤에 두고 싶으면 ↓
            result = alts + [original_query]

            # ② 자연어를 아예 빼고 싶으면 ↓
            # result = alts

            print(f"🔍 최종 생성된 쿼리들: {result}")
            return result

        except Exception as e:
            print(f"⚠️ 파싱 오류: {e}")
            return [original_query]
    
    async def _search_single_query(self, query: str, boost_factor: float = 1.0) -> List[Document]:
        TABLE_KWS = ["표", "table", "도표", "차트"]
        wants_table = any(kw in query.lower() for kw in TABLE_KWS)
        
        # 캐시 확인
        query_normalized = query.lower().strip()
        cache_key = f"{query_normalized}:{self.category}:{boost_factor}"
        current_time = time.time()
        
        cache_entry = self._cache.get(cache_key)
        if cache_entry and current_time - cache_entry["timestamp"] < self._cache_ttl:
            return cache_entry["results"]
        
        # 캐시 정리
        if len(self._cache) >= self._cache_size:
            oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])[:len(self._cache) // 3]
            for old_key in oldest_keys:
                del self._cache[old_key]

        # 임베딩 생성
        query_embedding = self.embedding_function([query_normalized])[0]

        # 하이브리드 검색 쿼리 구성
        base_should = [
            # BM25 / phrase match
            {"match_phrase": {"text": {"query": query, "boost": 6.0 * boost_factor, "slop": 3}}},
            {"match": {
                "text": {"query": query, "boost": 4.0 * boost_factor,
                         "operator": "OR", "minimum_should_match": "60%"}
            }},
            # 벡터 유사도 (script_score)
            {"script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.qv, 'embedding') + 1.0",
                    "params": {"qv": query_embedding},
                },
                "boost": 3.0 * boost_factor,
            }},
        ]
        
        # 표 부스팅 부분
        table_boost = []
        if wants_table:
            table_boost = [
                {"terms": {"element_type": ["table", "table_row"] , "boost": 6.0 * boost_factor}}
            ]
        should_clauses = base_should + table_boost

        base_filter = [{"term": {"category": self.category}}]

        table_filter = []
        if query.strip() in TABLE_KWS:
            # 표만 달라는 쿼리라면 table + table_row 로 한정
            table_filter = [{"terms": {"element_type": ["table", "table_row"]}}]
        filter_clauses = base_filter + table_filter

        hybrid_query = {
            "size": self.k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "bool": {
                    "should": should_clauses,
                    "filter": filter_clauses,
                    "minimum_should_match": 1
                }
            }
        }

        # 피드백 기반 최적화 적용
        optimized_query = self.search_optimizer.apply_optimizations_to_query(query_normalized, hybrid_query)
        
        # 검색 실행
        response = self.es_client.search(index=self.index_name, body=optimized_query, request_timeout=30)

        # 결과 처리
        docs = []
        for hit in response["hits"]["hits"]:
            meta_src = hit["_source"]
            metadata = {k: v for k, v in meta_src.items() if k not in ["text", "embedding"]}
            metadata.update(
                {
                    "document_id": hit["_id"],
                    "relevance_score": hit["_score"] * boost_factor,
                    "source": meta_src.get("source", "unknown"),
                    "page": meta_src.get("page", 1),
                    # table_row일 경우 위치 추적용
                    "table_id": meta_src.get("table_id"),
                    "row_no": meta_src.get("row_no"),
                }
            )
            if chunk_id := meta_src.get("chunk_id"):
                metadata["chunk_id"] = str(chunk_id)

            docs.append(Document(page_content=meta_src.get("text", ""), metadata=metadata))


        # 캐싱
        self._cache[cache_key] = {"results": docs, "timestamp": current_time}
        return docs
    
    def _deduplicate_results(self, docs: List[Document]) -> List[Document]:
        """중복 제거"""
        seen, unique_docs = set(), []
        for doc in docs:
            key = (
                doc.metadata.get("document_id"),    # ES _id
                doc.metadata.get("page"),
                doc.metadata.get("chunk_id"),
            )
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        return unique_docs


    async def get_source_preview_document(
        self, 
        source_path: str, 
        page: int, 
        chunk_id: str = None, 
        original_query: str = None,
        keywords: List[str] = None
    ) -> Dict[str, Any]:
        """소스 프리뷰를 위한 문서 검색"""
        
        if not self.es_client:
            return {
                "status": "error",
                "message": "검색 서비스를 사용할 수 없습니다.",
                "content": None
            }
        
        # 간단한 우선순위 검색
        search_queries = []
        
        # 1순위: 정확한 매칭
        if chunk_id:
            search_queries.append({
                "bool": {
                    "must": [
                        {"term": {"source": source_path}},
                        {"term": {"page": page}},
                        {"term": {"chunk_id": str(chunk_id)}}
                    ]
                }
            })
        
        # 2순위: 페이지 매칭
        search_queries.append({
            "bool": {
                "must": [
                    {"term": {"source": source_path}},
                    {"term": {"page": page}}
                ]
            }
        })
        
        # 검색 실행
        found_document = None
        for query in search_queries:
            try:
                response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 1,
                        "_source": {"excludes": ["embedding"]},
                        "query": query
                    }
                )
                
                hits = response.get("hits", {}).get("hits", [])
                if hits:
                    found_document = hits[0]
                    break
                    
            except Exception as e:
                print(f"검색 오류: {e}")
                continue
        
        if not found_document:
            return {
                "status": "error",
                "message": "요청한 문서를 찾을 수 없습니다.",
                "content": None
            }
        
        # 결과 반환
        doc_source = found_document["_source"]
        content = doc_source.get("text", "")
        
        return {
            "status": "success",
            "message": "문서를 성공적으로 찾았습니다.",
            "content": content,
            "source_metadata": {
                "filename": os.path.basename(source_path),
                "page": doc_source.get("page", page),
                "chunk_id": doc_source.get("chunk_id", chunk_id)
            }
        }
    
def extract_clean_filename(file_path: str) -> str:
    filename = os.path.basename(file_path)
    if '_' in filename:
        parts = filename.split('_', 1)
        if len(parts) > 1 and len(parts[0]) >= 8:
            return parts[1]
    return filename


async def generate_llm_response(
    tokenizer_for_template_application: Any,
    question: str,
    top_docs: List[Document],
    temperature: float = 0.1,
    conversation_history=None,
) -> dict:
    """LLM 답변 생성"""
    
    # 컨텍스트 구성
    context_parts = []
    for doc in top_docs:
        source_path = doc.metadata.get("source", "unknown")
        clean_filename = extract_clean_filename(source_path)
        page_num = doc.metadata.get("page", "")
        
        source_info = f"[{clean_filename} p.{page_num}]" if page_num and page_num > 1 else f"[{clean_filename}]"
        context_parts.append(f"{source_info}: {doc.page_content}")

    context_str = "\\n\\n".join(context_parts)
    
    # 프롬프트 생성
    messages = create_chat_messages(
        question=question,
        context=context_str,
        conversation_history=conversation_history,
        language="ko"
    )
    
    # 템플릿 적용
    if hasattr(tokenizer_for_template_application, 'apply_chat_template'):
        final_prompt_text = tokenizer_for_template_application.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # 수동 ChatML 구성
        chatML_parts = [f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>" for msg in messages]
        chatML_parts.append("<|im_start|>assistant\n")
        final_prompt_text = "\n\n".join(chatML_parts)

    # 소스 메타데이터 추출
    source_metadata = []
    for i, doc in enumerate(top_docs):
        source_path = doc.metadata.get("source", "unknown")
        source_metadata.append({
            "document_id": doc.metadata.get("document_id"),
            "path": source_path,
            "display_name": extract_clean_filename(source_path),
            "page": doc.metadata.get("page", 1),
            "chunk_id": doc.metadata.get("chunk_id", i),
            "score": doc.metadata.get("relevance_score", 0),
        })
    
    return {
        "prompt_text": final_prompt_text,
        "source_metadata": source_metadata,
        "top_docs": top_docs,
        "used_document_ids": [doc.metadata.get("document_id") for doc in top_docs]
    }