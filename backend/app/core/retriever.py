#core/retriever.py

import os
import time
import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from langchain.schema import Document
from app.utils.feedback_analyzer import SearchQualityOptimizer
from app.utils.indexing_utils import ES_INDEX_NAME
from app.utils.qwen3_prompts import create_query_optimization_prompt, create_chat_messages
import logging


logger = logging.getLogger(__name__)

@dataclass
class OptimizedQuery:
    original: str
    primary: str
    alternatives: List[str]
    keywords: List[str]
    confidence: float


class ElasticsearchRetriever:
    
    def __init__(self, es_client: Any, embedding_function: Any, category: str, k=25, 
                 llm_model=None, tokenizer=None, index_name=None):
        self.es_client = es_client
        self.index_name = index_name
        self.embedding_function = embedding_function
        self.k = k
        self.category = category
        
        # 캐시 설정
        self._cache = {}
        self._cache_size = 150
        self._cache_ttl = 7200
        
        # 검색 최적화기
        self.search_optimizer = SearchQualityOptimizer()
        
        # 쿼리 최적화기
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
                print(f"📝 최적화된 쿼리들: {search_queries}")
        
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
        """Qwen 쿼리 최적화"""
        prompt = create_query_optimization_prompt(query, self.category)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        
        # 디바이스 일치시키기
        device = self.llm_model.device
        print(f"🔍 Model device: {device}")
        print(f"🔍 Input device before: {inputs['input_ids'].device}")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"🔍 Input device after: {inputs['input_ids'].device}")
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id, #토큰으로 끝나면 자동종료
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return self._parse_qwen_response(response, query)
    
    def _parse_qwen_response(self, response: str, original_query: str) -> List[str]:
        """Qwen 응답 파싱"""
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if not json_match:
            return [original_query]
            
        data = json.loads(json_match.group(0))
        queries = data.get("queries", [])
        
        if not queries or not isinstance(queries, list):
            return [original_query]
        
        # 원본 + 대안 최대 2개
        result = [original_query]
        for q in queries[:2]:
            if q and q != original_query:
                result.append(q.strip())
        return result
    
    async def _search_single_query(self, query: str, boost_factor: float = 1.0) -> List[Document]:
        """단일 쿼리 검색"""
        TABLE_KWS = ["표", "table", "도표", "차트"]
        wants_table = any(kw in query.lower() for kw in TABLE_KWS)
        # 캐시 확인
        query_normalized = query.lower().strip()
        cache_key = f"{query_normalized}:{self.category}:{boost_factor}"
        current_time = time.time()
        
        cache_entry = self._cache.get(cache_key)
        if cache_entry and current_time - cache_entry["timestamp"] < self._cache_ttl:
            print(f"💾 캐시 사용: '{query[:20]}...'")
            return cache_entry["results"]
        
        # 캐시 정리
        if len(self._cache) >= self._cache_size:
            oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])[:len(self._cache) // 3]
            for old_key in oldest_keys:
                del self._cache[old_key]

        # 임베딩 생성
        query_embedding = self.embedding_function([query_normalized])[0]

        # 하이브리드 검색 쿼리
        hybrid_query = {
        "size": self.k,
        "_source": {"excludes": ["embedding"]},
        "query": {
            "bool": {
                "should": [
                    {"match_phrase": {"text": {"query": query, "boost": 3.5*boost_factor, "slop": 3}}},
                    {"match": {"text": {"query": query, "boost": 2.5*boost_factor, "operator": "OR",
                                        "minimum_should_match": "60%"}}},
                    {"script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.qv, 'embedding') + 1.0",
                            "params": {"qv": query_embedding},
                        },
                        "boost": 2.2*boost_factor
                    }}
                ] + (
                    # 표 키워드가 있을 때만 가중치 ↑
                    [{"term": {"element_type": {"value": "table", "boost": 4.0*boost_factor}}}]
                    if wants_table else []
                ),
                "filter": [
                    {"term": {"category": self.category}}
                ] + (
                    # “표만 보여줘” 같은 질문이면 필터를 강제해도 됨
                    [{"term": {"element_type": "table"}}] if query.strip() in TABLE_KWS else []
                ),
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
            metadata = {k: v for k, v in hit["_source"].items() if k not in ["text", "embedding"]}
            metadata.update({
                "relevance_score": hit["_score"] * boost_factor,
                "source": hit["_source"].get("source", "unknown"),
                "page": hit["_source"].get("page", 1)
            })
            
            chunk_id = hit["_source"].get("chunk_id")
            if chunk_id is not None:
                metadata["chunk_id"] = str(chunk_id)

            docs.append(Document(
                page_content=hit["_source"].get("text", ""), 
                metadata=metadata
            ))

        # 캐싱
        self._cache[cache_key] = {"results": docs, "timestamp": current_time}
        return docs
    
    def _deduplicate_results(self, docs: List[Document]) -> List[Document]:
        """중복 제거"""
        seen = set()
        unique_docs = []
        
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id or chunk_id not in seen:
                if chunk_id:
                    seen.add(chunk_id)
                unique_docs.append(doc)
        
        return unique_docs


async def generate_llm_response(
    tokenizer_for_template_application: Any,
    question: str,
    top_docs: List[Document],
    temperature: float = 0.1,
    conversation_history=None,
) -> dict:
    """LLM 답변 생성"""
    
    def extract_clean_filename(file_path: str) -> str:
        filename = os.path.basename(file_path)
        if '_' in filename:
            parts = filename.split('_', 1)
            if len(parts) > 1 and len(parts[0]) >= 8:
                return parts[1]
        return filename

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
            "path": source_path,
            "display_name": extract_clean_filename(source_path),
            "page": doc.metadata.get("page", 1),
            "chunk_id": doc.metadata.get("chunk_id", i),
            "score": doc.metadata.get("relevance_score", 0),
        })
    
    return {
        "prompt_text": final_prompt_text,
        "source_metadata": source_metadata,
        "top_docs": top_docs
    }