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
import faiss
import numpy as np
from langchain.schema import Document
from app.utils.feedback_analyzer import SearchQualityOptimizer
from app.utils.indexing_utils import ES_INDEX_NAME
from app.utils.qwen3_prompts import create_chat_messages
from app.utils.search_enhancer import QueryExpander
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
        self.query_expander = QueryExpander()
        # 쿼리 최적화기
        if llm_model and tokenizer:
            self.llm_model = llm_model
            self.tokenizer = tokenizer

        self.query_optimizer_enabled = bool(llm_model and tokenizer)
        
        if self.query_optimizer_enabled:
            print("✅ 쿼리 최적화 기능 활성화")
            
        # FAISS 인덱스 초기화
        self.faiss_index = None
        self.doc_embeddings = []
        self.doc_metadata = []
        self._build_faiss_index()

    def get_relevant_documents(self, query: str) -> List[Document]:
        """동기 검색 함수 - 원래 방식으로 복구"""
        # 일단 원래대로 두고, 호출하는 쪽에서 async로 바꾸거나
        # 이 함수를 호출하는 곳을 async가 아닌 환경에서 호출하도록 수정
        return asyncio.run(self.async_get_relevant_documents(query))

    async def async_get_relevant_documents(self, query: str) -> List[Document]:
        """2단계 검색 방식: 1차 검색 후 히트된 문서 내에서 추가 검색"""
        
        if not self.es_client or not self.embedding_function:
            return []

        print(f"🔍 1단계 검색 시작: {query[:30]}...")
        
        # 1단계: 기본 검색
        primary_docs = await self._search_single_query(query, boost_factor=1.0)
        
        if not primary_docs:
            print("❌ 1단계 검색 결과 없음")
            return []
        
        print(f"✅ 1단계 검색 완료: {len(primary_docs)}개 문서")
        
        # 2단계: 히트된 문서들 내에서 추가 검색
        secondary_docs = await self._search_within_hit_documents(query, primary_docs)
        
        # 결과 결합 및 중복 제거
        all_docs = primary_docs + secondary_docs
        unique_docs = self._deduplicate_results(all_docs)
        
        # 점수 기준 정렬
        unique_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        result = unique_docs[:10]  # 상위 10개 문서로 제한하여 컨텍스트 길이 줄이기
        
        print(f"✅ 2단계 검색 완료: 총 {len(result)}개 문서 (1단계: {len(primary_docs)}, 2단계: {len(secondary_docs)})")
        return result
    
    async def _search_within_hit_documents(self, original_query: str, hit_docs: List[Document]) -> List[Document]:
        """2단계: 히트된 문서들 내에서 추가 키워드 검색"""
        
        if not hit_docs:
            return []
        
        print(f"🔍 2단계 검색 시작: {len(hit_docs)}개 히트 문서 내 추가 검색")
        
        # 히트된 문서들에서 소스 파일과 페이지 정보 추출
        hit_sources = set()
        for doc in hit_docs:
            source = doc.metadata.get("source")
            page = doc.metadata.get("page", 1)
            if source:
                hit_sources.add((source, page))
        
        # 쿼리에서 키워드 추출 및 확장
        expanded_keywords = self._extract_and_expand_keywords(original_query)
        
        if not expanded_keywords:
            print("⚠️ 확장 키워드 없음, 2단계 검색 스킵")
            return []
        
        print(f"📝 확장된 키워드: {expanded_keywords}")
        
        # 각 히트 문서의 소스/페이지에서 추가 청크 검색
        secondary_docs = []
        
        for source, page in hit_sources:
            # 동일 문서/페이지 내 다른 청크들 검색
            additional_chunks = await self._search_same_document_chunks(
                source, page, expanded_keywords, original_query
            )
            secondary_docs.extend(additional_chunks)
        # ✨ 2단계 검색 결과 상세 로그
        for chunk in secondary_docs:
            chunk_page = chunk.metadata.get('page', 0)
            chunk_source = chunk.metadata.get('source', '')
            
            if chunk_page == 25 and '사내규정모음집' in chunk_source:
                print(f"🚨🚨 p.25 발견!! 2단계 검색에서 유입됨!")
                print(f"   점수: {chunk.metadata.get('relevance_score', 0):.3f}")
                print(f"   search_stage: {chunk.metadata.get('search_stage', 'unknown')}")
                print(f"   내용 일부: {chunk.page_content[:200]}...")
                print(f"   이 청크가 final_docs에 포함되는지 확인 필요!")
                break
        
        # 점수 조정 (2단계 검색 결과는 약간 낮은 점수)
        for doc in secondary_docs:
            current_score = doc.metadata.get("relevance_score", 0)
            doc.metadata["relevance_score"] = current_score * 0.7  # 2단계 페널티
            doc.metadata["search_stage"] = "secondary"
        
        print(f"✅ 2단계 검색 완료: {len(secondary_docs)}개 추가 문서")
        return secondary_docs
    
    def _extract_and_expand_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출 및 확장 (search_enhancer의 개선된 로직 사용, 저장된 동의어 사전 활용)"""
        
        try:
            # search_enhancer의 QueryExpander 사용
            query_expander = QueryExpander()
            
            # 개선된 키워드 추출
            keywords = query_expander.extract_keywords(query)
            print(f"🔍 개선된 키워드 추출: {len(keywords)}개 - {keywords}")
            return keywords[:6]  # 최대 6개로 제한 확장
            
        except Exception as e:
            print(f"⚠️ 개선된 키워드 추출 실패, 기본 방식 사용: {e}")
            
            # 백업: 기본 키워드 추출
            words = re.findall(r'[가-힣a-zA-Z0-9]{2,}', query)
            
            # 불용어 제거
            stopwords = {
                '이것', '그것', '저것', '무엇', '어떤', '어떻게', '왜', '어디서', '언제', 
                '누구', '어느', '얼마', '없는', '있는', '되는', '하는', '같은', '다른',
                '때문', '위해', '통해', '따라', '의해', '관련', '부분', '내용', '정보',
                '방법', '경우', '때는', '있습니다', '없습니다', '됩니다', '합니다', '대한'
            }
            
            keywords = []
            for word in words:
                if word.lower() not in stopwords and len(word) >= 2:
                    keywords.append(word)
            
            return keywords[:6]  # 기본 키워드만 반환
    
    async def _search_same_document_chunks(
        self, source: str, page: int, keywords: List[str], original_query: str
    ) -> List[Document]:
        """동일 문서/페이지 내에서 키워드로 추가 청크 검색"""
        
        try:
            # 키워드들로 should 쿼리 구성
            should_clauses = []
            
            for keyword in keywords[:10]:  # 상위 10개 키워드만 사용
                should_clauses.extend([
                    {"match": {"text": {"query": keyword, "boost": 2.5}}},
                    {"match": {"caption": {"query": keyword, "boost": 2.2}}},
                    {"match": {"table_caption": {"query": keyword, "boost": 2.0}}},
                    {"wildcard": {"text": {"value": f"*{keyword}*", "boost": 1.0}}}
                ])
            
            # 동일 문서/페이지 필터
            must_clauses = [
                {"term": {"source": source}},
                {"term": {"page": page}},
                {"term": {"category": self.category}}
            ]
            
            query = {
                "size": 8,  # 추가로 가져올 청크 수
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                }
            }
            
            response = self.es_client.search(index=self.index_name, body=query, request_timeout=15)
            
            # 결과 처리
            docs = []
            for hit in response["hits"]["hits"]:
                meta_src = hit["_source"]
                metadata = {k: v for k, v in meta_src.items() if k not in ["text", "embedding"]}
                metadata.update({
                    "document_id": hit["_id"],
                    "relevance_score": hit["_score"],
                    "source": meta_src.get("source", source),
                    "page": meta_src.get("page", page),
                    "table_id": meta_src.get("table_id"),
                    "row_no": meta_src.get("row_no"),
                    "row_index": meta_src.get("row_index"),
                    "hit_content": meta_src.get("text", ""),
                    "element_type": meta_src.get("element_type", "text"),
                })
                
                if chunk_id := meta_src.get("chunk_id"):
                    metadata["chunk_id"] = str(chunk_id)
                
                docs.append(Document(page_content=meta_src.get("text", ""), metadata=metadata))
            
            return docs
            
        except Exception as e:
            print(f"⚠️ 동일 문서 내 검색 오류: {e}")
            return []
    
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

        # 키워드 추출 (조합 검색용)
        extracted_keywords = self.query_expander.extract_keywords(query_normalized)


        # 하이브리드 검색 쿼리 구성 (조합 검색 추가)
        base_should = [
            # BM25 / phrase match (부스팅 조정: 10)
            {"match_phrase": {"text": {"query": query, "boost": 10.0 * boost_factor, "slop": 3}}},
            {"match": {
                "text": {"query": query, "boost": 8.0 * boost_factor,
                        "operator": "OR", "minimum_should_match": "60%"}
            }},
            # 기존 캡션 필드들 활용 (부스팅 조정: 8)
            {"match": {
                "caption": {"query": query, "boost": 150.0 * boost_factor,
                        "operator": "OR", "minimum_should_match": "10%"}
            }},
            {"match": {
                "table_caption": {"query": query, "boost": 150.0 * boost_factor,
                                "operator": "OR", "minimum_should_match": "10%"}
            }},
        ]

        # 키워드 조합 검색 추가
        if len(extracted_keywords) >= 2:
            print(f"🔍 키워드 조합 검색 활성화: {len(extracted_keywords)}개 키워드")
            
            # 기본 키워드 AND 조합
            keyword_query = " AND ".join(extracted_keywords[:4])  # 상위 4개만 사용 (너무 길면 매칭 어려움)
            combo_should = [
                {"match": {
                    "text": {
                        "query": keyword_query,
                        "operator": "AND",
                        "boost": 5.0 * boost_factor,  # 높은 부스팅
                        "minimum_should_match": "80%"  # 80% 이상 매칭
                    }
                }}
            ]
            base_should.extend(combo_should)

        # 경조사 특화 조합 검색
        if any(kw in extracted_keywords for kw in ['경조사', '지원', '부모', '회갑', '환갑']):

            economic_combo = [
                # 경조사 + 지급 조합 (매우 높은 부스팅)
                {"bool": {
                    "must": [
                        {"match": {"text": {"query": "회갑 OR 환갑", "boost": 1.0}}},
                        {"match": {"text": {"query": "부모 OR 아버지", "boost": 1.0}}}
                    ],
                    "boost": 15.0 * boost_factor  # 최고 부스팅
                }},
                # 금액 명시 검색
                {"match": {
                    "text": {
                        "query": "30만원 OR 300000 OR 삼십만원",
                        "boost": 5.0 * boost_factor
                    }
                }}
            ]
            base_should.extend(economic_combo)

        # 표 부스팅 부분
        table_boost = []
        if wants_table:
            table_boost = [
                {"terms": {"element_type": ["table", "table_row"], "boost": 8.0 * boost_factor}}
            ]

        should_clauses = base_should + table_boost


        base_filter = [{"term": {"category": self.category}}]

        table_filter = []
        if query.strip() in TABLE_KWS:
            # 표만 달라는 쿼리라면 table + table_row 로 한정
            table_filter = [{"terms": {"element_type": ["table", "table_row"]}}]
        filter_clauses = base_filter + table_filter

        hybrid_query = {
            "size": self.k * 3,  # 더 많은 후보를 가져와서 다양한 결과를 포함
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

        # 결과 처리 (BM25 점수)
        docs = []
        for hit in response["hits"]["hits"]:
            meta_src = hit["_source"]
            metadata = {k: v for k, v in meta_src.items() if k not in ["text", "embedding"]}
            metadata.update(
                {
                    "document_id": hit["_id"],
                    "bm25_score": hit["_score"] * boost_factor,
                    "source": meta_src.get("source", "unknown"),
                    "page": meta_src.get("page", 1),
                    # table_row일 경우 위치 추적용
                    "table_id": meta_src.get("table_id"),
                    "row_no": meta_src.get("row_no"),
                    "row_index": meta_src.get("row_index"),  # 테이블 행 인덱스 추가
                    # 정확한 hit 내용 저장 (하이라이트용)
                    "hit_content": meta_src.get("text", ""),  # 검색된 정확한 chunk 내용
                    "element_type": meta_src.get("element_type", "text"),  # chunk 타입
                }
            )
            if chunk_id := meta_src.get("chunk_id"):
                metadata["chunk_id"] = str(chunk_id)

            docs.append(Document(page_content=meta_src.get("text", ""), metadata=metadata))

        for i, doc in enumerate(docs):
            if doc.metadata.get('page') == 25 and '사내규정모음집' in doc.metadata.get('source', ''):
                score = doc.metadata.get('bm25_score', 0)
                print(f"  ES 직후 p.25: {i+1}위 (bm25_score: {score:.6f})")
        # FAISS를 사용한 벡터 검색
        faiss_docs = []
        if self.faiss_index is not None and self.doc_metadata:
            query_emb = np.array(query_embedding, dtype=np.float32)
            D, I = self.faiss_index.search(query_emb.reshape(1, -1), self.k)
            for i, idx in enumerate(I[0]):
                if idx >= 0 and idx < len(self.doc_metadata):
                    metadata = self.doc_metadata[idx].copy()
                    # FAISS 스코어를 하이라이트에 반영하기 위해 거리(L2)를 유사도로 변환 (1 / (1 + 거리))
                    metadata['faiss_score'] = 1.0 / (1.0 + float(D[0][i]))
                    faiss_docs.append(Document(page_content=self.doc_metadata[idx].get('text', ''), metadata=metadata))

        # BM25와 FAISS 결과 결합
        combined_docs = []
        seen_docs = set()
        for doc in docs:
            doc_id = doc.metadata.get("document_id")
            if doc_id not in seen_docs:
                combined_docs.append(doc)
                seen_docs.add(doc_id)
            else:
                for c_doc in combined_docs:
                    if c_doc.metadata.get("document_id") == doc_id:
                        c_doc.metadata['bm25_score'] = max(c_doc.metadata.get('bm25_score', 0), doc.metadata.get('bm25_score', 0))
                        break

        for doc in faiss_docs:
            doc_id = doc.metadata.get("document_id")
            if doc_id not in seen_docs:
                combined_docs.append(doc)
                seen_docs.add(doc_id)
            else:
                for c_doc in combined_docs:
                    if c_doc.metadata.get("document_id") == doc_id:
                        c_doc.metadata['faiss_score'] = max(c_doc.metadata.get('faiss_score', 0), doc.metadata.get('faiss_score', 0))
                        break

        # 최종 점수 계산 - 안전한 정규화 (오류 방지)
        bm25_scores = [doc.metadata.get('bm25_score', 0) for doc in combined_docs]
        faiss_scores = [doc.metadata.get('faiss_score', 0) for doc in combined_docs]

        # 안전한 최대값 계산 (0 방지)
        max_bm25 = max(bm25_scores) if bm25_scores and any(s > 0 for s in bm25_scores) else 1.0
        max_faiss = max(faiss_scores) if faiss_scores and any(s > 0 for s in faiss_scores) else 1.0

        print(f"🔍 점수 범위 - BM25: 0~{max_bm25:.3f}, FAISS: 0~{max_faiss:.3f}")

        # 정규화된 점수 계산 (안전한 나누기)
        normalized_scores = []
        for doc in combined_docs:
            # 안전한 정규화 (0 나누기 방지)
            bm25_score = doc.metadata.get('bm25_score', 0) / max_bm25 if max_bm25 > 0 else 0
            faiss_score = doc.metadata.get('faiss_score', 0) / max_faiss if max_faiss > 0 else 0
            
            # 가중 평균 계산
            if faiss_score > 0:
                combined_score = bm25_score * 0.7 + faiss_score * 0.3
            else:
                combined_score = bm25_score * 0.7  # BM25만 있는 경우
            
            # 점수 범위 보정 (0~1 범위 보장)
            combined_score = max(0.0, min(1.0, combined_score))
            
            normalized_scores.append(combined_score)
            doc.metadata['relevance_score'] = combined_score
        print(f"🔍 FAISS 결합 후 p.25 확인:")
        for i, doc in enumerate(combined_docs):
            if doc.metadata.get('page') == 25 and '사내규정모음집' in doc.metadata.get('source', ''):
                bm25 = doc.metadata.get('bm25_score', 0)
                faiss = doc.metadata.get('faiss_score', 0)
                print(f"  결합 후 p.25: {i+1}위 (bm25: {bm25:.6f}, faiss: {faiss:.6f})")
        combined_docs = self._apply_document_boosting(combined_docs, query_normalized)     
        # 점수 기준 정렬
        combined_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        print(f"🔍 부스팅 후 p.25 확인:")
        for i, doc in enumerate(combined_docs):
            if doc.metadata.get('page') == 25 and '사내규정모음집' in doc.metadata.get('source', ''):
                score = doc.metadata.get('relevance_score', 0)
                print(f"  부스팅 후 p.25: {i+1}위 (score: {score:.6f})")
        # 캐싱
        self._cache[cache_key] = {"results": combined_docs, "timestamp": current_time}
        return combined_docs
    
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
        p25_after = [d for d in unique_docs if d.metadata.get('page') == 25 and '사내규정모음집' in d.metadata.get('source', '')]

        return unique_docs
    def _apply_document_boosting(self, docs: List[Document], query: str) -> List[Document]:
        """사내규정모음집 부스팅 + 소프트맥스 정규화 (조정된 부스팅 로직)"""
        
        # 1. 사내규정모음집 부스팅 - 쿼리에 관련 키워드가 포함된 경우에만 적용 및 부스팅 배수 조정
        query_lower = query.lower()
        for doc in docs:
            content = doc.page_content.lower()
            source = doc.metadata.get('source', '')
            
            if '사내규정모음집' in source and ('회갑' in content or '환갑' in content) and '사내규정' in query_lower:
                current_score = doc.metadata.get("relevance_score", 0.0)
                doc.metadata["relevance_score"] = current_score * 5.0  # 부스팅 배수를 50에서 5로 낮춤
        
        # 2. 소프트맥스 정규화
        import torch
        import torch.nn.functional as F
        
        scores = [doc.metadata.get("relevance_score", 0.0) for doc in docs]
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        temperature = 0.3
        softmax_scores = F.softmax(scores_tensor / temperature, dim=0).tolist()
        
        for doc, soft_score in zip(docs, softmax_scores):
            doc.metadata["relevance_score"] = soft_score
        
        return docs
       
    def _build_faiss_index(self):
        """FAISS 인덱스 구축"""
        try:
            print("🔄 FAISS 인덱스 구축 시작...")
            # Elasticsearch에서 모든 문서의 임베딩 가져오기
            query = {
                "size": 10000,  # 최대 문서 수 제한
                "_source": ["embedding", "text", "source", "page", "chunk_id", "document_id", "element_type", "table_id", "row_no", "row_index"],
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"category": self.category}}
                        ]
                    }
                }
            }
            
            response = self.es_client.search(index=self.index_name, body=query, request_timeout=60)
            hits = response.get("hits", {}).get("hits", [])
            
            if not hits:
                print("⚠️ Elasticsearch에서 문서를 가져오지 못했습니다.")
                return
            
            dimension = len(hits[0]["_source"].get("embedding", []))
            if dimension == 0:
                print("⚠️ 임베딩 데이터가 없습니다.")
                return
                
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.doc_embeddings = []
            self.doc_metadata = []
            
            for hit in hits:
                embedding = hit["_source"].get("embedding", [])
                if embedding:
                    self.doc_embeddings.append(np.array(embedding, dtype=np.float32))
                    metadata = {k: v for k, v in hit["_source"].items() if k != "embedding"}
                    metadata.update({
                        "document_id": hit["_id"],
                        "hit_content": hit["_source"].get("text", ""),
                    })
                    self.doc_metadata.append(metadata)
            
            if self.doc_embeddings:
                self.faiss_index.add(np.array(self.doc_embeddings))
                print(f"✅ FAISS 인덱스 구축 완료: {len(self.doc_embeddings)}개 문서")
            else:
                print("⚠️ 임베딩 데이터가 없습니다.")
                self.faiss_index = None
                
        except Exception as e:
            print(f"⚠️ FAISS 인덱스 구축 중 오류 발생: {e}")
            self.faiss_index = None


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
    """파일명에서 UUID/해시 접두어를 제거 (indexing_utils의 strip_uuid_prefix와 동일한 로직)"""
    from app.utils.indexing_utils import strip_uuid_prefix
    return strip_uuid_prefix(file_path)


async def generate_llm_response(
    tokenizer_for_template_application: Any,
    question: str,
    top_docs: List[Document],
    temperature: float = 0.1,
    conversation_history=None,
) -> dict:
    """LLM 답변 생성"""
    conversation_history = []
    # 컨텍스트 구성 - 각 문서를 명확하게 구분
    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        source_path = doc.metadata.get("source", "unknown")
        clean_filename = extract_clean_filename(source_path)
        page_num = doc.metadata.get("page", "")
        
        source_info = f"[{clean_filename} p.{page_num}]" if page_num and page_num > 1 else f"[{clean_filename}]"
        
        # 각 문서를 명확하게 구분하여 할루시네이션 방지
        document_block = f"""==== 문서 {i}: {source_info} ====
{doc.page_content}
==== 문서 {i} 끝 ===="""
        
        context_parts.append(document_block)

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
        
        # 이미지 정보 추출 및 처리
        has_images = doc.metadata.get("has_images", False)
        images = doc.metadata.get("images", [])
        image_captions = doc.metadata.get("image_captions", [])
        
        # 디버깅을 위한 로그
        if has_images or images:
            print(f"📸 이미지 발견: {extract_clean_filename(source_path)} - has_images: {has_images}, images: {len(images) if images else 0}")
        
        # 이미지 정보를 구조화하여 프론트엔드에서 사용하기 쉽게 처리
        processed_images = []
        if has_images and images:
            for idx, img_path in enumerate(images):
                img_info = {
                    "path": img_path,
                    "url": f"/api/image-viewer/{img_path}",
                    "caption": "",
                    "ocr_text": "",
                    "ai_description": ""
                }
                
                # image_captions에서 해당 이미지 정보 찾기
                if image_captions and idx < len(image_captions):
                    caption_info = image_captions[idx]
                    if isinstance(caption_info, dict):
                        img_info["caption"] = caption_info.get("ai_caption", "")
                        img_info["ocr_text"] = caption_info.get("ocr_text", "")
                        img_info["ai_description"] = caption_info.get("ai_caption", "")
                    elif isinstance(caption_info, str):
                        img_info["caption"] = caption_info
                
                # 기본 캡션이 없으면 생성
                if not img_info["caption"]:
                    img_info["caption"] = f"페이지 {doc.metadata.get('page', 1)} 이미지 {idx + 1}"
                
                processed_images.append(img_info)
        
        # 이미지가 없는 경우에도 확실하게 False로 설정
        if not images or len(images) == 0:
            has_images = False
        
        source_metadata.append({
            "document_id": doc.metadata.get("document_id"),
            "path": source_path,
            "display_name": extract_clean_filename(source_path),
            "page": doc.metadata.get("page", 1),
            "chunk_id": doc.metadata.get("chunk_id", i),
            "score": doc.metadata.get("relevance_score", 0),
            "has_images": has_images,
            "images": images,
            "processed_images": processed_images,
            "image_count": len(processed_images)
        })
    
    # 질문과 관련성이 높은 문서를 우선적으로 출처로 포함
    final_source_metadata = []
    question_lower = question.lower()
    for meta in source_metadata:
        display_name = meta["display_name"].lower()
        if any(keyword in display_name for keyword in question_lower.split()):
            final_source_metadata.append(meta)
        if len(final_source_metadata) >= 3:  # 최대 3개로 제한
            break
    
    
    return {
        "prompt_text": final_prompt_text,
        "source_metadata": final_source_metadata,
        "top_docs": top_docs,
        "used_document_ids": [meta["document_id"] for meta in final_source_metadata]
    }
