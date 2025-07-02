"""
쿼리 최적화 및 검색 품질 향상을 위한 유틸리티 모듈
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from langchain.schema import Document
from app.utils.synonym_builder import get_qwen_synonym_builder


class QueryExpander:
    """
    사용자 쿼리를 확장하여 검색 품질을 높이는 클래스
    """
    
    def __init__(self):
        # 한국어 불용어 목록
        self.stopwords = {
            "이", "가", "을", "를", "은", "는", "에", "에서", "로", "으로", 
            "과", "와", "도", "의", "들", "좀", "등", "및", "그", "저", 
            "것", "수", "알려줘", "궁금해", "대한", "대해", "내용", 
            "무엇인가요", "뭔가요", "뭐야", "설명해줘", "알고싶어"
        }
        
    def clean_query(self, query: str) -> str:
        """기본적인 쿼리 정제"""
        # 특수문자 제거 (단, 한글/영문/숫자/공백 유지)
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', query)
        # 여러 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
        
    def extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 핵심 키워드 추출"""
        # 불용어 제거 및 길이 1 이하 단어 제거
        tokens = query.split()
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        return keywords
        
    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 확장 및 변형 생성 (동의어 사전 활용)
        """
        cleaned_query = self.clean_query(query)
        keywords = self.extract_keywords(cleaned_query)
        
        # 동의어 빌더 가져오기
        try:
            synonym_builder = get_qwen_synonym_builder()
        except Exception as e:
            print(f"동의어 빌더 로드 실패: {e}")
            synonym_builder = None
        
        # 확장된 쿼리 생성
        expanded_variants = []
        
        # 1. 원본 쿼리 (항상 포함)
        expanded_variants.append(query.strip())
        
        # 2. 동의어 기반 쿼리 변형 (가장 중요)
        if synonym_builder and keywords:
            # 각 키워드의 동의어 찾기
            keyword_synonyms = {}
            for keyword in keywords:
                synonyms = synonym_builder.get_synonyms(keyword)
                if len(synonyms) > 1:  # 동의어가 실제로 있는 경우
                    keyword_synonyms[keyword] = [s for s in synonyms if s != keyword][:3]  # 최대 3개
            
            # 동의어로 치환한 쿼리 생성
            if keyword_synonyms:
                # 각 키워드를 동의어로 치환한 버전들 생성
                for original_keyword, synonyms in keyword_synonyms.items():
                    for synonym in synonyms:
                        # 원본 쿼리에서 키워드를 동의어로 치환
                        synonym_query = re.sub(
                            rf'\b{re.escape(original_keyword)}\b', 
                            synonym, 
                            query, 
                            flags=re.IGNORECASE
                        )
                        if synonym_query != query and synonym_query.strip():
                            expanded_variants.append(synonym_query.strip())
                
                # 여러 키워드를 동시에 치환한 조합 생성
                if len(keyword_synonyms) >= 2:
                    # 가장 중요한 2개 키워드의 첫 번째 동의어로 치환
                    combination_query = query
                    replacement_count = 0
                    for original_keyword, synonyms in list(keyword_synonyms.items())[:2]:
                        if synonyms:
                            combination_query = re.sub(
                                rf'\b{re.escape(original_keyword)}\b', 
                                synonyms[0], 
                                combination_query, 
                                flags=re.IGNORECASE
                            )
                            replacement_count += 1
                    
                    if replacement_count >= 2 and combination_query != query:
                        expanded_variants.append(combination_query.strip())
        
        # 3. 핵심 키워드만 포함한 쿼리 (동의어 확장된 키워드 포함)
        if keywords:
            # 원본 키워드 + 동의어 조합
            expanded_keywords = []
            if synonym_builder:
                for keyword in keywords:
                    synonyms = synonym_builder.get_synonyms(keyword)
                    expanded_keywords.extend(synonyms[:2])  # 각 키워드당 최대 2개 동의어
            else:
                expanded_keywords = keywords
            
            # 중복 제거하고 키워드 쿼리 생성
            unique_keywords = list(dict.fromkeys(expanded_keywords))
            if len(unique_keywords) >= 2:
                keyword_query = " ".join(unique_keywords[:4])  # 최대 4개 키워드
                if keyword_query != query.strip():
                    expanded_variants.append(keyword_query)
        
        # 4. 질문 형태 변환 (의문문 -> 평서문)
        transformed_queries = []
        for variant in expanded_variants:
            if "?" in variant:
                # 다양한 질문 패턴 처리
                statement_query = variant
                statement_query = re.sub(r'\?+$', '', statement_query)
                statement_query = re.sub(r'까\?*$', '', statement_query)
                statement_query = re.sub(r'니\?*$', '', statement_query)
                statement_query = re.sub(r'나\?*$', '', statement_query)
                statement_query = statement_query.strip()
                
                if statement_query and statement_query != variant:
                    transformed_queries.append(statement_query)
        
        expanded_variants.extend(transformed_queries)
        
        # 5. 어미 변형 (추가 변형)
        morphology_variants = []
        for variant in expanded_variants[:3]:  # 처음 3개 변형에 대해서만
            # 높임말 -> 평어 변환
            morphed = variant
            morphed = re.sub(r'해주세요$', '', morphed)
            morphed = re.sub(r'해주니$', ' 지급', morphed)
            morphed = re.sub(r'습니다$', '다', morphed)
            morphed = re.sub(r'니까$', '', morphed)
            morphed = morphed.strip()
            
            if morphed and morphed != variant and len(morphed) > 2:
                morphology_variants.append(morphed)
        
        expanded_variants.extend(morphology_variants)
        
        # 중복 제거 및 빈 문자열 제거
        unique_variants = []
        seen = set()
        for variant in expanded_variants:
            variant = variant.strip()
            if variant and variant not in seen and len(variant) > 1:
                unique_variants.append(variant)
                seen.add(variant)
        
        return {
            "original": query,
            "cleaned": cleaned_query,
            "keywords": keywords,
            "variants": unique_variants[:5]  # 최대 5개 변형만 사용
        }


class SearchScoreEnhancer:
    """
    검색 결과 점수를 다양한 요소를 고려하여 보정하는 클래스
    """
    
    def __init__(self, 
                 keyword_match_boost: float = 0.15,
                 exact_match_boost: float = 0.2,
                 title_match_boost: float = 0.25,
                 recency_weight: float = 0.05):
        self.keyword_match_boost = keyword_match_boost  # 키워드 매치 가중치
        self.exact_match_boost = exact_match_boost  # 정확한 문구 매치 가중치
        self.title_match_boost = title_match_boost  # 제목 매치 가중치
        self.recency_weight = recency_weight  # 최신성 가중치
    
    def enhance_scores(self, 
                       documents: List[Document], 
                       query_info: Dict[str, Any]) -> List[Document]:
        """
        검색 결과 문서들의 점수를 다양한 요소를 고려하여 보정
        """
        if not documents:
            return []
            
        original_query = query_info["original"]
        keywords = query_info["keywords"]
        
        # 검색어와 키워드 기반 정규식 생성
        keyword_patterns = [re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE) for kw in keywords]
        exact_pattern = re.compile(re.escape(original_query), re.IGNORECASE)
        
        # 결과 점수 보정
        for doc in documents:
            # 기본 관련성 점수 (기존 ES 또는 리랭커 점수)
            base_score = doc.metadata.get("relevance_score", 0)
            
            # 텍스트 컨텐츠
            content = doc.page_content
            
            # 1. 키워드 매치 점수 계산
            keyword_matches = sum(1 for pattern in keyword_patterns if pattern.search(content))
            keyword_boost = min(1.0, (keyword_matches / len(keywords)) * self.keyword_match_boost) if keywords else 0
            
            # 2. 정확한 문구 매치 점수
            exact_matches = len(exact_pattern.findall(content))
            exact_boost = min(self.exact_match_boost, exact_matches * 0.05)
            
            # 3. 문서 메타데이터 고려 (제목 매치 등)
            title_boost = 0
            if "title" in doc.metadata and doc.metadata["title"]:
                title = doc.metadata["title"]
                title_keyword_matches = sum(1 for pattern in keyword_patterns if pattern.search(title))
                if title_keyword_matches > 0:
                    title_boost = min(self.title_match_boost, (title_keyword_matches / len(keywords)) * self.title_match_boost)
            
            # 4. 최신성 점수 (indexed_at 필드 있을 경우)
            recency_boost = 0
            if "indexed_at" in doc.metadata and doc.metadata["indexed_at"]:
                try:
                    indexed_time = time.mktime(time.strptime(doc.metadata["indexed_at"], "%Y-%m-%dT%H:%M:%S.%f"))
                    now = time.time()
                    days_diff = (now - indexed_time) / (86400)  # 86400 = 1일 초단위
                    recency_boost = self.recency_weight * max(0, min(1, 1 - (days_diff / 365)))  # 1년 내 문서는 보너스
                except:
                    pass
                    
            # 최종 보정 점수 계산 및 적용
            boost_factor = 1 + keyword_boost + exact_boost + title_boost + recency_boost
            enhanced_score = base_score * boost_factor
            
            # 메타데이터에 점수 업데이트
            doc.metadata["original_score"] = base_score
            doc.metadata["enhanced_score"] = enhanced_score
            doc.metadata["boost_factor"] = boost_factor
            doc.metadata["keyword_boost"] = keyword_boost
            doc.metadata["exact_boost"] = exact_boost
            doc.metadata["title_boost"] = title_boost
            doc.metadata["recency_boost"] = recency_boost
            
            # 점수 필드 업데이트
            doc.metadata["relevance_score"] = enhanced_score
        
        # 보정된 점수로 재정렬
        documents.sort(key=lambda x: x.metadata.get("enhanced_score", 0), reverse=True)
        
        return documents


class ContextualReranker:
    """
    검색 문서를 컨텍스트 기반으로 재순위화하는 클래스
    """
    
    def __init__(self, 
                 diversity_weight: float = 0.1,
                 coherence_weight: float = 0.2,
                 max_docs: int = 15):
        self.diversity_weight = diversity_weight
        self.coherence_weight = coherence_weight
        self.max_docs = max_docs
        
    def rerank_with_diversity(self, 
                              documents: List[Document],
                              query_info: Dict[str, Any]) -> List[Document]:
        """
        다양성과 일관성을 고려하여 문서 재순위화
        - 너무 유사한 문서들은 중복 제거
        - 다양한 소스와 시점의 문서들이 포함되도록 함
        """
        if not documents:
            return []
            
        # 원본 문서 복사 및 초기 점수 저장
        ranked_docs = []
        remaining_docs = documents.copy()
        selected_sources = set()
        
        # 최고 점수 문서는 무조건 포함
        if remaining_docs:
            top_doc = remaining_docs.pop(0)
            ranked_docs.append(top_doc)
            if "source" in top_doc.metadata:
                selected_sources.add(top_doc.metadata["source"])
        
        # MMR(Maximal Marginal Relevance) 방식 적용
        while remaining_docs and len(ranked_docs) < self.max_docs:
            best_score = -1
            best_idx = -1
            
            for i, doc in enumerate(remaining_docs):
                # 기본 관련성 점수
                rel_score = doc.metadata.get("enhanced_score", 0)
                
                # 다양성 점수 계산: 이미 선택된 문서와의 유사성 확인
                source_diversity = 1.0
                if "source" in doc.metadata and doc.metadata["source"] in selected_sources:
                    source_diversity = 0.7  # 같은 소스는 약간 페널티
                
                # 일관성 점수: 이미 선택된 문서들과의 내용 관련성
                coherence_score = 1.0
                
                # 최종 점수 계산 (관련성 + 다양성 + 일관성)
                score = rel_score * (
                    (1 - self.diversity_weight - self.coherence_weight) + 
                    self.diversity_weight * source_diversity +
                    self.coherence_weight * coherence_score
                )
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx != -1:
                # 최고 점수 문서 선택
                selected_doc = remaining_docs.pop(best_idx)
                ranked_docs.append(selected_doc)
                
                # 소스 추적
                if "source" in selected_doc.metadata:
                    selected_sources.add(selected_doc.metadata["source"])
            else:
                break
        
        return ranked_docs


class EnhancedSearchPipeline:
    """
    개선된 검색 파이프라인 클래스
    - 쿼리 확장
    - 점수 보정
    - 컨텍스트 기반 리랭킹
    """
    
    def __init__(self):
        self.query_expander = QueryExpander()
        self.score_enhancer = SearchScoreEnhancer()
        self.contextual_reranker = ContextualReranker()
        
    def process(self, 
                query: str, 
                search_results: List[Document]) -> Tuple[Dict[str, Any], List[Document]]:
        """
        검색 결과 개선 파이프라인 실행
        """
        # 1. 쿼리 확장 및 변형
        query_info = self.query_expander.expand_query(query)
        
        # 2. 검색 결과 점수 보정
        enhanced_results = self.score_enhancer.enhance_scores(search_results, query_info)
        
        # 3. 컨텍스트 기반 리랭킹
        reranked_results = self.contextual_reranker.rerank_with_diversity(enhanced_results, query_info)
        
        return query_info, reranked_results
