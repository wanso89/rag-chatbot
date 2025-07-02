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
        """쿼리에서 핵심 키워드 추출 (도메인 특화 개선, 동의어 사전 활용 활성화)"""
        # 복합어 및 도메인 특화 키워드 사전
        compound_keywords = {
            r'\d+년\s*이상': lambda m: m.group().replace(' ', ''),  # "1년 이상" -> "1년이상"
            r'\d+개월\s*이상': lambda m: m.group().replace(' ', ''),  # "6개월 이상" -> "6개월이상"
            r'회갑|환갑|칠순|팔순|고희|생신|잔치|축하|행사': '경조사',  # 생일 관련 -> 경조사
            r'결혼식|결혼|혼례|웨딩': '경조사',  # 결혼 관련 -> 경조사
            r'장례식|장례|초상|상|부고|사망': '경조사',  # 장례 관련 -> 경조사
            r'출산|출산휴가|육아휴직|아기|신생아': '출산육아',  # 출산 관련
            r'지원금|지원은|얼마|금액|보조|보조금|수당|혜택|복지|경조사비|만원|정액지급': '지원',  # 지원 관련
            r'직원|근로자|사원|임직원|종업원|근무자': '직원',  # 직원 관련 통일
            r'급여|임금|월급|연봉|보수|수입': '급여',  # 급여 관련 통일
            r'휴가|휴직|휴일|연차|병가|특별휴가|청원휴가|특별청원휴가': '휴가',  # 휴가 관련 통일
            r'자녀|유\s*자녀|자식|아들|딸': '자녀',  # 자녀 관련 통일
            r'조부모|조\s*부\s*모|할아버지|할머니|외조부|외조모': '조부모',  # 조부모 관련 통일
            r'장인|장모|장인장모|장인\s*장모|시아버지|시어머니': '장인장모',  # 장인장모 관련 통일
            r'아버지|어머니|아빠|엄마|부모|부모님': '부모',  # 부모 관련 통일
            r'백숙|백\s*숙|백숙부모|백\s*숙\s*부\s*모|삼촌|이모|고모|숙모': '백숙부모',  # 백숙부모 관련 통일
            r'배우자|배\s*우\s*자|남편|아내': '배우자',  # 배우자 관련 통일
            r'형제자매|형제\s*자매|형|오빠|누나|동생|남매': '형제자매',  # 형제자매 관련 통일
            r'승중|승\s*중|승중상|승\s*중\s*상': '승중상',  # 승중상 관련 통일
        }
        
        # 1. 복합어 및 도메인 특화 키워드 추출
        domain_keywords = []
        processed_query = query
        
        for pattern, replacement in compound_keywords.items():
            matches = re.finditer(pattern, processed_query, re.IGNORECASE)
            for match in matches:
                if callable(replacement):
                    keyword = replacement(match)
                else:
                    keyword = replacement
                
                if keyword not in domain_keywords:
                    domain_keywords.append(keyword)
                
                # 매치된 부분을 마커로 대체하여 중복 추출 방지
                processed_query = processed_query.replace(match.group(), f' _PROCESSED_{len(domain_keywords)}_ ')
        
        # 2. 기본 토큰화 (남은 부분에 대해)
        tokens = processed_query.split()
        basic_keywords = []
        
        for token in tokens:
            # 처리된 마커는 무시
            if token.startswith('_PROCESSED_'):
                continue
                
            # 불용어 제거 및 길이 조건
            if token not in self.stopwords and len(token) > 1:
                # 숫자+단위 조합 보존
                if re.match(r'\d+[년개월일주시간]', token):
                    basic_keywords.append(token)
                # 일반 키워드
                elif not token.isdigit():  # 단순 숫자는 제외
                    basic_keywords.append(token)
        
        # 3. 최종 키워드 조합 (도메인 키워드 우선)
        final_keywords = domain_keywords + basic_keywords
        
        # 중복 제거하면서 순서 보존
        unique_keywords = []
        seen = set()
        for kw in final_keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        
        # 4. 동의어 사전 활용 활성화
        print("🔍 동의어 사전 활용 활성화됨")
        synonym_builder = get_qwen_synonym_builder()
        expanded_keywords = []
        for kw in unique_keywords:
            synonyms = synonym_builder.get_synonyms(kw)
            if synonyms:
                expanded_keywords.extend(synonyms)
            else:
                expanded_keywords.append(kw)
        
        # 중복 제거하면서 순서 보존
        final_expanded_keywords = []
        seen = set()
        for kw in expanded_keywords:
            if kw not in seen:
                final_expanded_keywords.append(kw)
                seen.add(kw)
        
        print(f"🔍 개선된 키워드 추출: {len(unique_keywords)}개 - {unique_keywords}")
        print(f"🔍 동의어 사전 확장: {len(unique_keywords)} → {len(final_expanded_keywords)}개")
        print(f"📝 확장된 키워드: {final_expanded_keywords}")
        return final_expanded_keywords
        
    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 확장 및 변형 생성 (동의어 사전 활용 활성화)
        """
        cleaned_query = self.clean_query(query)
        keywords = self.extract_keywords(cleaned_query)
        
        # 확장된 쿼리 생성
        must_variants = []
        should_variants = []
        
        # 1. 원본 쿼리 (항상 포함, 'must' 방식으로 높은 가중치 부여)
        must_variants.append(query.strip())
        
        # 2. 핵심 키워드만 포함한 쿼리 ('must' 방식)
        if keywords:
            # 중복 제거하고 키워드 쿼리 생성
            unique_keywords = list(dict.fromkeys(keywords))
            if len(unique_keywords) >= 2:
                keyword_query = " ".join(unique_keywords[:4])  # 최대 4개 키워드
                if keyword_query != query.strip():
                    must_variants.append(keyword_query)
        
        # 3. 질문 형태 변환 (의문문 -> 평서문, 'must' 방식)
        transformed_queries = []
        for variant in must_variants:
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
        
        must_variants.extend(transformed_queries)
        
        # 4. 어미 변형 (추가 변형, 'should' 방식)
        morphology_variants = []
        for variant in must_variants[:3]:  # 처음 3개 변형에 대해서만
            # 높임말 -> 평어 변환
            morphed = variant
            morphed = re.sub(r'해주세요$', '', morphed)
            morphed = re.sub(r'해주니$', ' 지급', morphed)
            morphed = re.sub(r'습니다$', '다', morphed)
            morphed = re.sub(r'니까$', '', morphed)
            morphed = morphed.strip()
            
            if morphed and morphed != variant and len(morphed) > 2:
                morphology_variants.append(morphed)
        
        should_variants.extend(morphology_variants)
        
        # 5. 동의어 기반 변형 ('should' 방식)
        synonym_builder = get_qwen_synonym_builder()
        for kw in unique_keywords:
            synonyms = synonym_builder.get_synonyms(kw)
            if synonyms and len(synonyms) > 1:
                for syn in synonyms[1:]:  # 첫 번째는 원본이므로 제외
                    syn_query = query.replace(kw, syn)
                    if syn_query != query:
                        should_variants.append(syn_query)
        
        # 중복 제거 및 빈 문자열 제거
        must_unique_variants = []
        should_unique_variants = []
        seen_must = set()
        seen_should = set()
        for variant in must_variants:
            variant = variant.strip()
            if variant and variant not in seen_must and len(variant) > 1:
                must_unique_variants.append(variant)
                seen_must.add(variant)
        for variant in should_variants:
            variant = variant.strip()
            if variant and variant not in seen_should and len(variant) > 1:
                should_unique_variants.append(variant)
                seen_should.add(variant)
        
        return {
            "original": query,
            "cleaned": cleaned_query,
            "keywords": keywords,
            "must_variants": must_unique_variants[:3],  # 최대 3개 'must' 변형 사용
            "should_variants": should_unique_variants[:5]  # 최대 5개 'should' 변형 사용
        }


class SearchScoreEnhancer:
    """
    검색 결과 점수를 다양한 요소를 고려하여 보정하는 클래스
    """
    
    def __init__(self, 
                 keyword_match_boost: float = 0.05,
                 exact_match_boost: float = 0.1,
                 title_match_boost: float = 0.25,
                 recency_weight: float = 0.02):
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
        must_variants = query_info.get("must_variants", [])
        should_variants = query_info.get("should_variants", [])
        
        # 검색어와 키워드 기반 정규식 생성
        keyword_patterns = [re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE) for kw in keywords]
        exact_pattern = re.compile(re.escape(original_query), re.IGNORECASE)
        must_patterns = [re.compile(rf'\b{re.escape(variant)}\b', re.IGNORECASE) for variant in must_variants]
        should_patterns = [re.compile(rf'\b{re.escape(variant)}\b', re.IGNORECASE) for variant in should_variants]
        
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
            
            # 3. must_variants와 should_variants에 대한 점수 계산
            must_boost = sum(1 for pattern in must_patterns if pattern.search(content)) * 0.3  # must는 더 높은 가중치
            should_boost = sum(1 for pattern in should_patterns if pattern.search(content)) * 0.02  # should는 더 낮은 가중치
            
            # 4. 문서 메타데이터 고려 (제목 매치 등)
            title_boost = 0
            if "title" in doc.metadata and doc.metadata["title"]:
                title = doc.metadata["title"]
                title_keyword_matches = sum(1 for pattern in keyword_patterns if pattern.search(title))
                if title_keyword_matches > 0:
                    title_boost = min(self.title_match_boost, (title_keyword_matches / len(keywords)) * self.title_match_boost)
            
            # 5. 최신성 점수 (indexed_at 필드 있을 경우)
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
            boost_factor = 1 + keyword_boost + exact_boost + must_boost + should_boost + title_boost + recency_boost
            enhanced_score = base_score * boost_factor
            
            # 메타데이터에 점수 업데이트
            doc.metadata["original_score"] = base_score
            doc.metadata["enhanced_score"] = enhanced_score
            doc.metadata["boost_factor"] = boost_factor
            doc.metadata["keyword_boost"] = keyword_boost
            doc.metadata["exact_boost"] = exact_boost
            doc.metadata["must_boost"] = must_boost
            doc.metadata["should_boost"] = should_boost
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
