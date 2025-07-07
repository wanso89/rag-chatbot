"""
쿼리 최적화 및 검색 품질 향상을 위한 유틸리티 모듈
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

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
        """쿼리에서 핵심 키워드 추출 (동의어 사전 제거, 단순화)"""
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
        
        # 3. 최종 키워드 조합 (도메인 키워드 우선) - 동의어 확장 제거
        final_keywords = domain_keywords + basic_keywords
        
        # 중복 제거하면서 순서 보존
        unique_keywords = []
        seen = set()
        for kw in final_keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        
        # 동의어 사전 확장 완전 제거!
        print(f"🔍 키워드 추출 완료: {len(unique_keywords)}개 - {unique_keywords}")
        return unique_keywords

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 확장 및 변형 생성 (동의어 사전 제거)
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
        
        # 동의어 기반 변형 완전 제거!
        # (기존에 있던 synonym_builder 관련 코드 삭제됨)
        
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
