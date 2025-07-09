"""
쿼리 최적화 및 검색 품질 향상을 위한 유틸리티 모듈
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

class QueryExpander:
    """
    사용자 쿼리를 확장하여 검색 품질을 높이는 클래스 (동의어 사전 유지 + 단순화)
    """
    
    def __init__(self):
        # 한국어 불용어 목록
        self.stopwords = {
            "그럼", "이거", "저거", "왜", "어떻게", "뭐", "무엇", 
            "이", "가", "을", "를", "은", "는", "에", "에서", "로", "으로",
            "과", "와", "도", "의", "들", "좀", "등", "및", "그", "저",
            "것", "수", "알려줘", "궁금해", "설명해줘", "알고싶어", "는?", "다른"
        }
        
        # 동의어 사전 (단순 문자열 매칭으로 변경)
        self.synonyms = {
            # 경조사 관련
            "회갑": "경조사", "환갑": "경조사", "칠순": "경조사", "팔순": "경조사", 
            "고희": "경조사", "생신": "경조사", "잔치": "경조사",
            "결혼식": "경조사", "결혼": "경조사", "혼례": "경조사", "웨딩": "경조사",
            "장례식": "경조사", "장례": "경조사", "초상": "경조사", "부고": "경조사",
            
            # 가족 관계
            "아버지": "부모", "어머니": "부모", "아빠": "부모", "엄마": "부모", "부모님": "부모",
            "조부모": "조부모", "할아버지": "조부모", "할머니": "조부모",
            "장인": "장인장모", "장모": "장인장모", "시아버지": "장인장모", "시어머니": "장인장모",
            "삼촌": "백숙부모", "이모": "백숙부모", "고모": "백숙부모", "숙모": "백숙부모",
            "남편": "배우자", "아내": "배우자",
            "형": "형제자매", "오빠": "형제자매", "누나": "형제자매", "동생": "형제자매",
            "아들": "자녀", "딸": "자녀", "자식": "자녀",
            
            # 지원 관련
            "지원금": "지원", "금액": "지원", "보조": "지원", "보조금": "지원", 
            "수당": "지원", "경조사비": "지원", "만원": "지원",
            
            # 직장 관련
            "직원": "직원", "근로자": "직원", "사원": "직원", "임직원": "직원", "종업원": "직원",
            "급여": "급여", "임금": "급여", "월급": "급여", "연봉": "급여", "보수": "급여",
            "휴가": "휴가", "휴직": "휴가", "연차": "휴가", "병가": "휴가", "특별휴가": "휴가",
            "출산": "출산휴가", "출산휴가": "출산휴가", "육아휴직": "출산휴가",
            
            # 회사명
            "3ssoft": "쓰리에스소프트", "쓰리에스소프트": "쓰리에스소프트", 
            "3s소프트": "쓰리에스소프트", "쓰리에스soft": "쓰리에스소프트"
        }
        
    def clean_query(self, query: str) -> str:
        """기본적인 쿼리 정제"""
        # 물음표 제거
        cleaned = query.replace("?", "").replace("는?", "").strip()
        # 여러 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 핵심 키워드 추출 (동의어 변환 + 단순화)"""
        # 1. 물음표 제거
        cleaned = query.replace("?", "").replace("는?", "").strip()
        
        # 2. 단순 토큰화 (공백 기준)
        tokens = cleaned.split()
        
        # 3. 토큰별 처리 (동의어 변환 + 불용어 제거)
        keywords = []
        for token in tokens:
            # 특수문자 제거
            clean_token = re.sub(r'[^\w가-힣]', '', token)
            
            # 조사 제거 (끝에 붙은 조사들) - $ 추가해서 끝부분만 매칭
            clean_token = re.sub(r'(는|은|이|가|을|를|에|에서|로|으로|와|과|도|의|들|만|부터|까지|처럼|같이|쪽|부분)$', '', clean_token)
            
            # 불용어 체크
            if clean_token in self.stopwords or len(clean_token) < 2:
                continue
            
            # 동의어 변환
            if clean_token in self.synonyms:
                converted = self.synonyms[clean_token]
                if converted not in keywords:  # 중복 방지
                    keywords.append(converted)
            else:
                if clean_token not in keywords:  # 중복 방지
                    keywords.append(clean_token)
        
        print(f"🔍 키워드 추출 완료: {len(keywords)}개 - {keywords}")
        return keywords

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 확장 및 변형 생성
        """
        cleaned_query = self.clean_query(query)
        keywords = self.extract_keywords(cleaned_query)
        
        # 단순한 변형만 생성
        must_variants = [query.strip()]
        should_variants = []
        
        # 키워드만으로 구성된 쿼리 추가
        if keywords:
            keyword_query = " ".join(keywords[:5])  # 최대 5개 키워드
            if keyword_query != query.strip():
                must_variants.append(keyword_query)
        
        return {
            "original": query,
            "cleaned": cleaned_query,
            "keywords": keywords,
            "must_variants": must_variants[:2],
            "should_variants": should_variants
        }
    def get_bm25_query(self, query: str) -> str:
        """BM25용 정제된 쿼리"""
        keywords = self.extract_keywords(query)
        return " ".join(keywords)
    
    def get_faiss_query(self, query: str) -> str:
        """FAISS용 원본 쿼리 (물음표만 제거)"""
        return query.replace("?", "").strip()