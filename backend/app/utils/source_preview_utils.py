# app/utils/source_preview_utils.py
"""
소스 프리뷰를 위한 전용 유틸리티 함수들

이 모듈은 소스 프리뷰 기능에 필요한 모든 보조 기능들을 제공합니다.
각 함수는 독립적으로 작동하며, 다른 부분에서도 재사용할 수 있도록 설계되었습니다.
"""

import re
import os
from typing import List, Dict, Any
from collections import Counter


def extract_keywords_from_query(query: str) -> List[str]:
    """
    검색 쿼리에서 의미있는 키워드를 추출합니다.
    
    이 함수는 불용어를 제거하고 실제 검색에 사용된 핵심 단어들만 추출합니다.
    한국어와 영어 모두 지원하며, 너무 짧은 단어는 제외합니다.
    
    Args:
        query (str): 검색 쿼리 문자열
        
    Returns:
        List[str]: 추출된 키워드 목록 (최대 10개)
    """
    if not query:
        return []
    
    # 불용어 목록 - 검색에서 의미가 없는 단어들
    korean_stopwords = {'이', '가', '을', '를', '의', '에', '은', '는', '와', '과', '로', '으로', '에서', '부터', '까지', '이다'}
    english_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    # 한글, 영문, 숫자만 추출 (2글자 이상)
    words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{2,}|\d+', query)
    
    # 불용어 제거 및 키워드 정리
    keywords = []
    for word in words:
        word_lower = word.lower()
        if (word_lower not in korean_stopwords and 
            word_lower not in english_stopwords and
            len(word) >= 2):
            keywords.append(word)
    
    return keywords[:10]  # 최대 10개로 제한


def extract_keywords_from_content(content: str) -> List[str]:
    """
    문서 내용에서 자동으로 키워드를 추출합니다.
    
    단어 빈도를 기반으로 중요한 단어들을 찾아냅니다.
    너무 흔한 단어들은 제외하고, 의미있는 단어들만 선별합니다.
    
    Args:
        content (str): 문서 내용
        
    Returns:
        List[str]: 추출된 키워드 목록 (최대 8개)
    """
    if not content:
        return []
    
    # 한글, 영문 단어 추출 (3글자 이상으로 더 엄격하게)
    words = re.findall(r'[가-힣]{3,}|[a-zA-Z]{3,}', content)
    
    # 단어 빈도 계산
    word_freq = Counter(word.lower() for word in words)
    
    # 너무 흔한 단어들 제외 (문서에서 자주 나오지만 의미가 적은 단어들)
    common_words = {
        '입니다', '있습니다', '합니다', '됩니다', '것입니다', '수있습니다',
        'the', 'and', 'for', 'are', 'with', 'that', 'this', 'have'
    }
    
    # 빈도가 높으면서 불용어가 아닌 단어들 선택
    keywords = []
    for word, freq in word_freq.most_common(20):
        if word not in common_words and freq >= 2:  # 최소 2번 이상 등장
            keywords.append(word)
    
    return keywords[:8]



def apply_highlighting(content: str, keywords: List[str], original_query: str = None, metadata: Dict[str, Any] = None, embedding_function: Any = None) -> tuple:
    """
    Semantic similarity 기반으로 답변과 관련된 문장을 하이라이트합니다.
    
    Args:
        content (str): 하이라이트할 문서 내용
        keywords (List[str]): 하이라이트에 사용할 키워드 목록
        original_query (str, optional): 답변 텍스트
        metadata (Dict[str, Any], optional): 문서 메타데이터 (리랭킹 점수 포함)
        embedding_function (Any, optional): 임베딩 함수
    
    Returns:
        tuple: (하이라이트된_내용, 하이라이트_여부)
    """
    if not content:
        return content, False
    
    has_highlights = False
    
    # 2. 답변에서 핵심 정보 추출 (NER 스타일)
    answer_entities = _extract_key_entities(original_query)
    
    # 3. 내용을 문장 단위로 분리
    sentences = _split_into_sentences(content)
    
    # 4. 각 문장의 중요도 점수 계산
    highlighted_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        importance_score = _calculate_sentence_importance(sentence, answer_entities, keywords, metadata, original_query, embedding_function)
        
        # 점수가 높은 문장은 하이라이트 (임계값 강화)
        if importance_score > 0.75:  # 임계값을 0.6에서 0.75로 높임
            highlighted_sentence = f'<span class="highlight-strong">{sentence.strip()}</span>'
            highlighted_sentences.append(highlighted_sentence)
            has_highlights = True
            print(f"🔍 하이라이트: '{sentence.strip()[:50]}...' (점수: {importance_score:.2f})")
        else:
            highlighted_sentences.append(sentence.strip())
    
    result_content = '. '.join([s for s in highlighted_sentences if s])
    return result_content, has_highlights


def _extract_key_entities(text: str) -> Dict[str, List[str]]:
    """답변 텍스트에서 중요한 엔티티들을 추출합니다."""
    entities = {
        'numbers': re.findall(r'\d+(?:[일개월년원달러주차회번째원만원천원])?', text),
        'dates': re.findall(r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}', text),
        'periods': re.findall(r'\d+[일개월년주]', text),
        'parenthetical': re.findall(r'\(([^)]+)\)', text),
        'quoted': re.findall(r'[\'"]([^\'"]+)[\'"]', text),
    }
    return entities


def _split_into_sentences(content: str) -> List[str]:
    """텍스트를 문장 단위로 분리합니다."""
    # 마침표, 느낌표, 물음표, 줄바꿈 기준으로 분리
    sentences = re.split(r'[.\n!?]+', content)
    return [s.strip() for s in sentences if s.strip()]


def _calculate_sentence_importance(sentence: str, answer_entities: Dict, keywords: List[str], metadata: Dict[str, Any] = None, original_query: str = None, embedding_function: Any = None) -> float:
    """문장의 중요도 점수를 계산합니다."""
    score = 0.0
    
    # 1. 답변 엔티티와의 일치도 (가장 중요)
    for entity_type, entities in answer_entities.items():
        for entity in entities:
            if entity.strip() and entity.strip() in sentence:
                if entity_type == 'numbers':
                    # 숫자가 쿼리와 관련이 있는지 확인
                    if original_query and entity.strip() in original_query:
                        score += 0.1  # 쿼리와 관련된 숫자는 낮은 점수 부여
                    else:
                        score += 0.0  # 관련 없는 숫자는 점수 부여하지 않음
                elif entity_type == 'periods':
                    score += 0.1  # 기간도 낮은 점수 부여
                else:
                    score += 0.0  # 기타 엔티티는 점수 부여하지 않음
    
    # 2. 키워드 밀도
    matching_keywords = sum(1 for k in keywords if k.lower() in sentence.lower())
    if keywords and matching_keywords >= 1:  # 최소 1개 이상 키워드 일치해야 점수 부여
        keyword_density = matching_keywords / len(keywords)
        score += keyword_density * 0.4  # 키워드 비중 0.4    
    # 3. 문서 메타데이터에서 리랭킹 점수 반영 (BM25 + FAISS 결합 점수)
    if metadata:
        if 'relevance_score' in metadata:
            relevance_score = metadata.get('relevance_score', 0.0)
            score += relevance_score * 0.6  # 리랭킹 점수 비중 0.6
        else:
            print("⚠️ 메타데이터에 relevance_score가 없습니다.")
    else:
        print("⚠️ 메타데이터가 전달되지 않았습니다.")
    
    # 5. 문장 길이 보정 (너무 짧거나 긴 문장은 덜 중요)
    sentence_length = len(sentence.split())
    if 5 <= sentence_length <= 30:  # 적절한 길이
        score += 0.1
    return score


def format_source_metadata(metadata: Dict[str, Any], doc_source: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    소스 메타데이터를 사용자 친화적인 형태로 포맷팅합니다.
    
    파일명에서 해시를 제거하고, 표 정보가 있으면 추가하는 등
    사용자가 이해하기 쉬운 형태로 정보를 정리합니다.
    
    Args:
        metadata (Dict): 기본 메타데이터
        doc_source (Dict, optional): 원본 문서 소스 정보
        
    Returns:
        Dict: 포맷팅된 메타데이터
    """
    formatted = metadata.copy()
    
    # 파일명에서 해시 제거
    if 'filename' in formatted:
        formatted['display_name'] = _extract_clean_filename(formatted['filename'])
    
    # 표 정보 추가 (doc_source가 제공된 경우)
    if doc_source and doc_source.get("element_type") == "table_row":
        table_info = []
        
        if doc_source.get("table_type"):
            table_type_names = {
                "pipe_separated": "파이프 구분 테이블",
                "tab_separated": "탭 구분 테이블", 
                "space_aligned": "공백 정렬 테이블",
                "numbered_list": "번호 목록 테이블"
            }
            table_info.append(table_type_names.get(doc_source["table_type"], doc_source["table_type"]))
        
        if doc_source.get("headers"):
            headers = doc_source["headers"]
            table_info.append(f"컬럼: {', '.join(headers[:3])}{'...' if len(headers) > 3 else ''}")
        
        if table_info:
            formatted['table_info'] = " | ".join(table_info)
            formatted['is_table'] = True
    
    return formatted


def enhance_content_with_answer_context(content: str, answer_text: str) -> str:
    """
    답변 텍스트와 문서 내용 간의 연관성을 강화합니다.
    
    답변에서 직접 인용된 부분이나 중요한 정보가 문서에서 어떤 부분과
    연결되는지 시각적으로 표시합니다. 이렇게 하면 사용자가 답변의
    근거를 문서에서 쉽게 찾을 수 있습니다.
    
    Args:
        content (str): 문서 내용
        answer_text (str): 답변 텍스트
        
    Returns:
        str: 연관성이 강화된 문서 내용
    """
    if not answer_text or not content:
        return content
    
    enhanced_content = content
    
    # 답변에서 핵심 문구 추출
    quoted_phrases = re.findall(r'"([^"]+)"', answer_text)  # 따옴표 안의 내용
    parenthetical = re.findall(r'\(([^)]+)\)', answer_text)  # 괄호 안의 내용
    
    # 답변에서 직접 인용된 문구들을 문서에서 강조
    for phrase in quoted_phrases:
        if len(phrase) > 5 and phrase in content:  # 5글자 이상의 의미있는 문구만
            enhanced_content = enhanced_content.replace(
                phrase, 
                f'<mark class="answer-reference">{phrase}</mark>'
            )
    
    # 괄호 안의 중요 정보들도 강조
    for info in parenthetical:
        if len(info) > 3 and info in content:
            enhanced_content = enhanced_content.replace(
                info,
                f'<mark class="contextual-info">{info}</mark>'
            )
    
    return enhanced_content


def _extract_clean_filename(file_path: str) -> str:
    """
    파일 경로에서 깔끔한 파일명을 추출합니다.
    
    해시 프리픽스를 제거하여 사용자가 이해하기 쉬운 파일명을 반환합니다.
    예: "a1b2c3d4e5f6_문서.pdf" -> "문서.pdf"
    
    Args:
        file_path (str): 원본 파일 경로
        
    Returns:
        str: 정리된 파일명
    """
    filename = os.path.basename(file_path)
    
    # 해시 프리픽스 제거 (12자리 해시_파일명 형태)
    if '_' in filename:
        parts = filename.split('_', 1)
        if len(parts) > 1 and len(parts[0]) >= 8:  # 해시로 보이는 부분
            return parts[1]
    
    return filename

def filter_meaningful_keywords(keywords):
    """무의미한 키워드 제외"""
    meaningless_keywords = {
        'pdf', 'p', 'docx', 'doc', 'txt',  # 파일 확장자
        '25', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',  # 단순 숫자
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
        '이야', '에서', '니다', '니까', '해주', '니???', '???',  # 불용어/특수문자
        '회사', '문서', '파일', '페이지',  # 너무 일반적인 단어
        '개정', '20100129', '20230109', '250514'  # 날짜 형식들
    }
    
    meaningful_keywords = [
        kw for kw in keywords 
        if len(kw) > 2 and kw not in meaningless_keywords and not kw.isdigit()
    ]
    
    return meaningful_keywords


def excel_to_html(path: str) -> Dict[str, str]:
    """
    엑셀 파일을 불러와 각 시트를 HTML <table> 문자열로 변환하여 딕셔너리로 반환합니다.
    
    Args:
        path (str): 엑셀 파일 경로
        
    Returns:
        Dict[str, str]: 시트 이름을 키로 하고 HTML 테이블 문자열을 값으로 하는 딕셔너리
    """
    import pandas as pd
    
    try:
        # 엑셀 파일의 모든 시트 읽기
        sheets_dict = pd.read_excel(path, sheet_name=None)
        html_dict = {}
        
        for sheet_name, df in sheets_dict.items():
            # DataFrame을 HTML 테이블로 변환
            html_table = df.to_html(index=False, render_links=True, escape=False)
            html_dict[sheet_name] = html_table
            
        return html_dict
    except Exception as e:
        print(f"엑셀 파일 처리 중 오류 발생: {e}")
        return {}
