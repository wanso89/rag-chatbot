# app/utils/table_parser.py
"""
표 파싱을 위한 전용 모듈

이 모듈은 다양한 형태의 표(테이블)를 감지하고 파싱하는 기능을 제공합니다.
OCR 텍스트나 일반 텍스트에서 표 구조를 인식하고, 각 행을 의미있는 Document 객체로 변환합니다.

주요 기능:
1. 표 감지 (detect_table_in_text): 텍스트에서 표 구조 존재 여부 판단
2. 표 파싱 (parse_ocr_table): 감지된 표를 구조적으로 분석하여 Document 리스트 생성
3. 표 연속성 판단 (is_same_table): 페이지 간 분할된 표의 연결성 분석
4. 타입별 전용 파싱: 파이프, 탭, 공백 정렬, 번호 목록 등 다양한 표 형태 지원
"""

import re
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.schema import Document


def detect_table_in_text(text: str) -> bool:
    """
    텍스트에서 표 구조를 감지하는 메인 함수
    
    다층적 분석을 통해 다양한 표 형태를 감지합니다:
    - 구분자 기반 표 (파이프 |, 탭, 연속 공백)
    - HTML/마크다운 표
    - 번호 목록형 표
    - 정렬된 데이터 패턴
    
    Args:
        text (str): 분석할 텍스트
        
    Returns:
        bool: 표 구조가 감지되면 True, 아니면 False
        
    Example:
        >>> text = "이름 | 나이 | 직업\n홍길동 | 30 | 개발자"
        >>> detect_table_in_text(text)
        True
    """
    
    if not text or len(text.strip()) < 10:  # 20 → 10으로 완화
        return False
    
    # 원본 텍스트 그대로 사용 (휴리스틱 처리 없이)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # 한 줄짜리도 허용 (파이프나 탭이 있으면 표일 가능성)
    if len(lines) < 1:
        return False
    
    table_indicators = 0  # 표 지표 점수 시스템
    
    # === 1단계: 구분자 기반 감지 ===
    # 파이프(|)와 탭으로 구분된 표는 가장 명확한 표 형태입니다
    pipe_lines = [line for line in lines if line.count('|') >= 2]
    tab_lines = [line for line in lines if line.count('\t') >= 1]
    
    if len(pipe_lines) >= 3:  # 파이프 구분 표
        table_indicators += 3
    if len(tab_lines) >= 3:   # 탭 구분 표  
        table_indicators += 3
    
    # === 2단계: 정렬된 컬럼 구조 감지 ===
    # OCR에서 자주 나오는 "공백으로 정렬된 표" 형태를 감지합니다
    aligned_lines = []
    for line in lines:
        if re.search(r'\s{3,}', line):  # 3개 이상 연속 공백
            parts = re.split(r'\s{3,}', line)
            if len(parts) >= 2:  # 최소 2개 컬럼
                aligned_lines.append(parts)
    
    if len(aligned_lines) >= 3:
        # 컬럼 수가 일정한지 확인 - 표라면 각 행의 컬럼 수가 비슷해야 합니다
        col_counts = [len(parts) for parts in aligned_lines]
        if len(set(col_counts)) <= 2:  # 컬럼 수가 1-2가지 패턴만 있으면 표로 판단
            table_indicators += 2
    
    # === 3단계: 표 헤더 패턴 감지 ===
    # 표의 첫 부분에는 보통 헤더가 있습니다
    header_patterns = [
        # 한국어 표 헤더 - 일반적인 컬럼명들
        r'(항목|구분|분류|번호|순번|제목|내용|설명|값|수량|금액|날짜|시간|기간)',
        r'(요구사항|규격|제품|사양|여부|상태|결과|평가|점수|등급)',
        r'(이름|성명|부서|직급|담당|역할|업무|위치|주소)',
        # 영어 표 헤더
        r'(No\.|Item|Type|Category|Description|Value|Amount|Date|Time|Status)',
        r'(Name|Department|Position|Role|Location|Address|Phone|Email)',
        # 구분선 패턴 - 헤더 아래 구분선이 있는 경우
        r'^(\s*-{2,}\s*){2,}$',  # 대시 구분선 (---)
        r'^(\s*={2,}\s*){2,}$',  # 등호 구분선 (===)
        r'^\s*\|(\s*-+\s*\|){2,}\s*$',  # 마크다운 테이블 구분선
    ]
    
    header_found = False
    for i, line in enumerate(lines[:5]):  # 처음 5줄에서 헤더 찾기
        for pattern in header_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                header_found = True
                table_indicators += 2
                break
        if header_found:
            break
    
    # === 4단계: 반복 패턴 감지 ===
    # 번호가 있는 목록형 표 (1. 항목, 2. 항목 형태)
    numbered_lines = [line for line in lines if re.match(r'^\s*\d+[\.\)\s]', line)]
    if len(numbered_lines) >= 3:
        table_indicators += 1
    
    # 특정 키워드의 반복 - 표에서 자주 나오는 단어들
    repeated_words = ['적격', '지원', '사양', '요구', '규격', '승인', '거부', '통과', '실패']
    for word in repeated_words:
        count = text.count(word)
        if count >= 4:  # 4회 이상 반복시 표 가능성
            table_indicators += 1
        if count >= 7:  # 7회 이상이면 확실히 표
            table_indicators += 2
    
    # === 5단계: HTML/마크다운 표 감지 ===
    if re.search(r'<table.*?>.*?</table>', text, re.DOTALL | re.IGNORECASE):
        table_indicators += 3
    if re.search(r'\|.*\|.*\n\s*\|[-\s:]+\|', text):  # 마크다운 표
        table_indicators += 3
    
    # === 6단계: 데이터 패턴 분석 ===
    # 숫자와 텍스트가 규칙적으로 배열된 패턴 감지
    data_pattern_lines = 0
    for line in lines:
        # "텍스트 숫자 텍스트" 또는 "숫자 텍스트 숫자" 패턴
        if re.search(r'[가-힣a-zA-Z]+\s+\d+\s+[가-힣a-zA-Z]+', line) or \
           re.search(r'\d+\s+[가-힣a-zA-Z]+\s+\d+', line):
            data_pattern_lines += 1
    
    if data_pattern_lines >= 3:
        table_indicators += 2
    
    # === 최종 판단 ===
    # 점수가 2점 이상이면 표로 판단 (기존 3점에서 완화)
    # 더 많은 테이블을 감지하도록 임계값 조정
    print(f"DEBUG TABLE SCORE ▸ 표 감지 점수: {table_indicators}점")
    return table_indicators >= 2


def parse_ocr_table(text: str) -> List[Document]:
    """
    감지된 표를 구조적으로 파싱하여 Document 리스트로 변환
    
    표의 타입을 먼저 감지한 후, 각 타입에 특화된 파싱 방법을 적용합니다.
    각 행은 개별 Document로 변환되며, 헤더 정보와 셀 구조가 메타데이터에 포함됩니다.
    
    Args:
        text (str): 표가 포함된 텍스트
        
    Returns:
        List[Document]: 파싱된 표 행들의 Document 리스트
        
    Example:
        >>> text = "이름|나이|직업\n홍길동|30|개발자\n김철수|25|디자이너"
        >>> docs = parse_ocr_table(text)
        >>> len(docs)
        2
        >>> docs[0].page_content
        "이름: 홍길동 | 나이: 30 | 직업: 개발자"
    """
    if not detect_table_in_text(text):
        return []
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return []
    
    # 표 타입 감지 후 적절한 파싱 함수 호출
    table_type = _detect_table_type(lines)
    
    # 각 표 타입별로 전용 파싱 함수를 사용합니다
    # 이렇게 하면 각 타입의 특성에 맞는 최적화된 파싱이 가능합니다
    if table_type == "pipe_separated":
        return _parse_pipe_table(lines, text)
    elif table_type == "tab_separated":
        return _parse_tab_table(lines, text)
    elif table_type == "space_aligned":
        return _parse_space_aligned_table(lines, text)
    elif table_type == "numbered_list":
        return _parse_numbered_table(lines, text)
    else:
        return _parse_generic_table(lines, text)


def is_same_table(prev: Document, cur: Document) -> bool:
    """
    두 Document가 같은 표의 연속인지 판단
    
    페이지가 넘어가면서 분할된 표를 연결하기 위해 사용됩니다.
    단순히 형태적 유사성뿐만 아니라 의미적 연관성도 함께 고려합니다.
    
    Args:
        prev (Document): 이전 페이지의 표 Document
        cur (Document): 현재 페이지의 표 Document
        
    Returns:
        bool: 같은 표의 연속이면 True, 아니면 False
        
    Example:
        >>> # 같은 표의 연속인 경우
        >>> prev_doc = Document(page_content="이름: 홍길동 | 나이: 30", 
        ...                     metadata={"page": 1, "headers": ["이름", "나이"]})
        >>> cur_doc = Document(page_content="이름: 김철수 | 나이: 25",
        ...                    metadata={"page": 2, "headers": ["이름", "나이"]})
        >>> is_same_table(prev_doc, cur_doc)
        True
    """
    
    # === 1단계: 기본 연속성 확인 ===
    prev_page = prev.metadata.get("page", 0)
    cur_page = cur.metadata.get("page", 0)
    
    # 페이지가 연속되지 않으면 같은 표가 아닙니다
    if cur_page != prev_page + 1:
        return False
    
    # 둘 다 table_row 타입이어야 비교 가능합니다
    if (prev.metadata.get("element_type") != "table_row" or 
        cur.metadata.get("element_type") != "table_row"):
        return False
    
    # === 2단계: 표 타입 일치 확인 ===
    prev_type = prev.metadata.get("table_type", "unknown")
    cur_type = cur.metadata.get("table_type", "unknown")
    
    # 표 타입이 명확히 다르면 다른 표로 판단
    if prev_type != "unknown" and cur_type != "unknown" and prev_type != cur_type:
        return False
    
    # === 3단계: 구조적 유사성 분석 ===
    similarity_score = 0
    
    # 3-1. 헤더 정보 비교 - 가장 중요한 지표
    prev_headers = prev.metadata.get("headers", [])
    cur_headers = cur.metadata.get("headers", [])
    
    if prev_headers and cur_headers:
        if prev_headers == cur_headers:  # 헤더가 완전히 같음
            similarity_score += 3
        elif len(prev_headers) == len(cur_headers):  # 헤더 개수만 같음
            similarity_score += 1
    
    # 3-2. 셀 개수 비교 (컬럼 수)
    prev_cells = prev.metadata.get("raw_cells", [])
    cur_cells = cur.metadata.get("raw_cells", [])
    
    if prev_cells and cur_cells:
        if len(prev_cells) == len(cur_cells):  # 컬럼 수가 정확히 같음
            similarity_score += 2
        elif abs(len(prev_cells) - len(cur_cells)) <= 1:  # 컬럼 수 차이가 1 이하
            similarity_score += 1
    
    # 3-3. 내용 패턴 유사성 - 공통 단어 비율 계산
    prev_content = prev.page_content.lower()
    cur_content = cur.page_content.lower()
    
    # 의미있는 단어들만 추출 (한글/영문 2글자 이상)
    prev_words = set(re.findall(r'[가-힣a-zA-Z]{2,}', prev_content))
    cur_words = set(re.findall(r'[가-힣a-zA-Z]{2,}', cur_content))
    
    if prev_words and cur_words:
        common_words = prev_words & cur_words
        word_similarity = len(common_words) / max(len(prev_words), len(cur_words))
        
        if word_similarity >= 0.3:  # 30% 이상 공통 단어
            similarity_score += 2
        elif word_similarity >= 0.15:  # 15% 이상 공통 단어
            similarity_score += 1
    
    # === 4단계: 기존 파이프 로직 보완 ===
    # 파이프 구분 표의 경우 추가 검증
    if prev_type == "pipe_separated" and cur_type == "pipe_separated":
        n_bar_prev = prev.page_content.count("|")
        n_bar_cur = cur.page_content.count("|")
        
        if n_bar_prev == n_bar_cur and n_bar_prev > 0:
            similarity_score += 1
            
            # 첫 번째 셀 내용 유사성 확인
            try:
                prev_first_cells = prev.page_content.split("|")[:2]
                cur_first_cells = cur.page_content.split("|")[:2]
                
                if len(prev_first_cells) >= 2 and len(cur_first_cells) >= 2:
                    if prev_first_cells == cur_first_cells:
                        similarity_score += 2
                    elif any(word in cur_first_cells[0] for word in prev_first_cells[0].split() if len(word) > 2):
                        similarity_score += 1
            except:
                pass  # 파싱 실패시 무시
    
    # === 5단계: 최종 판단 ===
    # 유사성 점수가 3점 이상이면 같은 표로 판단
    # 이 임계값은 false positive와 false negative를 균형있게 고려한 값입니다
    return similarity_score >= 3


def _detect_table_type(lines: List[str]) -> str:
    """
    표의 구체적인 형태를 감지하여 적절한 파싱 전략을 결정
    
    각 라인을 분석하여 가장 많이 나타나는 패턴을 기준으로
    표 타입을 결정합니다. 이렇게 하면 각 타입에 최적화된 파싱이 가능합니다.
    """
    pipe_count = sum(1 for line in lines if line.count('|') >= 2)
    tab_count = sum(1 for line in lines if '\t' in line)
    aligned_count = sum(1 for line in lines if re.search(r'\s{3,}', line))
    numbered_count = sum(1 for line in lines if re.match(r'^\s*\d+[\.\)]', line))
    
    # 비율 기반으로 표 타입 결정
    total_lines = len(lines)
    if pipe_count >= total_lines * 0.6:  # 60% 이상이 파이프 구분
        return "pipe_separated"
    elif tab_count >= total_lines * 0.6:
        return "tab_separated" 
    elif aligned_count >= total_lines * 0.5:
        return "space_aligned"
    elif numbered_count >= total_lines * 0.4:
        return "numbered_list"
    else:
        return "generic"


def _find_header_row(lines: List[str]) -> int:
    """
    표에서 헤더 행의 인덱스를 찾기
    
    헤더는 보통 표의 첫 부분에 있으며, 특정 키워드 패턴을 가집니다.
    한국어와 영어 표 헤더를 모두 지원합니다.
    """
    header_patterns = [
        r'(항목|구분|분류|번호|순번|제목|내용|설명|값|수량|금액)',
        r'(요구사항|규격|제품|사양|여부|상태|결과|평가)',
        r'(No\.|Item|Type|Category|Description|Value|Amount)',
        r'(Name|Department|Position|Role|Location)'
    ]
    
    for idx, line in enumerate(lines[:3]):  # 처음 3줄에서만 헤더 찾기
        for pattern in header_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return idx
    
    return -1  # 헤더 없음


def _parse_pipe_table(lines: List[str], original_text: str) -> List[Document]:
    """
    파이프(|) 구분 표 파싱
    
    마크다운 스타일 표나 CSV의 파이프 구분자 버전을 파싱합니다.
    헤더를 찾아서 각 행을 "헤더: 값" 형태로 구조화합니다.
    """
    table_docs = []
    table_id = uuid.uuid4().hex[:8]
    
    # 헤더 찾기
    header_idx = _find_header_row(lines)
    headers = []
    
    if header_idx >= 0:
        header_line = lines[header_idx]
        headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]
    
    # 데이터 행 파싱
    for idx, line in enumerate(lines):
        if not line or line.count('|') < 1:
            continue
            
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if not cells:
            continue
        
        # 헤더 행은 건너뛰기
        if idx == header_idx:
            continue
        
        # 구분선(예: |---|---|) 건너뛰기
        if all(re.match(r'^[-=\s]*$', cell) for cell in cells):
            continue
        
        # 행 내용 구성 - 헤더가 있으면 "헤더: 값" 형태로
        if headers and len(headers) == len(cells):
            row_content = " | ".join([f"{headers[i]}: {cells[i]}" for i in range(len(cells))])
        else:
            row_content = " | ".join(cells)
        
        table_docs.append(Document(
            page_content=row_content,
            metadata={
                "element_type": "table_row",
                "table_id": table_id,
                "row_index": idx,
                "n_rows": len(lines),
                "table_type": "pipe_separated",
                "headers": headers,
                "raw_cells": cells,
                "row_text": original_text  # 파싱 실패시 원문 보존
            }
        ))
    
    return table_docs


def _parse_space_aligned_table(lines: List[str], original_text: str) -> List[Document]:
    """
    공백으로 정렬된 표 파싱
    
    OCR 결과에서 자주 나오는 형태입니다. 3개 이상 연속 공백을
    컬럼 구분자로 사용하여 셀을 분리합니다.
    """
    table_docs = []
    table_id = uuid.uuid4().hex[:8]
    
    # 헤더 찾기
    header_idx = _find_header_row(lines)
    headers = []
    
    if header_idx >= 0:
        header_line = lines[header_idx]
        headers = re.split(r'\s{3,}', header_line.strip())
        headers = [h.strip() for h in headers if h.strip()]
    
    # 데이터 행 파싱
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
            
        # 3개 이상 연속 공백으로 분할
        if re.search(r'\s{3,}', line):
            cells = re.split(r'\s{3,}', line.strip())
            cells = [cell.strip() for cell in cells if cell.strip()]
        else:
            cells = [line.strip()]  # 분할 불가능한 경우 전체를 하나의 셀로
        
        if not cells:
            continue
        
        # 헤더 행은 건너뛰기
        if idx == header_idx:
            continue
        
        # 행 내용 구성
        if headers and len(headers) == len(cells):
            row_content = " | ".join([f"{headers[i]}: {cells[i]}" for i in range(len(cells))])
        else:
            row_content = " | ".join(cells)
        
        table_docs.append(Document(
            page_content=row_content,
            metadata={
                "element_type": "table_row",
                "table_id": table_id,
                "row_index": idx,
                "n_rows": len(lines),
                "table_type": "space_aligned",
                "headers": headers,
                "raw_cells": cells,
                "row_text": original_text
            }
        ))
    
    return table_docs


def _parse_tab_table(lines: List[str], original_text: str) -> List[Document]:
    """
    탭 구분 표 파싱
    
    Excel에서 복사한 데이터나 TSV 형태의 표를 파싱합니다.
    탭 문자를 컬럼 구분자로 사용합니다.
    """
    table_docs = []
    table_id = uuid.uuid4().hex[:8]
    
    header_idx = _find_header_row(lines)
    headers = []
    
    if header_idx >= 0:
        headers = [cell.strip() for cell in lines[header_idx].split('\t') if cell.strip()]
    
    for idx, line in enumerate(lines):
        if '\t' not in line:
            continue
            
        cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
        if not cells or idx == header_idx:
            continue
        
        if headers and len(headers) == len(cells):
            row_content = " | ".join([f"{headers[i]}: {cells[i]}" for i in range(len(cells))])
        else:
            row_content = " | ".join(cells)
        
        table_docs.append(Document(
            page_content=row_content,
            metadata={
                "element_type": "table_row",
                "table_id": table_id,
                "row_index": idx,
                "n_rows": len(lines),
                "table_type": "tab_separated",
                "headers": headers,
                "raw_cells": cells,
                "row_text": original_text
            }
        ))
    
    return table_docs


def _parse_numbered_table(lines: List[str], original_text: str) -> List[Document]:
    """
    번호 목록형 표 파싱
    
    "1. 항목", "2. 항목" 형태의 번호가 있는 목록을 표로 처리합니다.
    각 항목의 번호와 내용을 분리하여 구조화합니다.
    """
    table_docs = []
    table_id = uuid.uuid4().hex[:8]
    
    for idx, line in enumerate(lines):
        match = re.match(r'^\s*(\d+)[\.\)]\s*(.+)', line)
        if not match:
            continue
            
        number = match.group(1)
        content = match.group(2).strip()
        
        table_docs.append(Document(
            page_content=f"항목 {number}: {content}",
            metadata={
                "element_type": "table_row",
                "table_id": table_id,
                "row_index": idx,
                "n_rows": len(lines),
                "table_type": "numbered_list",
                "item_number": number,
                "raw_cells": [number, content],
                "row_text": original_text
            }
        ))
    
    return table_docs


def _parse_generic_table(lines: List[str], original_text: str) -> List[Document]:
    """
    일반적인 표 형태 파싱 (fallback)
    
    특정 패턴으로 분류되지 않는 표에 대한 기본 파싱 방법입니다.
    키워드를 기반으로 표 행을 감지합니다.
    """
    table_docs = []
    table_id = uuid.uuid4().hex[:8]
    
    # 표에서 자주 나오는 키워드들
    table_keywords = ['적격', '요구', '사양', '승인', '거부', '통과', '실패', '확인', '검토']
    
    for idx, line in enumerate(lines):
        if any(keyword in line for keyword in table_keywords):
            table_docs.append(Document(
                page_content=f"표 데이터: {line}",
                metadata={
                    "element_type": "table_row",
                    "table_id": table_id,
                    "row_index": idx,
                    "n_rows": len(lines),
                    "table_type": "generic",
                    "raw_cells": [line],
                    "row_text": original_text
                }
            ))
    
    return table_docs


# === 공개 함수들 (외부에서 import할 함수들) ===
__all__ = [
    'detect_table_in_text',
    'parse_ocr_table', 
    'is_same_table'
]
