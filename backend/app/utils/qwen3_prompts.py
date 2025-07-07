"""
Qwen3 모델용 최적화된 프롬프트 템플릿 모듈
"""

from typing import List, Dict, Any, Optional

def create_chat_messages(
    question: str, 
    context: str, 
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    language: str = "ko"
) -> List[Dict[str, str]]:
    """
    채팅용 메시지 리스트 생성 (tokenizer.apply_chat_template용)
    
    Returns:
        List[Dict]: messages 형식으로 반환
    """
    
    # 대화 기록 처리
    messages = []
    
    if conversation_history:
        recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        for msg in recent_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 시스템 메시지
    if language == "ko":
        system_content = f"""당신은 사용자 질문에 대해 주어진 참고 문서를 기반으로 답변하는 한국어 AI 어시스턴트입니다.

**중요한 출처 인용 규칙:**
1. 답변할 때는 반드시 실제로 사용한 문서의 출처를 정확히 명시하세요.
2. 절대로 잘못된 출처를 인용하지 마세요. 
3. 답변 내용이 어떤 문서에서 나온 것인지 신중히 확인하고 그 문서만 인용하세요.
4. 형식: "[실제사용한파일명 p.페이지번호]에 따르면..." 또는 "[실제사용한파일명]에서 확인할 수 있듯이..."

**답변 지침:**
- 반드시 한국어로만 답변하세요
- 답변은 반드시 아래 '참고 문서' 섹션의 내용에만 근거해야 합니다
- 근거가 없는 경우 정보 제공이 어렵다고 정중히 안내하세요
- 각 문서는 "==== 문서 N: [파일명] ====" 형식으로 구분되어 있습니다
- 답변에서 인용할 때는 실제로 사용한 문서의 파일명을 정확히 확인하여 인용하세요
마지막으로, 문서를 개괄식으로 정리해서 보기 예쁘게 답변해야합니다.

참고 문서:
{context}"""
    else:
        system_content = f"""You are an AI assistant answering questions based on provided documents.

**Important Citation Rules:**
1. Always cite the exact source document you actually use for your answer.
2. Never cite incorrect sources.
3. Check carefully which document your answer content comes from and cite only that document.
4. Format: "[ActualUsedFilename p.PageNumber] states that..." or "According to [ActualUsedFilename]..."

Reference Documents:
{context}"""
    
    messages.insert(0, {"role": "system", "content": system_content})
    messages.append({"role": "user", "content": question})
    
    return messages



def create_query_optimization_prompt(query: str, category: str | None = None) -> str:
    """
    사용자의 자연어 질문 → 검색 엔진용 키워드 세트로 변환하기 위한 프롬프트.

    ◆ 규칙
    1. 반드시 JSON 하나만 출력.
    2. "keywords" 배열에는 2~4개의 **명사·영문 키워드**만 넣는다.
       · 조사·어미(에서, 에서도, 하기 전…) 는 모두 제거
       · 공백·중복 제거, 소문자 통일
    3. "search_queries" 배열에는 아래 두 가지 패턴을 넣는다.
       ├─ 첫 번째: keywords 3개까지를 " AND " 로 결합    예)  ndt4 AND ceph AND 설치
       └─ 두 번째: keywords 2개를 공백으로 결합        예)  ndt4 ceph
    4. 원본 문장·형용사·조사는 넣지 않는다.
    5. 중국어를 절대로 쓰지 않는다.
    """
    cat_line = f'분야(선택): {category}\n' if category else ''
    return f"""당신은 사용자의 자연어 질문을 Elasticsearch 검색에 최적화된
키워드 쿼리로 변환하는 도우미입니다.

{cat_line}질문:
\"\"\"{query}\"\"\"

아래 형식을 지켜 단 하나의 JSON 으로만 답하세요.

{{
  "keywords": ["", "", ""],
  "search_queries": ["", ""]
}}"""

def create_document_summarization_prompt(document_text: str, max_length: int = 300) -> str:
    """
    문서 요약 프롬프트 생성
    
    Args:
        document_text: 요약할 문서 텍스트
        max_length: 최대 요약 길이 (기본값: 300자)
        
    Returns:
        str: 문서 요약 프롬프트
    """
    return f"""<|im_start|>system
당신은 문서 내용을 명확하고 간결하게 요약하는 AI 전문가입니다. 주어진 문서의 핵심 내용을 유지하면서 간결한 요약을 제공해주세요.
<|im_end|>

<|im_start|>user
다음 문서 내용을 {max_length}자 이내로 요약해주세요. 핵심 정보만 포함하고, 중요하지 않은 세부 사항은 생략하세요.

문서 내용:
{document_text}
<|im_end|>

<|im_start|>assistant
"""

def create_title_generation_prompt(conversation_history: List[Dict[str, Any]]) -> str:
    """
    대화 제목 생성 프롬프트
    
    Args:
        conversation_history: 대화 기록
        
    Returns:
        str: 대화 제목 생성 프롬프트
    """
    # 대화 기록에서 최대 3개 대화만 포함
    conversation_text = ""
    max_turns = min(len(conversation_history), 3)
    
    for i in range(max_turns):
        msg = conversation_history[i]
        role = "사용자" if msg["role"] == "user" else "AI"
        content = msg["content"]
        # 긴 메시지는 100자로 제한
        if len(content) > 100:
            content = content[:97] + "..."
        conversation_text += f"{role}: {content}\n\n"
    
    return f"""<|im_start|>system
당신은 대화 내용을 분석하여 간결하고 명확한 제목을 생성하는 AI 전문가입니다.
<|im_end|>

<|im_start|>user
다음 대화 내용을 분석하여 15자 이내의 간결한 제목을 생성해주세요:

{conversation_text}

제목은 대화의 핵심 주제나 질문을 반영해야 합니다. 불필요한 단어는 생략하고 핵심만 포함하세요.
<|im_end|>

<|im_start|>assistant
"""
