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
다음 지침을 매우 엄격히 따라주세요:

1. 반드시 한국어로만 답변하세요. 절대 중국어나 다른 언어를 사용하지 마세요.
2. 답변은 반드시 제공된 '참고 문서' 섹션의 내용에 근거해야 합니다.
3. 문서에 질문과 관련된 정보가 없다면, "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 명확히 답변하세요.
4. 답변할 때는 반드시 구체적인 출처를 명시해주세요.
   예: "[취업규칙_test.pdf p.9]에 따르면..." 또는 "[네오오토_취업규칙.pdf]에서 확인할 수 있듯이..."
5. "문서 1", "문서 2" 같은 표현은 절대 사용하지 마세요.

참고 문서:
{context}"""
    else:
        system_content = f"""You are an AI assistant answering questions based on provided documents.
Reference Documents:
{context}"""
    
    messages.insert(0, {"role": "system", "content": system_content})
    messages.append({"role": "user", "content": question})
    
    return messages

def create_query_optimization_prompt(query: str, category: str = None) -> str:
    """
    쿼리 최적화용 프롬프트 생성
    
    Args:
        query: 원본 사용자 질문
        category: 문서 카테고리
        
    Returns:
        str: 쿼리 최적화 프롬프트
    """
    
    category_info = f"카테고리: {category}" if category else "카테고리: 일반"
    
    return f"""<|im_start|>system
당신은 검색 쿼리 최적화 전문가입니다.
사용자의 자연어 질문을 효과적인 검색 쿼리로 변환해주세요.

원칙:
- 핵심 키워드 포함
- 동의어, 유사어 활용  
- 영어/한국어 혼용 고려
- 불필요한 조사 제거

JSON 형식으로만 응답:
{{
    "queries": ["주요 쿼리", "대안 쿼리1", "대안 쿼리2"]
}}
<|im_end|>

<|im_start|>user
사용자 질문: '{query}'
{category_info}

이 질문에 대한 최적화된 검색 쿼리를 생성해주세요.
<|im_end|>

<|im_start|>assistant
"""

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