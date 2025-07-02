# synonym_builder.py
"""
Qwen7B 기반 동의어 사전 자동 구축 시스템

업로드된 문서들을 Qwen7B로 분석하여 자동으로 동의어/유사어 사전을 구축하고,
retriever에서 동적으로 활용할 수 있도록 제공합니다.
"""

import re
import json
import asyncio
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class QwenSynonymBuilder:
    def __init__(self, synonym_file_path: str = "app/data/qwen_synonyms.json"):
        self.synonym_file_path = Path(synonym_file_path)
        self.synonym_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 자동 구축된 동의어 사전
        self.synonyms = {}
        
        # Qwen 모델 (외부에서 주입)
        self.llm_model = None
        self.tokenizer = None
        
        # 로드
        self.load_synonyms()
    
    def set_models(self, llm_model, tokenizer):
        """Qwen 모델 설정"""
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        logger.info("Qwen 모델이 동의어 빌더에 설정되었습니다")
    
    def load_synonyms(self):
        """저장된 동의어 사전 로드"""
        try:
            if self.synonym_file_path.exists():
                with open(self.synonym_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.synonyms = data.get('synonyms', {})
                    logger.info(f"Qwen 동의어 사전 로드 완료: {len(self.synonyms)}개 용어")
            else:
                logger.info("동의어 사전 파일 없음, 새로 생성")
        except Exception as e:
            logger.error(f"동의어 사전 로드 실패: {e}")
            self.synonyms = {}
    
    def save_synonyms(self):
        """동의어 사전 저장"""
        try:
            data = {
                'synonyms': self.synonyms,
                'last_updated': datetime.now().isoformat(),
                'stats': {
                    'total_terms': len(self.synonyms),
                    'total_synonyms': sum(len(syns) for syns in self.synonyms.values())
                }
            }
            
            with open(self.synonym_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Qwen 동의어 사전 저장 완료: {data['stats']}")
            
        except Exception as e:
            logger.error(f"동의어 사전 저장 실패: {e}")
    
    async def extract_key_terms_with_qwen(self, document_text: str) -> List[str]:
        """Qwen을 사용하여 문서에서 핵심 용어 추출"""
        
        if not self.llm_model or not self.tokenizer:
            logger.warning("Qwen 모델이 설정되지 않음")
            return []
        
        # 텍스트가 너무 길면 첫 3000자 사용 (제한 완화)
        if len(document_text) > 3000:
            text_sample = document_text[:3000]
        else:
            text_sample = document_text
        
        prompt = f"""다음 문서에서 중요한 용어들을 추출해주세요. 회사명, 제품명, 기술용어, 시스템명 등을 포함해주세요. 단, 중국어는 사용하면 안됩니다.

문서 내용:
{text_sample}

추출 규칙:
1. 고유명사 위주로 추출 (회사명, 제품명, 시스템명 등)
2. 기술 관련 용어 (소프트웨어, 하드웨어, 프로토콜 등)
3. 줄임말/약어 포함
4. 각 용어는 한 줄씩 나열
5. 최대 100개까지 추출 (제한 완화)

핵심 용어 목록:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=500,  # 더 많은 용어 생성을 위해 증가
                    temperature=0.3,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # 추출된 용어들 파싱
            terms = []
            for line in generated_text.split('\n'):
                term = line.strip()
                # 불필요한 접두어 제거
                term = re.sub(r'^[\d\-\.\*\+\s]*', '', term)
                term = term.strip()
                
                if term and len(term) >= 2 and len(term) <= 30:
                    # 특수문자 제거
                    clean_term = re.sub(r'[^\w\s가-힣]', '', term).strip()
                    if clean_term and len(clean_term) >= 2:
                        terms.append(clean_term)
            
            # 중복 제거 및 최대 100개 제한 (제한 완화)
            unique_terms = list(dict.fromkeys(terms))[:100]
            logger.info(f"Qwen 핵심 용어 추출 완료: {len(unique_terms)}개")
            
            return unique_terms
            
        except Exception as e:
            logger.error(f"Qwen 핵심 용어 추출 실패: {e}")
            return []
    
    async def generate_synonyms_with_qwen(self, term: str, context: str = "") -> List[str]:
        """Qwen을 사용하여 특정 용어의 동의어/유사어 생성"""
        
        if not self.llm_model or not self.tokenizer:
            return [term]
        
        # 컨텍스트 준비
        context_text = ""
        if context and len(context) > 200:
            context_text = f"\n\n참고 문맥: {context[:200]}..."
        elif context:
            context_text = f"\n\n참고 문맥: {context}"
        
        prompt = f"""'{term}'이라는 용어의 동의어, 유사어, 다른 표현들을 생성해주세요.{context_text}

생성 규칙:
1. 한국어와 영어 표현 모두 포함
2. 줄임말, 약어, 전체 형태 모두 포함
3. 업계에서 흔히 사용하는 다른 표현들
4. 문맥상 같은 의미로 사용될 수 있는 용어들
5. 각 동의어는 한 줄씩 나열
6. 최대 20개까지 생성 (제한 완화)
7. 중국어는 사용하지 말 것
'{term}'의 동의어 목록:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=300,  # 더 많은 동의어 생성을 위해 증가
                    temperature=0.4,  # 창의성을 위해 약간 높임
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # 동의어들 파싱
            synonyms = [term]  # 원본 용어 포함
            
            for line in generated_text.split('\n'):
                synonym = line.strip()
                # 불필요한 접두어 제거
                synonym = re.sub(r'^[\d\-\.\*\+\s]*', '', synonym)
                synonym = synonym.strip()
                
                if synonym and len(synonym) >= 2 and len(synonym) <= 30:
                    # 특수문자 제거 (단, 필요한 것들은 유지)
                    clean_synonym = re.sub(r'[^\w\s가-힣\+\-]', '', synonym).strip()
                    if clean_synonym and clean_synonym.lower() != term.lower():
                        synonyms.append(clean_synonym)
            
            # 중복 제거 및 최대 20개 제한 (제한 완화)
            unique_synonyms = list(dict.fromkeys(synonyms))[:20]
            logger.info(f"'{term}' 동의어 생성 완료: {len(unique_synonyms)}개")
            
            return unique_synonyms
            
        except Exception as e:
            logger.error(f"Qwen 동의어 생성 실패 ({term}): {e}")
            return [term]
    
    async def analyze_documents_batch(self, documents: List[Any], batch_size: int = 5):
        """문서들을 배치로 분석하여 동의어 사전 구축"""
        
        if not self.llm_model or not self.tokenizer:
            logger.error("Qwen 모델이 설정되지 않음. 동의어 분석 불가.")
            return
        
        logger.info(f"Qwen 동의어 분석 시작: {len(documents)}개 문서")
        
        new_terms = set()
        
        # 배치별로 문서 처리
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"배치 {i//batch_size + 1} 처리 중... ({len(batch)}개 문서)")
            
            for doc in batch:
                try:
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    
                    # 핵심 용어 추출
                    key_terms = await self.extract_key_terms_with_qwen(content)
                    new_terms.update(key_terms)
                    
                    # 메모리 절약을 위해 잠시 대기
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"문서 분석 실패: {e}")
                    continue
        
        logger.info(f"총 {len(new_terms)}개 핵심 용어 추출 완료")
        
        # 새로운 용어들에 대해 동의어 생성
        synonym_tasks = []
        for term in list(new_terms)[:100]:  # 더 많은 용어 처리 (20개 -> 100개)
            if term not in self.synonyms:
                synonym_tasks.append(self._generate_term_synonyms(term))
        
        if synonym_tasks:
            logger.info(f"{len(synonym_tasks)}개 용어의 동의어 생성 중...")
            results = await asyncio.gather(*synonym_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    self.synonyms.update(result)
        
        # 저장
        self.save_synonyms()
        logger.info(f"Qwen 동의어 분석 완료: 총 {len(self.synonyms)}개 용어")
    
    async def _generate_term_synonyms(self, term: str) -> Dict[str, List[str]]:
        """개별 용어의 동의어 생성 (내부 함수)"""
        try:
            synonyms = await self.generate_synonyms_with_qwen(term)
            return {term: synonyms}
        except Exception as e:
            logger.error(f"용어 '{term}' 동의어 생성 실패: {e}")
            return {}
    
    def get_synonyms(self, term: str) -> List[str]:
        """특정 용어의 동의어 목록 반환"""
        # 직접 매칭
        if term in self.synonyms:
            return self.synonyms[term]
        
        # 대소문자 무시하고 매칭
        term_lower = term.lower()
        for key, synonyms in self.synonyms.items():
            if key.lower() == term_lower:
                return synonyms
            
            # 동의어 목록에서 매칭
            for synonym in synonyms:
                if synonym.lower() == term_lower:
                    return synonyms
        
        return [term]  # 동의어가 없으면 원본만 반환
    
    def get_expanded_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 목록을 동의어로 확장"""
        expanded = set(keywords)
        
        for keyword in keywords:
            synonyms = self.get_synonyms(keyword)
            expanded.update(synonyms)
        
        # 길이순 정렬 (긴 것부터)
        return sorted(list(expanded), key=len, reverse=True)
    
    def get_all_synonyms(self) -> Dict[str, List[str]]:
        """전체 동의어 사전 반환"""
        return self.synonyms.copy()
    
    async def add_manual_synonyms(self, term: str, synonyms: List[str]):
        """수동으로 동의어 추가"""
        if term not in self.synonyms:
            self.synonyms[term] = [term]
        
        # 기존 동의어와 병합
        existing = set(self.synonyms[term])
        new_synonyms = set(synonyms)
        self.synonyms[term] = list(existing | new_synonyms)
        
        self.save_synonyms()
        logger.info(f"수동 동의어 추가: '{term}' -> {synonyms}")

# 전역 인스턴스
_qwen_synonym_builder = None

def get_qwen_synonym_builder() -> QwenSynonymBuilder:
    """Qwen 동의어 빌더 싱글톤 반환"""
    global _qwen_synonym_builder
    if _qwen_synonym_builder is None:
        _qwen_synonym_builder = QwenSynonymBuilder()
    return _qwen_synonym_builder

def set_qwen_models_for_synonyms(llm_model, tokenizer):
    """동의어 빌더에 Qwen 모델 설정"""
    builder = get_qwen_synonym_builder()
    builder.set_models(llm_model, tokenizer)

async def update_synonyms_from_indexing(documents: List[Any]):
    """인덱싱 시 Qwen 기반 동의어 사전 자동 업데이트"""
    try:
        builder = get_qwen_synonym_builder()
        await builder.analyze_documents_batch(documents)
    except Exception as e:
        logger.error(f"Qwen 동의어 사전 업데이트 실패: {e}")

async def update_synonyms_from_text(combined_text: str, max_length: int = 8000):
    """
    결합된 텍스트를 기반으로 동의어를 업데이트합니다. (ES 기반 동의어 구축용)
    """
    if not combined_text or not combined_text.strip():
        logger.warning("빈 텍스트가 전달되어 동의어 구축을 건너뜁니다")
        return
    
    builder = get_qwen_synonym_builder()
    
    if not builder.llm_model or not builder.tokenizer:
        logger.error("Qwen 모델이 설정되지 않음. 동의어 분석 불가.")
        return
    
    logger.info(f"📝 텍스트 길이: {len(combined_text):,}자")
    
    # 텍스트를 적절한 크기로 청킹
    chunks = []
    if len(combined_text) > max_length:
        # 문단 단위로 분할
        paragraphs = combined_text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) < max_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        chunks = [combined_text]
    
    logger.info(f"📦 텍스트를 {len(chunks)}개 청크로 분할")
    
    # 각 청크에서 핵심 용어 추출
    all_terms = set()
    for i, chunk in enumerate(chunks):
        try:
            logger.info(f"🔍 청크 {i+1}/{len(chunks)} 분석 중...")
            
            # 청크 미리보기 (처음 50자)
            preview = chunk.strip()[:50].replace('\n', ' ')
            logger.info(f"   📄 청크 내용: '{preview}...'")
            
            # 핵심 용어 추출
            key_terms = await builder.extract_key_terms_with_qwen(chunk)
            all_terms.update(key_terms)
            
            logger.info(f"   ✅ 청크 {i+1}에서 {len(key_terms)}개 용어 추출 완료")
            if key_terms:
                logger.info(f"   🔤 추출된 용어: {key_terms[:5]}")  # 처음 5개만 표시
            
        except Exception as e:
            logger.error(f"❌ 청크 {i+1} 분석 실패: {e}")
            continue
    
    logger.info(f"🎯 총 {len(all_terms)}개 고유 핵심 용어 추출 완료")
    
    # 새로운 용어들에 대해 동의어 생성
    new_terms = [term for term in all_terms if term not in builder.synonyms]
    
    if not new_terms:
        logger.info("🔄 새로운 용어가 없어 동의어 생성을 건너뜁니다")
        return
    
    # 메모리 고려해서 최대 200개까지 처리 (제한 완화)
    terms_to_process = new_terms[:200]
    logger.info(f"🤖 {len(terms_to_process)}개 신규 용어의 동의어 생성 시작...")
    
    generated_count = 0
    for i, term in enumerate(terms_to_process):
        try:
            logger.info(f"   🔄 {i+1}/{len(terms_to_process)}: '{term}' 동의어 생성 중...")
            
            synonyms = await builder.generate_synonyms_with_qwen(term, combined_text[:300])
            builder.synonyms[term] = synonyms
            generated_count += 1
            
            logger.info(f"   ✅ '{term}' → {len(synonyms)}개 동의어: {synonyms}")
            
            # GPU 메모리 여유를 위해 잠시 대기
            await asyncio.sleep(0.2)
            
        except Exception as e:
            logger.error(f"   ❌ '{term}' 동의어 생성 실패: {e}")
            continue
    
    # 저장
    builder.save_synonyms()
    
    logger.info(f"🎉 동의어 구축 완료: {generated_count}개 용어 처리, 총 {len(builder.synonyms)}개 용어 보유")
