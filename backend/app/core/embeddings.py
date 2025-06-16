import os
import traceback
import asyncio
from typing import List, Union, Any
from datetime import datetime

from elasticsearch.helpers import bulk 
from elasticsearch import AsyncElasticsearch

import torch
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ---------- 환경 변수 / 상수 ----------
EMBEDDING_MODEL_NAME = "/home/root/kpf-sbert-v1.1"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B"


# 기본값을 Qwen으로 바꾸려면 "qwen", 기존 SBERT를 유지하려면 "sbert"
DEFAULT_EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sbert").lower()

# ---------- SBERT 래퍼 ----------
class LangchainEmbeddingFunction:
    def __init__(self, model_name: str):
        try:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
                cache_folder="./.cache",
                multi_process=False
            )
            print(f"[Emb] SBERT 로드 성공 → {model_name}")
        except Exception as e:
            print(f"[Emb] SBERT 로드 실패 → CPU 재시도: {e}")
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"}
            )

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # 문자열 보장
        texts = [str(t) for t in texts]

        batch_size = 8
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            out.extend(self.embeddings_model.embed_documents(batch))
        return out


# ---------- Qwen 임베딩 ----------
class QwenEmbeddingModel:
    def __init__(self, model_name: str = QWEN_MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Emb] Qwen 로딩 중… ({model_name})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(self.device).eval()
        print(f"[Emb] Qwen 로딩 완료 ({self.device})")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().tolist()

    def __call__(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            return self._embed_batch([texts])[0]
        return self._embed_batch(texts)

# ---------- 싱글턴 ----------
_qwen_singleton: Union[QwenEmbeddingModel, None] = None
def get_qwen_embedding_function():
    global _qwen_singleton
    if _qwen_singleton is None:
        _qwen_singleton = QwenEmbeddingModel()
    return _qwen_singleton

# ---------- 팩토리 ----------
def get_embedding_function():
    backend = DEFAULT_EMBEDDING_BACKEND
    print(f"[Emb] backend = {backend}")

    if backend == "qwen":
        try:
            return get_qwen_embedding_function()
        except Exception as e:
            print(f"[Emb] Qwen 실패 → SBERT 대체 ({e})")

    # fallback = SBERT
    try:
        return LangchainEmbeddingFunction(EMBEDDING_MODEL_NAME)
    except Exception:
        traceback.print_exc()
        return None

# 기존 index_chunks_to_elasticsearch 함수에서 사용
async def index_chunks_to_elasticsearch_with_qwen(
    es_client: Any, 
    chunks: List[Document], 
    category: str,
    use_qwen_embedding: bool = True
):
    """
    Qwen 임베딩을 사용한 Elasticsearch 인덱싱
    (기존 함수를 감싸는 래퍼)
    """
    
    if use_qwen_embedding:
        # Qwen 임베딩 함수 사용
        qwen_embedding_function = get_qwen_embedding_function()
        if qwen_embedding_function:
            print("✅ Qwen 임베딩 모델 사용")
            embedding_function = qwen_embedding_function
        else:
            print("⚠️ Qwen 임베딩 실패, 기존 모델 사용")
            # 기존 embedding_function 사용 (여기서 import)
            from your_existing_module import embedding_function
    else:
        # 기존 embedding_function 사용
        from your_existing_module import embedding_function
    
    # 기존 index_chunks_to_elasticsearch 함수 호출
    return await index_chunks_to_elasticsearch(
        es_client, embedding_function, chunks, category
    )

# 이미지가 포함된 텍스트 특별 처리
def enhance_text_for_embedding(doc: Document) -> str:
    """임베딩을 위한 텍스트 강화 (이미지 컨텍스트 포함)"""
    content = doc.page_content
    element_type = doc.metadata.get('element_type', 'text')
    
    # 이미지인 경우 컨텍스트 정보 추가
    if element_type == 'image':
        smart_caption = doc.metadata.get('smart_caption', '')
        if smart_caption:
            # "이미지"를 명시적으로 표시하여 검색 시 구분
            content = f"[이미지] {smart_caption}"
    
    # 테이블인 경우 구조 정보 추가
    elif element_type == 'table':
        table_caption = doc.metadata.get('table_caption', '')
        if table_caption:
            content = f"[테이블] {table_caption}\n{content}"
        else:
            content = f"[테이블]\n{content}"
    
    return content

# 기존 인덱싱 함수 업데이트 (최소 변경)
async def index_chunks_to_elasticsearch_enhanced(
    es_client: AsyncElasticsearch,
    chunks: List[Document],
    category: str,
) -> bool:
    """
    Qwen/SBERT 임베딩 + 이미지·테이블 컨텍스트 강화 버전
    실패하면 예외를 그대로 올려서 호출측에서 처리하도록 함
    """
    if not chunks:
        raise ValueError("chunks가 비어 있음")

    # 1. 임베딩 함수 준비 (환경변수에 따라 Qwen 또는 SBERT)
    embed_fn = get_embedding_function()
    if embed_fn is None:
        raise RuntimeError("임베딩 함수 초기화 실패")

    # 2. 텍스트 강화
    enhanced_texts = [enhance_text_for_embedding(c) for c in chunks]

    # 3. 임베딩 (blocking -> thread 오프로딩)
    embeddings = await asyncio.to_thread(embed_fn, enhanced_texts)

    # 4. ES bulk 문서 구성
    actions = []
    for doc, enh_text, emb in zip(chunks, enhanced_texts, embeddings):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", f"{src}_{page}")

        actions.append({
            "_index": ES_INDEX_NAME,
            "_id": f"{src.replace('.', '_')}_{page}_{chunk_id}",
            "_source": {
                "text": doc.page_content,
                "enhanced_text": enh_text,
                "embedding": emb,
                "source": src,
                "page": page,
                "category": category,
                "indexed_at": datetime.utcnow().isoformat() + "Z",
                "element_type": doc.metadata.get("element_type", "text"),
            },
        })

    # 5. Bulk 인서트 (실패 시 예외 그대로 throw)
    success, _ = await asyncio.to_thread(
        bulk, es_client, actions,
        chunk_size=len(actions),
        request_timeout=180,
        max_retries=2,
        raise_on_error=True,   # 실패하면 Exception 발생
    )
    print(f"[ES] {success} docs indexed")
    return True