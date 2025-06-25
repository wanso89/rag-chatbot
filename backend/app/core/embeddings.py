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


