import os
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

LLM_MODEL_NAME = r"/home/root/Gukbap-Qwen2.5-7B"

_cached_model = None
_cached_tokenizer = None

def get_llm_model_and_tokenizer():
    global _cached_model, _cached_tokenizer
    
    # 캐시된 모델이 있으면 바로 반환
    if _cached_model is not None and _cached_tokenizer is not None:
        print("기존 모델 재사용!")
        return _cached_model, _cached_tokenizer
    
    print("새 모델 로딩...")
    try:
        torch.cuda.empty_cache()  # 메모리 정리

        # 토크나이저 로드 최적화: 병렬 처리 옵션 활성화
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            use_fast=True,  # 빠른 토크나이저 사용
            padding_side="left",  # 왼쪽 패딩 (생성 모델에 적합)
            use_auth_token=None,  # 인증 토큰 불필요 시 명시적으로 None
            trust_remote_code=True,  # 원격 코드 신뢰 (일부 모델에 필요)
        )

        # 특수 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Qwen2.5 모델에 최적화된 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # float16 -> bfloat16로 변경 (Qwen 최적화)
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=False  # A6000 환경에서는 비활성화하는 것이 더 효율적
        )

        # 모델 로드 최적화 설정
        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.bfloat16,  # float16 -> bfloat16로 변경 (Qwen 최적화)
            "device_map": "auto",  # 자동 장치 맵핑
            "revision": "main",
            "low_cpu_mem_usage": True,  # 낮은 CPU 메모리 사용
            "attn_implementation": "flash_attention_2",  # Flash Attention 2 사용 (지원 시)
            "use_cache": True,  # KV 캐시 활성화
            "trust_remote_code": True,  # 원격 코드 신뢰
        }

        # 모델 로드 시도
        try:
            # 먼저 Flash Attention으로 로드 시도
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                **model_kwargs
            )
            print("LLM model loaded successfully with Flash Attention.")
        except Exception as flash_att_error:
            print(f"Flash Attention 로드 실패, 표준 방식으로 재시도: {flash_att_error}")
            # Flash Attention 실패 시 일반 방식으로 로드
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                **model_kwargs
            )
            print("LLM model loaded successfully with standard attention.")

        # 모델 최적화 설정 (추론 전용)
        model.eval()  # 평가 모드 설정

        # 모델 메모리 사용 정보 출력 (옵션)
        if torch.cuda.is_available():
            print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        # 캐시에 저장하고 반환
        _cached_model = model
        _cached_tokenizer = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"LLM 모델 또는 토크나이저 로딩 중 오류 발생: {e}")
        traceback.print_exc()
        return None, None
