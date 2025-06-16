"""
SQLCoder 모듈 초기화 유틸리티
"""

import os
import traceback
from typing import Tuple

def initialize_sqlcoder() -> Tuple[bool, str]:
    """
    SQLCoder 모델을 초기화합니다.
    
    Returns:
        Tuple[bool, str]: 초기화 성공 여부와 메시지
    """
    try:
        # SQLCoder 관련 모듈 가져오기
        try:
            from app.utils.sqlcoder_utils import load_sqlcoder_model, test_db_connection
        except ImportError:
            return False, "SQLCoder 모듈을 임포트할 수 없습니다. 관련 파일이 올바르게 설치되었는지 확인하세요."
        
        # 데이터베이스 연결 테스트
        try:
            db_connected = test_db_connection()
            
            if not db_connected:
                return False, "SQLCoder 데이터베이스 연결 실패"
            
        except Exception as db_err:
            return False, f"SQLCoder 데이터베이스 연결 테스트 실패: {str(db_err)}"
            
    except Exception as e:
        traceback.print_exc()
        return False, f"SQLCoder 초기화 중 오류: {str(e)}" 