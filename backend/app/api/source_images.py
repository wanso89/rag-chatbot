from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

class SourceImageRequest(BaseModel):
    source_path: str
    page: Optional[int] = 1

class ImageInfo(BaseModel):
    url: str
    path: str
    page: int
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

class SourceImageResponse(BaseModel):
    status: str
    message: Optional[str] = None
    images: List[ImageInfo] = []

@router.post("/source-images")
async def get_source_images(request: SourceImageRequest):
    """출처 문서에서 이미지를 가져옵니다."""
    try:
        logger.info(f"이미지 요청: {request.source_path}, 페이지: {request.page}")
        
        # 파일 경로 정리
        source_path = request.source_path
        if source_path.startswith('/'):
            source_path = source_path[1:]
        
        # 절대 경로 생성
        full_path = Path(source_path).resolve()
        
        # 파일 존재 확인
        if not full_path.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {full_path}")
            return SourceImageResponse(
                status="error",
                message="파일을 찾을 수 없습니다."
            )
        
        # 이미지 폴더 경로 생성
        image_folder = get_image_folder_path(source_path)
        
        if not os.path.exists(image_folder):
            logger.warning(f"이미지 폴더를 찾을 수 없습니다: {image_folder}")
            return SourceImageResponse(
                status="error",
                message="이미지 폴더를 찾을 수 없습니다."
            )
        
        # 이미지 파일 목록 가져오기
        images = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in supported_formats):
                file_path = os.path.join(image_folder, file)
                
                # 페이지 번호 추출 (파일명에서)
                page_num = extract_page_number(file)
                
                # 특정 페이지만 요청된 경우 필터링
                if request.page and page_num != request.page:
                    continue
                
                # 이미지 URL 생성 (정적 파일 서빙용)
                image_url = f"/static/document_images/{os.path.basename(image_folder)}/{file}"
                
                # OCR 텍스트 파일 확인
                ocr_text = get_ocr_text(file_path)
                
                images.append(ImageInfo(
                    url=image_url,
                    path=file_path,
                    page=page_num,
                    caption=f"페이지 {page_num} 이미지",
                    ocr_text=ocr_text
                ))
        
        # 페이지 번호로 정렬
        images.sort(key=lambda x: x.page)
        
        logger.info(f"이미지 {len(images)}개를 찾았습니다.")
        
        return SourceImageResponse(
            status="success",
            images=images
        )
        
    except Exception as e:
        logger.error(f"이미지 가져오기 중 오류 발생: {str(e)}")
        return SourceImageResponse(
            status="error",
            message=f"이미지 가져오기 중 오류가 발생했습니다: {str(e)}"
        )

def get_image_folder_path(source_path: str) -> str:
    """소스 파일 경로로부터 이미지 폴더 경로를 생성합니다."""
    # 파일명에서 확장자 제거
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    
    # UUID가 포함된 경우 제거
    import re
    uuid_pattern = r'^[a-f0-9]{8,}-?[a-f0-9-]*_'
    clean_name = re.sub(uuid_pattern, '', base_name, flags=re.IGNORECASE)
    
    # 이미지 폴더 경로 생성
    image_folder = os.path.join("static", "document_images", clean_name)
    
    return image_folder

def extract_page_number(filename: str) -> int:
    """파일명에서 페이지 번호를 추출합니다."""
    import re
    
    # 파일명에서 페이지 번호 패턴 찾기
    patterns = [
        r'page[_-]?(\d+)',
        r'p[_-]?(\d+)',
        r'(\d+)\.(?:jpg|jpeg|png|gif|bmp|webp)$'
    ]
    
    filename_lower = filename.lower()
    
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            return int(match.group(1))
    
    # 패턴을 찾을 수 없으면 1로 반환
    return 1

def get_ocr_text(image_path: str) -> Optional[str]:
    """이미지에 대응하는 OCR 텍스트를 가져옵니다."""
    try:
        # OCR 텍스트 파일 경로 생성
        base_name = os.path.splitext(image_path)[0]
        ocr_file = f"{base_name}.txt"
        
        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        return None
        
    except Exception as e:
        logger.warning(f"OCR 텍스트 읽기 실패: {str(e)}")
        return None
