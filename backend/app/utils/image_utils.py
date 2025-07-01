import fitz  
from pathlib import Path

IMAGE_DIR = Path("app/static/document_images")

def extract_images_from_pdf(pdf_path: str, doc_id: str) -> list[str]:
    """
    PDF에서 모든 이미지를 추출해 IMAGE_DIR/clean_doc_id 에 저장
    반환값: 저장된 이미지 파일들의 상대경로 리스트
    """
    # UUID 제거된 clean doc_id 사용
    from .indexing_utils import strip_uuid_prefix
    clean_doc_id = strip_uuid_prefix(doc_id) if doc_id else "unknown"
    if '.' in clean_doc_id:
        clean_doc_id = clean_doc_id.rsplit('.', 1)[0]
    
    img_dir = IMAGE_DIR / clean_doc_id     # 문서별 하위 폴더 (UUID 제거됨)
    img_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    with fitz.open(pdf_path) as pdf:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(pdf, xref)
                if pix.n > 4:              # CMYK → RGB 변환
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_name = f"{page_index+1}_{img_index+1}.png"
                img_path = img_dir / img_name
                pix.save(img_path)
                image_paths.append(str(img_path))
                pix = None                 # 메모리 해제
    return image_paths
