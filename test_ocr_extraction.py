from pdf2image import convert_from_path
import subprocess
import os

# PDF 파일 경로 설정
pdf_path = "/home/test_code/test01/rag-chatbot/backend/app/static/uploads/35afc81e-32ad-4c20-8c07-c1d467827b89_NDT 이미지만 있는 문서.pdf"
output_image = "test_output.jpg"

# PDF의 첫 페이지를 500 DPI로 JPG로 변환
print("PDF를 JPG로 변환 중... DPI=500")
page = convert_from_path(pdf_path, dpi=500, first_page=1, last_page=1)[0]
page.save(output_image, "JPEG", quality=95)
print(f"JPG로 저장 완료: {output_image}, 크기: {page.size}")

# PaddleOCR CLI 호출
print("\nPaddleOCR CLI를 호출하여 텍스트 추출 중...")
command = f"paddleocr --image_dir {output_image} --lang korean --show_log=True --output ./output"
try:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print("PaddleOCR 결과:")
    print(result.stdout)
    if result.stderr:
        print("에러 메시지:")
        print(result.stderr)
    # 결과를 파일에 저장
    result_file = os.path.join('./output', output_image.replace(".jpg", "_ocr_result.txt"))
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(f"OCR 결과가 {result_file}에 저장되었습니다.")
except Exception as e:
    print(f"PaddleOCR 실행 중 오류 발생: {e}")
