import os
import logging

def check_existing_files(unique_files, uploads_dir):
    print(f"서버 작업 디렉토리: {os.getcwd()}")
    print(f"uploads_dir: {uploads_dir}")
    existing_files = []
    
    for filename in unique_files:
        file_path = os.path.join(uploads_dir, filename)
        print(f"파일 경로: {file_path}")
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"파일 확인됨: {filename}")
        else:
            print(f"파일 없음: {filename}")
    
    return existing_files

# 테스트
test_files = ['40f7926e-cc46-490a-996e-a8664ffc648e_주요정보통신기반시설_기술적_취약점_분석_평가_방법_상세가이드.pdf']
result = check_existing_files(test_files, 'static/uploads')
print(f'결과: {result}')
