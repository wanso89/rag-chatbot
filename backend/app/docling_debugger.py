# simple_docling_test.py
import asyncio
import traceback
import torch
from pathlib import Path

# 실제 운영 함수들 import
from utils.indexing_utils import get_docling_converter, process_and_index_file
from core.elasticsearch import get_elasticsearch_client
from core.embeddings import get_embedding_function

async def test_real_docling_workflow():
    """
    실제 운영 환경과 동일한 방식으로 Docling을 테스트합니다.
    reindex_all_files API가 하는 일을 정확히 따라합니다.
    """
    
    print("=== 실제 운영 함수 직접 테스트 ===")
    
    # 1. 환경 준비 (API와 동일)
    uploads_dir = Path("app/static/uploads")
    
    if not uploads_dir.exists():
        print("❌ uploads 폴더가 없습니다")
        return
    
    files = [f for f in uploads_dir.iterdir() if f.is_file()]
    if not files:
        print("❌ 테스트할 파일이 없습니다")
        return
    
    # GPU 메모리 모니터링 시작
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 메모리 시작: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # 2. 실제 처리 함수들 준비 (API와 동일)
    try:
        es_client = get_elasticsearch_client()
        embed_fn = get_embedding_function()
        
        if not es_client:
            print("⚠️ Elasticsearch 연결 실패, 계속 진행")
        
        print(f"📁 총 {len(files)}개 파일 발견")
        
    except Exception as e:
        print(f"❌ 환경 준비 실패: {e}")
        return
    
    # 3. 파일별 개별 테스트
    for i, file_path in enumerate(files[:10], 1):  # 처음 10개만 테스트
        print(f"\n--- 파일 {i}: {file_path.name} ---")
        
        # GPU 메모리 상태 확인
        if torch.cuda.is_available():
            before_gpu = torch.cuda.memory_allocated(0) / 1024**3
            print(f"처리 전 GPU: {before_gpu:.2f} GB")
        
        try:
            # 실제 API와 동일한 방식으로 처리
            success = await process_and_index_file(
                es_client=es_client,
                embedding_function=embed_fn,
                uploaded_file_path=str(file_path),
                category="테스트",
                reindex=False  # 실제 인덱싱은 하지 않음
            )
            
            # GPU 메모리 상태 확인
            if torch.cuda.is_available():
                after_gpu = torch.cuda.memory_allocated(0) / 1024**3
                print(f"처리 후 GPU: {after_gpu:.2f} GB (증가: {after_gpu - before_gpu:.2f} GB)")
            
            if success:
                print("✅ 성공")
            else:
                print("⚠️ 실패 (하지만 오류는 없음)")
                
        except Exception as e:
            print(f"❌ 오류 발생: {type(e).__name__}: {str(e)}")
            
            # Meta tensor 오류인지 확인
            if "Cannot copy out of meta tensor" in str(e):
                print("🎯 Meta tensor 오류 발견!")
                print("상세 스택 트레이스:")
                traceback.print_exc()
                
                # 이 지점에서 상세 분석
                print("\n=== Meta Tensor 오류 상세 분석 ===")
                print(f"파일: {file_path.name}")
                print(f"크기: {file_path.stat().st_size / 1024:.2f} KB")
                print(f"확장자: {file_path.suffix}")
                
                # GPU 상태 출력
                if torch.cuda.is_available():
                    print(f"GPU 메모리: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                    print(f"예약된 메모리: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                
                break  # 첫 번째 meta tensor 오류에서 중단
            else:
                print("다른 종류의 오류입니다")
                print(f"오류 내용: {str(e)}")

# 직접 Docling 컨버터 테스트
async def test_docling_converter_directly():
    """
    get_docling_converter 함수를 직접 테스트합니다.
    """
    
    print("\n=== Docling 컨버터 직접 테스트 ===")
    
    try:
        converter = get_docling_converter()
        
        if converter is None:
            print("❌ Docling 컨버터 생성 실패")
            return
        
        print("✅ Docling 컨버터 생성 성공")
        
        # 간단한 파일 하나로 테스트
        uploads_dir = Path("app/static/uploads")
        pdf_files = [f for f in uploads_dir.glob("*.pdf")]
        
        if not pdf_files:
            print("❌ 테스트할 PDF 파일이 없습니다")
            return
        
        test_file = pdf_files[0]
        print(f"테스트 파일: {test_file.name}")
        
        # GPU 메모리 모니터링
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated(0) / 1024**3
            print(f"변환 전 GPU: {before:.2f} GB")
        
        # 실제 변환 수행
        result = converter.convert(str(test_file))
        
        if torch.cuda.is_available():
            after = torch.cuda.memory_allocated(0) / 1024**3
            print(f"변환 후 GPU: {after:.2f} GB")
        
        print("✅ 변환 성공")
        print(f"결과 타입: {type(result)}")
        
        # 실제 마크다운 출력 테스트
        if hasattr(result.document, 'export_to_markdown'):
            markdown = result.document.export_to_markdown()
            print(f"마크다운 길이: {len(markdown) if markdown else 0} 글자")
            
            if markdown and len(markdown) > 100:
                print("내용 추출 성공 ✅")
            else:
                print("내용 추출 실패 ❌")
        
    except Exception as e:
        print(f"❌ Docling 테스트 중 오류: {type(e).__name__}: {str(e)}")
        
        if "Cannot copy out of meta tensor" in str(e):
            print("🎯 Meta tensor 오류 발견!")
            traceback.print_exc()
        else:
            print("다른 종류의 오류")

async def test_hybrid_document_creation():
    """
    create_hybrid_documents 함수의 데이터 흐름을 상세히 진단합니다.
    metadata 구조 문제를 정확히 파악하기 위한 전용 테스트입니다.
    """
    
    print("\n=== 하이브리드 문서 생성 진단 ===")
    
    uploads_dir = Path("app/static/uploads")
    pdf_files = [f for f in uploads_dir.glob("*.pdf")]
    
    if not pdf_files:
        print("❌ 테스트할 PDF 파일이 없습니다")
        return
    
    test_file = pdf_files[0]
    print(f"진단 대상 파일: {test_file.name}")
    
    try:
        # PyPDFLoader로 텍스트 추출
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(test_file))
        pages = loader.load()
        
        # PDF 텍스트를 딕셔너리로 변환 (create_hybrid_documents 입력 형태)
        pdf_texts = {i+1: page.page_content for i, page in enumerate(pages)}
        
        print(f"추출된 페이지 수: {len(pdf_texts)}")
        
        # Docling 레이아웃 분석 시도
        converter = get_docling_converter()
        if converter:
            docling_result = converter.convert(str(test_file))
            layout_info = docling_result.document if hasattr(docling_result, 'document') else None
            
            print(f"Docling 레이아웃 정보: {type(layout_info)}")
            
            # 표 정보 상세 확인
            if layout_info and hasattr(layout_info, 'tables'):
                tables = getattr(layout_info, 'tables', [])
                print(f"감지된 표 개수: {len(tables)}")
                
                # 각 표 객체의 구조 진단
                for idx, tbl in enumerate(tables[:3]):  # 처음 3개만 확인
                    print(f"표 {idx}: type = {type(tbl)}")
                    print(f"표 {idx}: attributes = {dir(tbl)}")
                    
                    # page 속성 확인
                    if hasattr(tbl, 'page'):
                        page_val = getattr(tbl, 'page', None)
                        print(f"표 {idx}: page = {page_val} (type: {type(page_val)})")
                    
                    # markdown 생성 테스트
                    if hasattr(tbl, 'export_to_markdown'):
                        md = tbl.export_to_markdown()
                        print(f"표 {idx}: markdown 길이 = {len(md) if md else 0}")
        
        # 실제 create_hybrid_documents 호출 전 준비 상태 확인
        print("\n--- create_hybrid_documents 호출 준비 완료 ---")
        print("이제 실제 함수를 호출하여 어느 지점에서 오류가 발생하는지 확인합니다.")
        
    except Exception as e:
        print(f"❌ 진단 중 오류: {type(e).__name__}: {str(e)}")
        if "'list' object has no attribute 'get'" in str(e):
            print("🎯 목표 오류 발견!")
            traceback.print_exc()

async def test_parse_ocr_table_structure():
    """
    parse_ocr_table과 그 하위 함수들이 생성하는 Document 구조를 진단합니다.
    """
    
    print("\n=== parse_ocr_table Document 구조 진단 ===")
    
    # 여러 타입의 표 샘플 테스트
    test_cases = {
        "pipe_table": "이름|나이|직업\n홍길동|30|개발자\n김철수|25|디자이너",
        "tab_table": "이름\t나이\t직업\n홍길동\t30\t개발자\n김철수\t25\t디자이너",
        "space_table": "이름    나이  직업\n홍길동  30    개발자\n김철수  25    디자이너"
    }
    
    try:
        from utils.indexing_utils import parse_ocr_table
        
        for test_name, sample_text in test_cases.items():
            print(f"\n--- {test_name} 테스트 ---")
            
            result_docs = parse_ocr_table(sample_text)
            print(f"반환된 Document 개수: {len(result_docs)}")
            
            for i, doc in enumerate(result_docs):
                print(f"Document {i}:")
                print(f"  metadata 타입: {type(doc.metadata)}")
                
                if isinstance(doc.metadata, dict):
                    print(f"  ✅ 정상 (딕셔너리)")
                    print(f"  내용: {doc.metadata}")
                else:
                    print(f"  ❌ 비정상 ({type(doc.metadata).__name__})")
                    print(f"  내용: {doc.metadata}")
                    
                    # get() 메서드 테스트
                    try:
                        page_val = doc.metadata.get("page", "없음")
                        print(f"  .get() 테스트: 성공 - {page_val}")
                    except AttributeError as e:
                        print(f"  .get() 테스트: 실패 - {e}")
                        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

async def test_actual_hybrid_workflow():
    """
    실제 하이브리드 워크플로우에서 metadata 구조 변화를 추적합니다.
    어느 지점에서 metadata가 리스트로 변하는지 정확히 파악합니다.
    """
    
    print("\n=== 실제 하이브리드 워크플로우 진단 ===")
    
    uploads_dir = Path("app/static/uploads")
    docx_files = [f for f in uploads_dir.glob("*.docx")]
    
    if not docx_files:
        print("❌ 테스트할 DOCX 파일이 없습니다")
        return
    
    test_file = docx_files[0]
    print(f"진단 대상: {test_file.name}")
    
    try:
        # 1. 실제 파일 로딩과 동일한 과정 재현
        from langchain_community.document_loaders import PyPDFLoader
        from utils.indexing_utils import create_hybrid_documents, get_docling_converter
        
        # LibreOffice로 PDF 변환 (실제와 동일)
        import subprocess
        temp_pdf = f"temp_conversions/{test_file.stem}.pdf"
        subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", "temp_conversions", str(test_file)], check=False)
        
        if not Path(temp_pdf).exists():
            print("❌ PDF 변환 실패")
            return
            
        # 2. PyPDF로 텍스트 추출
        loader = PyPDFLoader(temp_pdf)
        pages = loader.load()
        pdf_texts = {i+1: page.page_content for i, page in enumerate(pages)}
        
        # 3. Docling 레이아웃 분석
        converter = get_docling_converter()
        docling_result = converter.convert(temp_pdf) if converter else None
        layout_info = docling_result.document if docling_result and hasattr(docling_result, 'document') else None
        
        print(f"PDF 페이지 수: {len(pdf_texts)}")
        print(f"Docling 레이아웃: {'있음' if layout_info else '없음'}")
        
        # 4. 여기서 create_hybrid_documents 호출하되, 오류 발생 시점 정확히 추적
        print("\n--- create_hybrid_documents 호출 ---")
        
        # 실제 함수 호출로 오류 재현
        result = await create_hybrid_documents(
            pdf_texts=pdf_texts,
            layout_info=layout_info,
            file_path=temp_pdf,
            llm_model=None,
            tokenizer=None
        )
        
        print(f"✅ 성공: {len(result)}개 문서 생성")
        
    except Exception as e:
        print(f"❌ 오류 발생: {type(e).__name__}: {str(e)}")
        if "'list' object has no attribute 'get'" in str(e):
            print("🎯 목표 오류 재현!")
            
            # 스택 트레이스에서 정확한 라인 찾기
            import traceback
            tb = traceback.format_exc()
            print("상세 스택 트레이스:")
            print(tb)
            
            # 어느 라인에서 발생했는지 파악
            lines = tb.split('\n')
            for line in lines:
                if 'create_hybrid_documents' in line and '.py' in line:
                    print(f"🔍 정확한 발생 위치: {line.strip()}")

async def test_pp_structure_output():
    """
    PP-Structure가 실제로 반환하는 데이터 구조를 진단합니다.
    region 객체의 정확한 형태를 파악하기 위한 목적입니다.
    """
    
    print("\n=== PP-Structure 출력 구조 진단 ===")
    
    uploads_dir = Path("app/static/uploads")
    pdf_files = [f for f in uploads_dir.glob("*.pdf")]
    
    if not pdf_files:
        print("❌ 테스트할 PDF 파일이 없습니다")
        return
    
    test_file = pdf_files[0]
    print(f"진단 대상: {test_file.name}")
    
    try:
        from pdf2image import convert_from_path
        from utils.ocr_utils import get_paddle_structure
        import numpy as np
        
        # PDF의 첫 페이지만 변환
        images = convert_from_path(str(test_file), dpi=200, thread_count=2)
        if not images:
            print("❌ PDF 이미지 변환 실패")
            return
            
        first_page = images[0]
        img_array = np.array(first_page)
        
        # PP-Structure 실행
        structure = get_paddle_structure()
        layout_result = structure(img_array)
        
        print(f"layout_result 타입: {type(layout_result)}")
        print(f"layout_result 길이: {len(layout_result) if hasattr(layout_result, '__len__') else '길이 없음'}")
        
        # 첫 번째 region 상세 분석
        if layout_result and len(layout_result) > 0:
            first_region = layout_result[0]
            print(f"\n첫 번째 region:")
            print(f"  타입: {type(first_region)}")
            print(f"  내용: {first_region}")
            
            # 딕셔너리인지 리스트인지 확인
            if isinstance(first_region, dict):
                print(f"  ✅ 딕셔너리 - 키들: {list(first_region.keys())}")
                if 'type' in first_region:
                    print(f"  region_type: {first_region.get('type', '없음')}")
                if 'res' in first_region:
                    res_content = first_region.get('res', {})
                    print(f"  res 타입: {type(res_content)}")
                    print(f"  res 내용: {res_content}")
            elif isinstance(first_region, list):
                print(f"  ❌ 리스트 - 길이: {len(first_region)}")
                print(f"  첫 번째 요소: {first_region[0] if first_region else '비어있음'}")
            else:
                print(f"  ❌ 예상치 못한 타입: {type(first_region)}")
                print(f"  속성들: {dir(first_region) if hasattr(first_region, '__dict__') else '속성 없음'}")
        
        # 처음 3개 region 구조 간단히 확인
        print(f"\n처음 3개 region 구조 요약:")
        for i, region in enumerate(layout_result[:3]):
            print(f"  Region {i}: {type(region)} - {region}")
            
    except Exception as e:
        print(f"❌ PP-Structure 테스트 중 오류: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

async def compare_production_vs_test_processing():
    """
    실제 운영 환경의 처리 방식과 테스트 환경의 처리 방식을 
    동일한 파일로 직접 비교하여 차이점을 찾아냅니다.
    """
    
    print("\n=== 운영 vs 테스트 환경 비교 분석 ===")
    
    # 표가 있다고 확인된 파일 선택
    uploads_dir = Path("/home/test_code/test01/rag-chatbot/backend/app/static/uploads/")
    docx_files = [f for f in uploads_dir.glob("*.pdf")]
    
    if not docx_files:
        print("❌ 비교할 DOCX 파일이 없습니다")
        return
    
    test_file = docx_files[0]  # 표가 있다고 확인된 파일
    print(f"비교 대상 파일: {test_file.name}")
    
    try:
        # 1단계: 운영 환경과 동일한 방식 - LibreOffice 변환
        print("\n--- 1단계: 운영 환경 방식 (LibreOffice 변환) ---")
        import subprocess
        temp_pdf_path = f"temp_conversions/{test_file.stem}.pdf"
        
        # LibreOffice로 변환
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", "pdf", 
            "--outdir", "temp_conversions", str(test_file)
        ], check=False)
        
        if Path(temp_pdf_path).exists():
            print("✅ LibreOffice PDF 변환 성공")
            
            # 변환된 PDF로 표 감지 테스트
            await test_table_detection_on_file(temp_pdf_path, "운영환경방식")
        else:
            print("❌ LibreOffice PDF 변환 실패")
        
        # 2단계: 테스트 환경 방식 - 기존 PDF 직접 사용
        print("\n--- 2단계: 테스트 환경 방식 (기존 PDF) ---")
        pdf_files = [f for f in uploads_dir.glob("*.pdf")]
        if pdf_files:
            direct_pdf = pdf_files[0]  # 직접 PDF 파일 사용
            print(f"직접 PDF 파일: {direct_pdf.name}")
            
            await test_table_detection_on_file(str(direct_pdf), "테스트환경방식")
        
        # 3단계: 결과 비교 분석
        print("\n--- 3단계: 차이점 분석 ---")
        print("두 방식에서 표 감지 결과가 다르다면, 파일 변환 과정에서")
        print("표 구조가 손상되거나 변형되었을 가능성이 높습니다.")
        
    except Exception as e:
        print(f"❌ 비교 분석 중 오류: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

async def test_table_detection_on_file(file_path, method_name):
    """
    특정 파일에서 표 감지가 어떻게 이루어지는지 상세히 분석합니다.
    """
    print(f"\n  [{method_name}] 표 감지 분석:")
    
    try:
        # Docling 표 감지
        converter = get_docling_converter()
        if converter:
            docling_result = converter.convert(file_path)
            layout_info = docling_result.document if hasattr(docling_result, 'document') else None
            
            if layout_info and hasattr(layout_info, 'tables'):
                docling_tables = getattr(layout_info, 'tables', [])
                print(f"  Docling 감지 표 개수: {len(docling_tables)}")
                
                for idx, tbl in enumerate(docling_tables):
                    if hasattr(tbl, 'export_to_markdown'):
                        md = tbl.export_to_markdown()
                        print(f"    표 {idx}: 마크다운 길이 {len(md) if md else 0}")
            else:
                print("  Docling: 표 감지 실패")
        
        # OCR 텍스트에서 표 감지
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        table_found_in_ocr = False
        for page_num, page in enumerate(pages, 1):
            from utils.indexing_utils import detect_table_in_text
            if detect_table_in_text(page.page_content):
                print(f"  OCR: 페이지 {page_num}에서 표 패턴 감지")
                table_found_in_ocr = True
        
        if not table_found_in_ocr:
            print("  OCR: 표 패턴 감지 실패")
            
    except Exception as e:
        print(f"  ❌ {method_name} 분석 중 오류: {e}")



# 실행
if __name__ == "__main__":
    print("Meta tensor 오류 재현을 위한 실제 운영 함수 테스트")
    print("=" * 60)
    
    # 먼저 Docling 컨버터 직접 테스트
    #asyncio.run(test_docling_converter_directly())
    
    # 하이브리드 문서 생성 진단 추가
    #asyncio.run(test_hybrid_document_creation())
    
    # 새로 추가됐는데 말안해줘서 내가 만들었다 이거 맞냐?
    #asyncio.run(test_parse_ocr_table_structure())
    #asyncio.run(test_actual_hybrid_workflow())
    #asyncio.run(test_pp_structure_output())
    # 그 다음 전체 워크플로우 테스트
    #asyncio.run(test_real_docling_workflow())

    #표감지
    asyncio.run(compare_production_vs_test_processing())