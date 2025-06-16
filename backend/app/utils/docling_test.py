# 하이브리드 방식: Docling 레이아웃 분석 + 선택적 OCR
import os
import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
from datetime import datetime

# Docling 임포트
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# 기존 PDF 텍스트 추출
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False

# LangChain Document
from langchain.docstore.document import Document

class HybridDocumentParser:
    """하이브리드 문서 파서: 레이아웃 분석 + 선택적 OCR"""
    
    def __init__(self):
        self.layout_converter = self._setup_layout_converter()
        self.ocr_converter = self._setup_ocr_converter()
        
    def _setup_layout_converter(self):
        """레이아웃 분석 전용 컨버터 (OCR 없음)"""
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = False  # OCR 비활성화
        pdf_options.do_table_structure = True  # 테이블 구조는 인식
        pdf_options.table_structure_options.do_cell_matching = True
        
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )
    
    def _setup_ocr_converter(self):
        """OCR 전용 컨버터 (이미지 영역만)"""
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = True  # OCR 활성화
        pdf_options.do_table_structure = False  # 테이블 구조 생략
        
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )
    
    async def parse_document_hybrid(self, file_path: str) -> List[Document]:
        """
        하이브리드 방식으로 문서 파싱
        1. Docling으로 레이아웃 분석
        2. 텍스트 영역: 기존 PDF 추출
        3. 이미지 영역: OCR 사용
        """
        try:
            print(f"🔍 하이브리드 파싱 시작: {Path(file_path).name}")
            start_time = time.time()
            
            # 1단계: 레이아웃 분석 (빠름)
            print("  📋 1단계: 레이아웃 분석 중...")
            layout_start = time.time()
            layout_result = await asyncio.to_thread(self.layout_converter.convert, file_path)
            layout_doc = layout_result.document
            layout_time = time.time() - layout_start
            print(f"    ✅ 레이아웃 분석 완료 ({layout_time:.2f}초)")
            
            # 2단계: 기존 방식으로 텍스트 추출 (빠름)
            print("  📝 2단계: PDF 텍스트 추출 중...")
            pdf_text_start = time.time()
            pdf_texts = await self._extract_pdf_text(file_path)
            pdf_text_time = time.time() - pdf_text_start
            print(f"    ✅ PDF 텍스트 추출 완료 ({pdf_text_time:.2f}초)")
            
            # 3단계: 영역별 문서 생성
            print("  🔧 3단계: 영역별 문서 생성 중...")
            documents = await self._create_hybrid_documents(
                layout_doc, pdf_texts, file_path
            )
            
            total_time = time.time() - start_time
            print(f"✅ 하이브리드 파싱 완료 ({total_time:.2f}초)")
            print(f"   📊 결과: {len(documents)}개 영역")
            
            return documents
            
        except Exception as e:
            print(f"❌ 하이브리드 파싱 실패: {e}")
            # Fallback: 기존 방식
            return await self._fallback_parsing(file_path)
    
    async def _extract_pdf_text(self, file_path: str) -> Dict[int, str]:
        """기존 방식으로 PDF 텍스트 추출 (페이지별)"""
        pdf_texts = {}
        
        if not PYPDF_AVAILABLE:
            print("⚠️ PyPDF 미설치, OCR 방식 사용")
            return pdf_texts
        
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    # 텍스트 정리
                    cleaned_text = re.sub(r'\s+', ' ', text).strip()
                    pdf_texts[page_num] = cleaned_text
        except Exception as e:
            print(f"⚠️ PDF 텍스트 추출 실패: {e}")
        
        return pdf_texts
    
    async def _create_hybrid_documents(
        self, 
        layout_doc, 
        pdf_texts: Dict[int, str], 
        file_path: str
    ) -> List[Document]:
        """레이아웃 정보와 PDF 텍스트를 결합하여 문서 생성"""
        documents = []
        file_name = Path(file_path).name
        
        # 텍스트 영역 처리
        text_docs = await self._process_text_regions(layout_doc, pdf_texts, file_path, file_name)
        documents.extend(text_docs)
        
        # 테이블 영역 처리
        table_docs = await self._process_table_regions(layout_doc, pdf_texts, file_path, file_name)
        documents.extend(table_docs)
        
        # 이미지 영역 처리 (필요 시 OCR)
        image_docs = await self._process_image_regions(layout_doc, file_path, file_name)
        documents.extend(image_docs)
        
        return documents
    
    async def _process_text_regions(
        self, 
        layout_doc, 
        pdf_texts: Dict[int, str], 
        file_path: str, 
        file_name: str
    ) -> List[Document]:
        """텍스트 영역 처리 - PDF 원본 텍스트 사용"""
        documents = []
        
        # PDF 텍스트가 있으면 우선 사용
        if pdf_texts:
            for page_num, page_text in pdf_texts.items():
                if not page_text.strip():
                    continue
                
                # 페이지 텍스트를 문단별로 분할
                paragraphs = self._split_into_paragraphs(page_text)
                
                for i, paragraph in enumerate(paragraphs):
                    if len(paragraph.strip()) < 10:  # 너무 짧은 텍스트 제외
                        continue
                    
                    # 텍스트 타입 분류
                    text_type = self._classify_text_type(paragraph, i)
                    
                    documents.append(Document(
                        page_content=paragraph,
                        metadata={
                            "source": file_path,
                            "file_name": file_name,
                            "page": page_num,
                            "element_type": "text",
                            "text_type": text_type,
                            "element_id": f"text_p{page_num}_{i}",
                            "extraction_method": "pdf_native",
                            "loaded_at": datetime.now().isoformat(),
                            "parser_engine": "Hybrid"
                        }
                    ))
        
        # PDF 텍스트가 없으면 Docling 텍스트 사용 (OCR 필요 시)
        else:
            print("⚠️ PDF 텍스트 없음, Docling 텍스트 사용")
            if hasattr(layout_doc, 'texts'):
                for i, text_item in enumerate(layout_doc.texts):
                    if not text_item.text or not text_item.text.strip():
                        continue
                    
                    documents.append(Document(
                        page_content=text_item.text.strip(),
                        metadata={
                            "source": file_path,
                            "file_name": file_name,
                            "page": getattr(text_item, 'page', 1),
                            "element_type": "text",
                            "text_type": "docling_extracted",
                            "element_id": f"docling_text_{i}",
                            "extraction_method": "docling_layout",
                            "loaded_at": datetime.now().isoformat(),
                            "parser_engine": "Hybrid"
                        }
                    ))
        
        return documents
    
    async def _process_table_regions(
        self, 
        layout_doc, 
        pdf_texts: Dict[int, str], 
        file_path: str, 
        file_name: str
    ) -> List[Document]:
        """테이블 영역 처리 - Docling 테이블 구조 사용"""
        documents = []
        
        if not hasattr(layout_doc, 'tables'):
            return documents
        
        for i, table in enumerate(layout_doc.tables):
            try:
                # 테이블 마크다운 변환
                table_content = ""
                caption = getattr(table, 'caption', '')
                
                if hasattr(table, 'export_to_markdown'):
                    try:
                        table_markdown = table.export_to_markdown()
                        table_content = table_markdown
                    except:
                        table_content = "[테이블 변환 실패]"
                
                # 캡션과 테이블 결합
                content = f"**테이블 {i+1}**"
                if caption:
                    content += f": {caption}"
                if table_content:
                    content += f"\n\n{table_content}"
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "file_name": file_name,
                        "page": getattr(table, 'page', 1),
                        "element_type": "table",
                        "element_id": f"table_{i}",
                        "table_caption": caption,
                        "extraction_method": "docling_structure",
                        "loaded_at": datetime.now().isoformat(),
                        "parser_engine": "Hybrid"
                    }
                ))
                
            except Exception as e:
                print(f"⚠️ 테이블 {i} 처리 오류: {e}")
                continue
        
        return documents
    
    async def _process_image_regions(
        self, 
        layout_doc, 
        file_path: str, 
        file_name: str
    ) -> List[Document]:
        """이미지 영역 처리 - 필요 시 OCR"""
        documents = []
        
        if not hasattr(layout_doc, 'pictures'):
            return documents
        
        for i, figure in enumerate(layout_doc.pictures):
            try:
                caption = getattr(figure, 'caption', '')
                image_type = getattr(figure, 'type', 'unknown')
                
                # 이미지 설명 생성
                content = f"**그림 {i+1}**"
                if caption:
                    content += f": {caption}"
                else:
                    content += f" ({image_type})"
                
                # 이미지에 텍스트가 있을 것 같으면 OCR 고려
                needs_ocr = self._should_apply_ocr(figure, caption)
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "file_name": file_name,
                        "page": getattr(figure, 'page', 1),
                        "element_type": "image",
                        "element_id": f"image_{i}",
                        "image_caption": caption,
                        "image_type": image_type,
                        "needs_ocr": needs_ocr,
                        "extraction_method": "docling_layout",
                        "loaded_at": datetime.now().isoformat(),
                        "parser_engine": "Hybrid"
                    }
                ))
                
            except Exception as e:
                print(f"⚠️ 이미지 {i} 처리 오류: {e}")
                continue
        
        return documents
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """텍스트를 문단별로 분할"""
        # 더블 개행으로 문단 분리
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 짧은 줄들은 합치기
        merged_paragraphs = []
        current_paragraph = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 짧은 줄이면 이전 문단에 합치기
            if len(para) < 100 and current_paragraph:
                current_paragraph += " " + para
            else:
                if current_paragraph:
                    merged_paragraphs.append(current_paragraph)
                current_paragraph = para
        
        # 마지막 문단 추가
        if current_paragraph:
            merged_paragraphs.append(current_paragraph)
        
        return merged_paragraphs
    
    def _classify_text_type(self, text: str, position: int) -> str:
        """텍스트 타입 분류"""
        text_lower = text.lower().strip()
        
        # 제목 패턴
        if position == 0 or (len(text) < 100 and not text.endswith('.')):
            return "heading"
        
        # 리스트 패턴
        if re.match(r'^(\d+\.|\-|\*|\•)', text.strip()):
            return "list"
        
        # 목차 패턴
        if '목차' in text_lower or re.search(r'\.\.\.\.\.\.|\.{3,}', text):
            return "toc"
        
        return "paragraph"
    
    def _should_apply_ocr(self, figure, caption: str) -> bool:
        """이미지에 OCR이 필요한지 판단"""
        # 차트, 다이어그램 등은 OCR 필요할 수 있음
        caption_lower = caption.lower() if caption else ""
        
        if any(word in caption_lower for word in ['차트', 'chart', '표', 'table', '그래프', 'graph']):
            return True
        
        # 캡션이 없거나 짧으면 OCR 필요할 수 있음
        if not caption or len(caption.strip()) < 10:
            return True
        
        return False
    
    async def _fallback_parsing(self, file_path: str) -> List[Document]:
        """Fallback: 기존 방식으로 파싱"""
        print("🔄 Fallback: 기존 방식 사용")
        
        # 기존 load_document_with_ocr 함수 사용
        # 여기서는 간단히 구현
        documents = []
        
        if PYPDF_AVAILABLE:
            try:
                reader = PdfReader(file_path)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append(Document(
                            page_content=text.strip(),
                            metadata={
                                "source": file_path,
                                "page": page_num,
                                "extraction_method": "fallback_pdf",
                                "parser_engine": "PyPDF"
                            }
                        ))
            except Exception as e:
                print(f"❌ Fallback 실패: {e}")
        
        return documents


# 성능 비교 테스트 함수
async def compare_parsing_methods(file_path: str):
    """파싱 방법별 성능 비교"""
    print(f"⚖️ 파싱 방법 비교: {Path(file_path).name}")
    print("=" * 60)
    
    hybrid_parser = HybridDocumentParser()
    
    # 1. 하이브리드 방식
    print("🔄 하이브리드 방식 테스트...")
    hybrid_start = time.time()
    hybrid_docs = await hybrid_parser.parse_document_hybrid(file_path)
    hybrid_time = time.time() - hybrid_start
    
    # 2. 기존 PDF 방식
    print("\n📄 기존 PDF 방식 테스트...")
    pdf_start = time.time()
    pdf_docs = await hybrid_parser._extract_pdf_text(file_path)
    pdf_time = time.time() - pdf_start
    
    # 결과 비교
    print(f"\n📊 === 성능 비교 결과 ===")
    print(f"하이브리드 방식:")
    print(f"  시간: {hybrid_time:.2f}초")
    print(f"  문서 수: {len(hybrid_docs)}개")
    
    total_hybrid_text = sum(len(doc.page_content) for doc in hybrid_docs)
    print(f"  총 텍스트: {total_hybrid_text:,}자")
    
    print(f"\n기존 PDF 방식:")
    print(f"  시간: {pdf_time:.2f}초")
    print(f"  페이지 수: {len(pdf_docs)}개")
    
    total_pdf_text = sum(len(text) for text in pdf_docs.values())
    print(f"  총 텍스트: {total_pdf_text:,}자")
    
    print(f"\n💡 성능 개선:")
    if hybrid_time > 0 and pdf_time > 0:
        if pdf_time < hybrid_time:
            speed_ratio = hybrid_time / pdf_time
            print(f"  PDF 방식이 {speed_ratio:.1f}배 빠름")
        else:
            speed_ratio = pdf_time / hybrid_time
            print(f"  하이브리드 방식이 {speed_ratio:.1f}배 빠름")
    
    # 품질 분석
    text_types = {}
    for doc in hybrid_docs:
        element_type = doc.metadata.get('element_type', 'unknown')
        text_types[element_type] = text_types.get(element_type, 0) + 1
    
    print(f"\n🔍 하이브리드 결과 구성:")
    for element_type, count in text_types.items():
        print(f"  {element_type}: {count}개")
    
    # 상세 결과 미리보기 추가
    print(f"\n📄 === 파싱 결과 상세 ===")
    
    # 하이브리드 결과 상세
    if hybrid_docs:
        print(f"📝 하이브리드 파싱 내용 (처음 3개):")
        for i, doc in enumerate(hybrid_docs[:3]):
            element_type = doc.metadata.get('element_type', 'unknown')
            page = doc.metadata.get('page', 1)
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            extraction_method = doc.metadata.get('extraction_method', 'unknown')
            
            print(f"  [{i+1}] {element_type} (페이지 {page}) - {extraction_method}")
            print(f"      내용: {content_preview}")
            print(f"      길이: {len(doc.page_content)}자")
            
            # 테이블인 경우 추가 정보
            if element_type == 'table':
                caption = doc.metadata.get('table_caption', '')
                if caption:
                    print(f"      캡션: {caption}")
            
            # 이미지인 경우 추가 정보
            elif element_type == 'image':
                image_caption = doc.metadata.get('image_caption', '')
                image_type = doc.metadata.get('image_type', '')
                needs_ocr = doc.metadata.get('needs_ocr', False)
                if image_caption:
                    print(f"      이미지 캡션: {image_caption}")
                if image_type:
                    print(f"      이미지 타입: {image_type}")
                print(f"      OCR 필요: {'예' if needs_ocr else '아니오'}")
            
            print()
    else:
        print("❌ 하이브리드 파싱 결과가 없습니다.")
    
    # PDF 원본 텍스트 미리보기
    if pdf_docs:
        print(f"📖 PDF 원본 텍스트 (처음 2페이지):")
        for page_num, text in list(pdf_docs.items())[:2]:
            text_preview = text[:200] + "..." if len(text) > 200 else text
            print(f"  페이지 {page_num}: {text_preview}")
            print(f"  길이: {len(text)}자")
            print()
    else:
        print("❌ PDF에서 추출된 텍스트가 없습니다. (스캔된 이미지 PDF일 가능성)")
    
    # 파일 특성 분석
    print(f"📊 === 파일 분석 ===")
    file_size = os.path.getsize(file_path) / 1024 / 1024
    print(f"파일 크기: {file_size:.2f}MB")
    
    if total_pdf_text == 0:
        print("📷 특징: 스캔된 이미지 PDF (텍스트 레이어 없음)")
        print("💡 권장: 이런 파일은 OCR이 필수입니다.")
    elif total_pdf_text > 0:
        print("📝 특징: 텍스트 레이어가 있는 PDF")
        print("💡 권장: 하이브리드 방식이 최적입니다.")
    
    # 개선 제안
    print(f"\n💡 === 개선 제안 ===")
    if total_pdf_text == 0:
        print("🔧 이 파일의 경우:")
        print("  - 전체 OCR 방식 사용 권장")
        print("  - 또는 이미지 영역만 선별적 OCR")
        print("  - 레이아웃 분석은 여전히 유용")
    else:
        print("🔧 이 파일의 경우:")
        print("  - 하이브리드 방식이 최적")
        print("  - PDF 텍스트 + Docling 레이아웃 분석")
        print("  - 빠른 속도 + 정확한 구조 분석")


# 사용 예제
async def main():
    """테스트 실행"""
    
    # 동적 경로 찾기 (이전에 작동했던 방식)
    current_dir = os.getcwd()
    print(f"현재 디렉토리: {current_dir}")

    # uploads 폴더 경로 찾기
    possible_paths = [
        "/home/test_code/test01/rag-chatbot/app/static/uploads",
        "../app/static/uploads",
        "../../app/static/uploads", 
        "../static/uploads",
        "static/uploads",
        "./uploads",
        "uploads"
    ]

    uploads_path = None
    for path in possible_paths:
        if os.path.exists(path):
            uploads_path = path
            break

    if not uploads_path:
        print("❌ uploads 폴더를 찾을 수 없습니다.")
        print("📁 확인된 경로들:")
        for path in possible_paths:
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {exists} {path}")
        return

    print(f"📁 uploads 폴더 발견: {uploads_path}")
    
    # PDF 파일 찾기 및 분류
    all_pdfs = []
    try:
        for file in os.listdir(uploads_path):
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(uploads_path, file)
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                all_pdfs.append((file_path, file, file_size))
    except Exception as e:
        print(f"❌ 파일 목록 읽기 실패: {e}")
        return
    
    if not all_pdfs:
        print("❌ PDF 파일이 없습니다.")
        return
    
    # 크기별로 정렬
    all_pdfs.sort(key=lambda x: x[2])
    
    print(f"\n📋 발견된 PDF 파일 ({len(all_pdfs)}개, 크기 순):")
    for i, (file_path, filename, size) in enumerate(all_pdfs[:10]):  # 처음 10개만
        file_type = "🖼️" if "이미지" in filename else "📄"
        print(f"  {i+1:2d}. {file_type} {filename[:50]}{'...' if len(filename) > 50 else ''} ({size:.1f}MB)")
    
    if len(all_pdfs) > 10:
        print(f"  ... 및 {len(all_pdfs) - 10}개 추가 파일")
    
    # 테스트할 파일들 선택
    test_candidates = []
    
    # 1. 가장 작은 일반 문서 (이미지 파일 제외)
    for file_path, filename, size in all_pdfs:
        if "이미지" not in filename and size < 5:  # 5MB 이하 일반 문서
            test_candidates.append((file_path, filename, size, "일반문서"))
            break
    
    # 2. 이미지 테스트 파일
    for file_path, filename, size in all_pdfs:
        if "이미지" in filename:
            test_candidates.append((file_path, filename, size, "이미지문서"))
            break
    
    # 3. 매뉴얼 파일
    for file_path, filename, size in all_pdfs:
        if any(word in filename.lower() for word in ["매뉴얼", "manual", "가이드", "guide"]) and size < 10:
            test_candidates.append((file_path, filename, size, "매뉴얼"))
            break
    
    # 후보가 없으면 가장 작은 3개 선택
    if not test_candidates:
        for file_path, filename, size in all_pdfs[:3]:
            test_candidates.append((file_path, filename, size, "소형파일"))
    
    print(f"\n🎯 테스트 후보 파일들:")
    for i, (file_path, filename, size, category) in enumerate(test_candidates):
        print(f"  {i+1}. [{category}] {filename} ({size:.1f}MB)")
    
    # 각 파일별로 테스트
    for i, (file_path, filename, size, category) in enumerate(test_candidates):
        print(f"\n{'='*80}")
        print(f"🧪 테스트 {i+1}/{len(test_candidates)}: [{category}] {filename}")
        print(f"{'='*80}")
        
        try:
            await compare_parsing_methods(file_path)
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # 다음 테스트 전 잠시 대기 (선택사항)
        if i < len(test_candidates) - 1:
            print(f"\n⏳ 다음 테스트까지 2초 대기...")
            await asyncio.sleep(2)
    
    print(f"\n🎉 모든 테스트 완료!")
    print(f"📋 결론:")
    print(f"  - 일반 PDF: 하이브리드 방식 권장 (빠름 + 정확)")
    print(f"  - 이미지 PDF: OCR 방식 필요 (텍스트 레이어 없음)")
    print(f"  - 매뉴얼 등: 하이브리드 방식 최적 (구조 복잡)")


if __name__ == "__main__":
    asyncio.run(main())