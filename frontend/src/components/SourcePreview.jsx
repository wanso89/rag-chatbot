import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import { 
  FiX, 
  FiCopy, 
  FiCheck, 
  FiMaximize2, 
  FiMinimize2, 
  FiFile, 
  FiFileText, 
  FiAlertTriangle
} from 'react-icons/fi';

/**
 * 출처 문서 미리보기 모달 컴포넌트
 */
const SourcePreview = ({ isOpen, onClose, source, content, image, isLoading, keywords }) => {
  if (!isOpen) return null;
  
  const [copySuccess, setCopySuccess] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // 오류 메시지인지 확인하는 함수
  const isErrorMessage = (text) => {
    if (!text) return false;
    return text.startsWith("소스를 불러오는 중 오류") || 
           text.startsWith("내용을 불러올 수 없습니다") ||
           text.includes("오류가 발생했습니다");
  };
  
  // 오류 메시지에서 복수 줄 처리
  const formatErrorMessage = (text) => {
    if (!text || !isErrorMessage(text)) return text;
    
    // 줄바꿈 기호가 포함된 경우 분리하여 렌더링
    const lines = text.split('\n');
    if (lines.length <= 1) return text;
    
    return (
      <>
        {lines.map((line, i) => (
          <React.Fragment key={i}>
            {line}
            {i < lines.length - 1 && <br />}
          </React.Fragment>
        ))}
      </>
    );
  };
  
  const handleCopyContent = () => {
    if (!content) return;
    
    navigator.clipboard.writeText(content);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
  };
  
  // HTML 태그가 포함된 콘텐츠인지 확인
  const hasHtmlTags = (text = '') => {
  if (typeof text !== 'string') return false;
  // <mark>, <span>, 또는 아무 HTML 태그나 탐지
  return /<\s*\/?\s*[a-zA-Z][^>]*?>/m.test(text);
};
   
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };
  
  // 렌더링 콘텐츠 결정
  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-full py-12">
          <div className="w-12 h-12 border-t-2 border-b-2 border-indigo-500 rounded-full animate-spin mb-4"></div>
          <p className="text-gray-400 text-sm font-medium">문서 내용을 불러오는 중...</p>
          <p className="text-gray-500 text-xs mt-2">잠시만 기다려주세요</p>
        </div>
      );
    }
    
    if (image) {
      return (
        <div className="flex justify-center items-center h-full">
          <img 
            src={image} 
            alt="문서 이미지" 
            className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-lg"
          />
        </div>
      );
    }
    
    if (!content) {
      return (
        <div className="flex flex-col items-center justify-center h-full py-12">
          <FiFileText className="text-gray-400 mb-4" size={48} />
          <p className="text-gray-500 font-medium">내용이 없습니다</p>
          <p className="text-gray-400 text-sm mt-2">해당 문서에 텍스트 내용이 없거나 추출할 수 없습니다.</p>
        </div>
      );
    }
    
    if (isErrorMessage(content)) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[200px] text-center py-12">
          <FiAlertTriangle className="text-yellow-500 mb-4" size={48} />
          <p className="text-gray-300 mb-3 font-medium text-lg">문서 내용을 불러올 수 없습니다</p>
          <div className="text-gray-500 text-sm max-w-md px-8">
            {formatErrorMessage(content.includes(":") ? content.split(":")[1].trim() : content)}
          </div>
          <button
            onClick={onClose}
            className="mt-6 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition-colors"
          >
            닫기
          </button>
        </div>
      );
    }
    
    // HTML 태그가 포함된 콘텐츠 처리
    if (hasHtmlTags(content)) {
      return (
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm, remarkMath]}
            skipHtml={false} 
            rehypePlugins={[
            rehypeRaw,         
            rehypeKatex,
            rehypeHighlight
          ]}
          >
            {content}
          </ReactMarkdown>
        </div>
      );
    }
    
    // 일반 마크다운 콘텐츠 - 키워드 하이라이트 강화
    const enhancedContent = keywords && keywords.length > 0
      ? highlightKeywordsInContent(content, keywords)
      : content;
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none px-1">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]} 
          rehypePlugins={[rehypeRaw, rehypeHighlight]}
          skipHtml={false} 
        >
          {enhancedContent}
        </ReactMarkdown>
      </div>
    );
  };
  
  // 키워드 하이라이트 함수
  const highlightKeywordsInContent = (text, keywords) => {
    if (!text || !keywords || keywords.length === 0) return text;
    
    // 정규식 특수문자 이스케이프
    const escapedKeywords = keywords.map(kw => String(kw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    
    // 문자열 내 모든 키워드를 강조 표시
    let highlightedText = text;
    escapedKeywords.forEach(keyword => {
      const regex = new RegExp(`(${keyword})`, 'gi');
      highlightedText = highlightedText.replace(regex, '**$1**');
    });
    
    return highlightedText;
  };
  
  return (
    <div 
      className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center animate-fade-in transition-all duration-300"
      style={{padding: isFullscreen ? 0 : '1rem'}}
      onClick={(e) => {
        // 배경 클릭 시 모달 닫기
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div 
        className={`bg-gray-900/90 backdrop-blur-md rounded-xl max-w-4xl w-full flex flex-col shadow-2xl border border-gray-700/30 overflow-hidden animate-slide-up ${
          isFullscreen ? 'h-full max-h-full rounded-none' : 'max-h-[90vh]'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 헤더 */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-700/50 bg-gray-800/50 backdrop-blur-sm">
          <div className="flex items-center space-x-3 overflow-hidden">
            <span className="p-2 bg-indigo-500/20 backdrop-blur-sm rounded-full flex-shrink-0">
              <FiFile className="text-indigo-400" size={16} />
            </span>
            <div className="overflow-hidden">
              <h3 className="font-medium text-gray-200 truncate max-w-xl">
                {source?.title || source?.display_name || (source?.path && source.path.split('/').pop().replace(/^[^_]*_/, ''))}
              </h3>
              <div className="flex items-center text-xs text-gray-400 mt-0.5">
                {source?.path && (
                  <span className="truncate max-w-xs opacity-70">{source.path.split('/').pop()}</span>
                )}
                {source?.page && (
                  <span className="text-xs text-gray-300 bg-gray-700/60 backdrop-blur-sm px-2 py-0.5 rounded-full ml-2 flex-shrink-0">
                    페이지 {source.page}
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            <button
              onClick={handleCopyContent}
              className="p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700/60 transition-all backdrop-blur-sm"
              title="내용 복사"
              disabled={!content || isLoading || isErrorMessage(content)}
            >
              {copySuccess ? <FiCheck className="text-green-400" /> : <FiCopy />}
            </button>
            <button
              onClick={toggleFullscreen}
              className="p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700/60 transition-all backdrop-blur-sm"
              title={isFullscreen ? "전체화면 종료" : "전체화면으로 보기"}
            >
              {isFullscreen ? <FiMinimize2 /> : <FiMaximize2 />}
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700/60 transition-all backdrop-blur-sm"
            >
              <FiX />
            </button>
          </div>
        </div>
        
        {/* 본문 */}
        <div className="flex-1 overflow-auto p-5 bg-gray-900/70 backdrop-blur-sm">
          {renderContent()}
        </div>
        
        {/* 키워드 표시 영역 */}
        {keywords && keywords.length > 0 && !isErrorMessage(content) && (
          <div className="px-5 py-3 border-t border-gray-700/50 bg-gray-800/30 backdrop-blur-sm">
            <p className="text-xs text-gray-400 mb-2 font-medium">관련 키워드:</p>
            <div className="flex flex-wrap gap-2">
              {keywords.map((keyword, idx) => (
                <span 
                  key={idx}
                  className="text-xs rounded-full px-3 py-1.5 bg-indigo-500/20 text-indigo-300 backdrop-blur-sm transition-all hover:bg-indigo-500/30"
                >
                  {keyword}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SourcePreview; 