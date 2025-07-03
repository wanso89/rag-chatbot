import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus as oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { useState, memo, useCallback, useMemo, useEffect, useRef } from "react";
import {
  FiEye,
  FiZoomIn,
  FiCopy,
  FiCheck,
  FiThumbsUp,
  FiThumbsDown,
  FiStar,
  FiLoader,
  FiExternalLink,
  FiUser,
  FiServer,
  FiX,
  FiClock,
  FiInfo,
  FiLink,
  FiCornerDownRight,
  FiMessageSquare,
  FiCircle,
  FiList,
  FiChevronDown,
  FiChevronRight,
  FiHash,
  FiBookmark,
  FiMessageCircle,
  FiMoreHorizontal,
  FiImage,
  FiMaximize2,
  FiMinimize2,
  FiSend,
  FiSmile,
  FiAlertCircle,
  FiFrown,
  FiHelpCircle,
  FiFile,
  FiSearch,
  FiAlertTriangle,
  FiFileText
} from "react-icons/fi";
import rehypeRaw from 'rehype-raw';  // HTML 태그 처리를 위한 플러그인 추가
import parse from 'html-react-parser'; // HTML 파싱을 위한 라이브러리 추가
import DOMPurify from 'dompurify';

// 키워드 하이라이트 애니메이션을 위한 키프레임 스타일 정의
const keyframesStyle = `
@keyframes highlightPulse {
  0% { background-color: rgba(253, 224, 71, 0.75); box-shadow: 0 0 5px rgba(250, 204, 21, 0.6); transform: scale(1); }
  50% { background-color: rgba(250, 204, 21, 0.9); box-shadow: 0 0 8px rgba(250, 204, 21, 0.8); transform: scale(1.03); }
  100% { background-color: rgba(253, 224, 71, 0.75); box-shadow: 0 0 5px rgba(250, 204, 21, 0.6); transform: scale(1); }
}

.animate-highlight-pulse {
  animation: highlightPulse 2s ease-in-out infinite;
}

.highlight-keyword {
  background-color: rgba(253, 224, 71, 0.85);
  color: rgba(0, 0, 0, 0.95);
  text-shadow: 0 0 0 rgba(0, 0, 0, 0.1);
  border-radius: 4px;
  padding: 2px 5px;
  margin: 0 -1px;
  font-weight: 700;
  position: relative;
  box-shadow: 0 0 5px rgba(250, 204, 21, 0.8);
  display: inline-block;
  border-bottom: 1px solid rgba(234, 179, 8, 0.5);
  z-index: 1;
}

/* 다크 모드 스타일 */
@media (prefers-color-scheme: dark) {
  .highlight-keyword {
    background-color: rgba(253, 224, 71, 0.85);
    color: rgba(0, 0, 0, 0.9);
    box-shadow: 0 0 6px rgba(250, 204, 21, 0.8);
  }
}
`;

// 스타일을 문서에 주입하는 함수
const injectStyleOnce = (id, css) => {
  if (!document.getElementById(id)) {
    const head = document.head || document.getElementsByTagName('head')[0];
    const style = document.createElement('style');
    style.id = id;
    style.appendChild(document.createTextNode(css));
    head.appendChild(style);
  }
};

const KOREAN_STOPWORDS = new Set([
  "이",
  "가",
  "을",
  "를",
  "은",
  "는",
  "에",
  "에서",
  "로",
  "으로",
  "과",
  "와",
  "도",
  "의",
  "들",
  "좀",
  "등",
  "및",
  "그",
  "저",
  "것",
  "수",
  "알려줘",
  "궁금해",
  "대한",
  "대해",
  "내용",
  "무엇인가요",
  "뭔가요",
  "뭐야",
  "설명해줘",
  "알고싶어",
]);


// 타이핑 효과 애니메이션 컴포넌트
const TypeWriter = ({ text, speed = 2, onComplete }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  
  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
        
        // 주기적으로 스크롤 이벤트 발생 (타이핑 중에도 스크롤 유지)
        if (currentIndex % 50 === 0) {
          window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
        }
      }, speed);
      
      return () => clearTimeout(timeout);
    } else if (!isComplete) {
      setIsComplete(true);
      
      // 타이핑 완료 시 스크롤 이벤트 발생
      window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
      
      if (onComplete) onComplete();
    }
  }, [text, currentIndex, speed, isComplete, onComplete]);
  
  return (
    <div>
      {displayText}
      {currentIndex < text.length && (
        <span className="inline-block w-1.5 h-4 ml-0.5 bg-indigo-400 animate-pulse"></span>
      )}
    </div>
  );
};

// 내용에서 제목 추출 함수
const extractHeadings = (markdownText) => {
  const headings = [];
  const lines = markdownText.split('\n');
  
  lines.forEach((line) => {
    // # 스타일 헤딩 매칭
    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      
      headings.push({
        level,
        text,
        id: text.toLowerCase().replace(/[^\w\s가-힣]/g, '').replace(/\s+/g, '-')
      });
    }
  });
  
  return headings;
};

// 목차 컴포넌트
const TableOfContents = ({ headings, onClickHeading }) => {
  const [isOpen, setIsOpen] = useState(true);
  
  if (!headings || headings.length === 0) return null;
  
  return (
    <div className="my-3 border border-gray-700/40 rounded-lg overflow-hidden">
      <div 
        className="bg-gray-800/80 px-3 py-2 flex items-center justify-between cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center text-sm font-medium text-gray-200">
          <FiList className="mr-2 text-indigo-400" size={16} />
          목차
        </div>
        <button className="text-gray-400 hover:text-white transition-colors">
          {isOpen ? <FiChevronDown size={16} /> : <FiChevronRight size={16} />}
        </button>
      </div>
      
      {isOpen && (
        <div className="bg-gray-850/50 p-3">
          <ul className="space-y-1 text-sm">
            {headings.map((heading, idx) => (
              <li key={idx} className="leading-snug">
                <button
                  onClick={() => onClickHeading(heading.id)}
                  className={`hover:text-indigo-400 transition-colors flex items-start ${
                    heading.level === 1 ? "font-medium text-gray-200" :
                    heading.level === 2 ? "pl-4 text-gray-300" : 
                    "pl-8 text-gray-400"
                  }`}
                >
                  <FiHash size={12} className="mr-1 mt-1 flex-shrink-0" />
                  <span className="truncate">{heading.text}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

// 프로필 아바타 컴포넌트
const ProfileAvatar = ({ role, isGrouped }) => {
  const isAssistant = role === 'assistant';
  
  return (
    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
      isGrouped ? 'opacity-0' : ''
    } transition-opacity duration-200 ${
      isAssistant 
        ? 'bg-gradient-to-br from-indigo-500 to-purple-600' 
        : 'bg-gradient-to-br from-blue-500 to-cyan-500'
    }`}>
      {isAssistant ? (
        <FiMessageSquare className="text-white" size={16} />
      ) : (
        <FiUser className="text-white" size={16} />
      )}
    </div>
  );
};

// 이미지 미리보기 컴포넌트
const ImagePreview = ({ src, alt }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className="relative group">
      <img 
        src={src} 
        alt={alt || '이미지'} 
        className={`rounded-lg shadow-md transition-all duration-300 ${
          isExpanded ? 'max-w-none' : 'max-w-md'
        }`}
      />
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="absolute top-2 right-2 p-2 bg-gray-800/80 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {isExpanded ? (
          <FiMinimize2 className="text-white" size={16} />
        ) : (
          <FiMaximize2 className="text-white" size={16} />
        )}
      </button>
    </div>
  );
};

// 피드백 모달 컴포넌트
const FeedbackModal = ({ isOpen, onClose, messageContent, onSubmit, feedbackType }) => {
  const [reasons, setReasons] = useState([]);
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [rating, setRating] = useState(0);
  
  const modalRef = useRef(null);
  
  // 좋아요/싫어요에 따른 피드백 이유 옵션
  const reasonOptions = feedbackType === 'up'
    ? [
        '정확한 정보를 제공함',
        '이해하기 쉽게 설명됨',
        '필요한 정보를 모두 담고 있음',
        '잘 정리되어 있음',
        '유용한 예시나 코드가 포함됨'
      ]
    : [
        '부정확한 정보가 포함됨',
        '이해하기 어려움',
        '너무 장황함',
        '질문에 제대로 답변하지 않음',
        '필요한 정보가 누락됨'
      ];
  
  // 모달 외부 클릭 시 닫기
  useEffect(() => {
    if (!isOpen) return;
    
    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        onClose();
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);
  
  // 모달 열릴 때마다 상태 초기화
  useEffect(() => {
    if (isOpen) {
      setReasons([]);
      setComment('');
      setRating(0);
      setIsSubmitting(false);
    }
  }, [isOpen, feedbackType]);
  
  // 이유 토글 핸들러
  const toggleReason = (reason) => {
    setReasons(prev => 
      prev.includes(reason)
        ? prev.filter(r => r !== reason)
        : [...prev, reason]
    );
  };
  
  // 제출 핸들러
  const handleSubmit = async () => {
    if (isSubmitting) return;
    
    setIsSubmitting(true);
    try {
      // 피드백 데이터 준비
      const feedbackData = {
        feedbackType,
        reasons,
        comment: comment.trim(),
        rating,
        content: messageContent
      };
      
      await onSubmit(feedbackData);
      onClose();
    } catch (error) {
      console.error('피드백 제출 중 오류 발생:', error);
    } finally {
      setIsSubmitting(false);
    }
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
      <div 
        ref={modalRef}
        className="bg-gray-800 rounded-xl max-w-md w-full shadow-2xl border border-gray-700/50 overflow-hidden animate-slide-up"
      >
        <div className="p-4 border-b border-gray-700/50 flex items-center">
          <div className={`mr-2 p-2 rounded-full ${
            feedbackType === 'up' 
              ? 'bg-green-900/30 text-green-400' 
              : 'bg-red-900/30 text-red-400'
          }`}>
            {feedbackType === 'up' 
              ? <FiThumbsUp size={18} /> 
              : <FiThumbsDown size={18} />
            }
          </div>
          <h3 className="text-lg font-medium text-gray-100">
            {feedbackType === 'up' ? '긍정적인 피드백' : '부정적인 피드백'}
          </h3>
          <button
            onClick={onClose}
            className="ml-auto p-2 text-gray-400 hover:text-gray-200 rounded-full hover:bg-gray-700/50"
          >
            <FiX size={20} />
          </button>
        </div>
        
        <div className="p-5 max-h-[70vh] overflow-y-auto">
          {/* 피드백 이유 선택 */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              {feedbackType === 'up' ? '좋았던 점' : '개선이 필요한 점'}
            </label>
            <div className="space-y-2">
              {reasonOptions.map((reason, idx) => (
                <button
                  key={idx}
                  onClick={() => toggleReason(reason)}
                  className={`w-full text-left px-3 py-2 rounded-lg border text-sm transition-colors flex items-center ${
                    reasons.includes(reason)
                      ? feedbackType === 'up'
                        ? 'bg-green-900/20 border-green-500/30 text-green-300'
                        : 'bg-red-900/20 border-red-500/30 text-red-300'
                      : 'bg-gray-700/50 border-gray-600 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <div className={`w-4 h-4 mr-2 rounded-full border flex-shrink-0 ${
                    reasons.includes(reason)
                      ? feedbackType === 'up'
                        ? 'bg-green-500 border-green-500'
                        : 'bg-red-500 border-red-500'
                      : 'border-gray-500'
                  }`}>
                    {reasons.includes(reason) && (
                      <FiCheck 
                        size={12} 
                        className="text-gray-900 m-auto" 
                      />
                    )}
                  </div>
                  <span>{reason}</span>
                </button>
              ))}
            </div>
          </div>
          
          {/* 별점 */}
          {feedbackType === 'up' && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                평가
              </label>
              <div className="flex items-center space-x-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={() => setRating(star)}
                    className={`p-1 transition-all ${
                      star <= rating
                        ? 'text-yellow-400 scale-110'
                        : 'text-gray-500 hover:text-gray-400'
                    }`}
                  >
                    <FiStar 
                      size={24} 
                      className={star <= rating ? 'fill-current' : ''}
                    />
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {/* 추가 코멘트 */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              추가 코멘트 (선택사항)
            </label>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="의견이 있으시면 자유롭게 작성해주세요..."
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg resize-none h-24 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-200 placeholder-gray-500"
            />
          </div>
        </div>
        
        <div className="p-4 border-t border-gray-700/50 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
          >
            취소
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting}
            className={`px-4 py-2 rounded-lg flex items-center ${
              feedbackType === 'up'
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-red-600 hover:bg-red-700 text-white'
            } transition-colors`}
          >
            {isSubmitting ? (
              <>
                <FiLoader className="animate-spin mr-2" size={16} />
                전송 중...
              </>
            ) : (
              <>
                <FiSend className="mr-2" size={16} />
                피드백 전송
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};



// 피드백 결과 토스트 컴포넌트
const FeedbackToast = ({ isVisible, message, type, onClose }) => {
  useEffect(() => {
    if (isVisible) {
      const timer = setTimeout(() => {
        onClose();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [isVisible, onClose]);
  
  if (!isVisible) return null;
  
  return (
    <div className="fixed bottom-4 right-4 z-50 animate-fade-in">
      <div className={`p-3 rounded-lg shadow-lg flex items-center ${
        type === 'success'
          ? 'bg-green-600 text-white'
          : 'bg-red-600 text-white'
      }`}>
        {type === 'success' ? (
          <FiCheck className="mr-2" size={18} />
        ) : (
          <FiAlertCircle className="mr-2" size={18} />
        )}
        <span>{message}</span>
        <button
          onClick={onClose}
          className="ml-2 p-1 rounded-full hover:bg-black/10"
        >
          <FiX size={16} />
        </button>
      </div>
    </div>
  );
};

// UUID 제거 함수 정의
const getCleanFileName = (filePath) => {
  if (!filePath) return '알 수 없는 출처';
  
  const fileName = filePath.split('/').pop();
  // UUID 패턴 제거 (UUID_파일명.확장자 형식)
  const uuidPattern = /^([a-f0-9]{8,}-?[a-f0-9-]*_|[a-f0-9]{8,}_)/i;
  return fileName.replace(uuidPattern, '');
};

// 출처 미리보기 모달 컴포넌트
const SourcePreviewModal = ({ isOpen, onClose, source, content, image, isLoading, keywords }) => {
  if (!isOpen) return null;
  
  const [copySuccess, setCopySuccess] = useState(false);
  
  // 오류 메시지인지 확인하는 함수
  const isErrorMessage = (text) => {
    if (!text || typeof text !== 'string') return false;
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
  const hasHtmlTags = (text) => {
    return text && typeof text === 'string' && (
      text.includes('<span class="highlight-strong">') || 
      text.includes('<span class="highlight-medium">')
    );
  };
  
  // 렌더링 콘텐츠 결정
  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-full">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-400 mb-3"></div>
          <p className="text-gray-400 text-sm">문서 내용을 불러오는 중...</p>
        </div>
      );
    }
    
    if (image) {
      return (
        <div className="flex justify-center">
          <img 
            src={image} 
            alt="문서 이미지" 
            className="max-w-full max-h-[70vh] object-contain"
          />
        </div>
      );
    }
    
    if (!content) {
      return (
        <div className="flex flex-col items-center justify-center h-full py-8">
          <FiFileText className="text-gray-600 mb-4" size={32} />
          <p className="text-gray-500">내용이 없습니다.</p>
        </div>
      );
    }
    
    if (isErrorMessage(content)) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[200px] text-center py-8">
          <FiAlertTriangle className="text-yellow-500 mb-4" size={32} />
          <p className="text-gray-300 mb-2 font-medium">문서 내용을 불러올 수 없습니다</p>
          <div className="text-gray-500 text-sm max-w-md">
            {formatErrorMessage(content.includes(":") ? content.split(":")[1].trim() : content)}
          </div>
        </div>
      );
    }
    
    // HTML 태그가 포함된 콘텐츠 처리
    if (hasHtmlTags(content)) {
      // [object Object] 제거
      const cleanContent = content?.replace(/\[object Object\]/g, '') || content;
      
      return (
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]} 
            rehypePlugins={[rehypeHighlight, rehypeRaw]}
          >
            {cleanContent}
          </ReactMarkdown>
        </div>
      );
}
    
    // 일반 마크다운 콘텐츠
    const cleanContent = content?.replace(/\[object Object\]/g, '') || content;
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]} 
          rehypePlugins={[rehypeHighlight]}
        >
          {cleanContent}
        </ReactMarkdown>
      </div>
    );
  };
  
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in transition-all duration-300">
      <div className="bg-gray-900/90 backdrop-blur-md rounded-xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl border border-gray-700/30 overflow-hidden animate-slide-up">
        {/* 헤더 */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-700/50 bg-gray-800/50 backdrop-blur-sm">
          <div className="flex items-center space-x-3">
            <span className="p-1.5 bg-indigo-500/20 backdrop-blur-sm rounded-full">
              <FiFile className="text-indigo-400" size={16} />
            </span>
            <h3 className="font-medium text-gray-200 truncate">
              {source?.title || source?.display_name || getCleanFileName(source?.path)}
            </h3>
            {source?.page && (
              <span className="text-xs text-gray-300 bg-gray-700/60 backdrop-blur-sm px-2.5 py-1 rounded-full ml-1">
                페이지 {source.page}
              </span>
            )}
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
                  className="text-xs rounded-full px-2.5 py-1 bg-indigo-500/20 text-indigo-300 backdrop-blur-sm transition-all hover:bg-indigo-500/30"
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

// 표 컴포넌트
const TableDisplay = ({ tableData, caption }) => {
  if (!tableData || !Array.isArray(tableData)) return null;
  
  // 표 데이터를 파싱하여 헤더와 행 분리
  const parseTableData = (data) => {
    const rows = data.map(item => item.text || item.content || '').filter(Boolean);
    if (rows.length === 0) return { headers: [], rows: [] };
    
    // 첫 번째 행을 헤더로 가정하고 파싱
    const headers = rows[0].split('|').map(h => h.trim()).filter(Boolean);
    const dataRows = rows.slice(1).map(row => 
      row.split('|').map(cell => cell.trim()).filter(Boolean)
    ).filter(row => row.length > 0);
    
    return { headers, rows: dataRows };
  };
  
  const { headers, rows } = parseTableData(tableData);
  
  if (headers.length === 0 && rows.length === 0) return null;
  
  return (
    <div className="my-4 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      {caption && (
        <div className="bg-gray-50 dark:bg-gray-800 px-3 py-2 border-b border-gray-200 dark:border-gray-700">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">{caption}</p>
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          {headers.length > 0 && (
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                {headers.map((header, idx) => (
                  <th key={idx} className="px-3 py-2 text-left font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {rows.map((row, rowIdx) => (
              <tr key={rowIdx} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                {row.map((cell, cellIdx) => (
                  <td key={cellIdx} className="px-3 py-2 text-gray-800 dark:text-gray-200">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// 이미지 모달 컴포넌트
const ImageModal = ({ isOpen, onClose, src, alt, caption }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  
  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      setError(false);
    }
  }, [isOpen, src]);
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[100] flex items-center justify-center p-4" onClick={onClose}>
      <div className="relative max-w-7xl max-h-[90vh] w-full h-full flex items-center justify-center">
        {/* 닫기 버튼 */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 p-2 bg-black/50 hover:bg-black/70 rounded-full text-white transition-colors"
        >
          <FiX size={24} />
        </button>
        
        {/* 이미지 컨테이너 */}
        <div className="relative max-w-full max-h-full flex flex-col">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 rounded-lg">
              <div className="flex flex-col items-center text-white">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mb-2"></div>
                <p className="text-sm">이미지 로딩 중...</p>
              </div>
            </div>
          )}
          
          {error ? (
            <div className="flex items-center justify-center bg-gray-800 rounded-lg p-8 text-white">
              <div className="text-center">
                <FiAlertCircle size={48} className="mx-auto mb-4 text-red-400" />
                <p className="text-lg font-medium mb-2">이미지를 불러올 수 없습니다</p>
                <p className="text-sm text-gray-400">이미지 파일이 존재하지 않거나 접근할 수 없습니다.</p>
              </div>
            </div>
          ) : (
            <img
              src={src}
              alt={alt}
              className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-2xl"
              onLoad={() => setLoading(false)}
              onError={() => {
                setLoading(false);
                setError(true);
              }}
            />
          )}
          
          {/* 캡션 */}
          {caption && !loading && !error && (
            <div className="mt-3 bg-black/60 backdrop-blur-sm rounded-lg p-3 text-white text-center">
              <p className="text-sm">{caption}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

  // 이미지 표시 컴포넌트
  const ImageDisplay = ({ images, captions }) => {
    const [selectedImage, setSelectedImage] = useState(null);
    
    console.log("ImageDisplay received images:", JSON.stringify(images, null, 2));
    console.log("ImageDisplay received captions:", JSON.stringify(captions, null, 2));

  if (!images || !Array.isArray(images) || images.length === 0) return null;
  
  // 문서명 기반 캡션 생성 함수
  const generateDocumentCaption = (imageData, idx) => {
    const sourceInfo = imageData.source_info;
    if (!sourceInfo) return `이미지 ${idx + 1}`;
    
    const documentName = sourceInfo.display_name || '문서';
    const pageInfo = sourceInfo.page ? ` 페이지 ${sourceInfo.page}` : '';
    
    return `${documentName}${pageInfo}에서 발췌`;
  };
  
  return (
    <>
      <div className="my-4 space-y-3">
        {images.map((imagePath, idx) => {
          const captionData = captions && captions[idx] ? captions[idx] : null;
          const documentCaption = generateDocumentCaption({ source_info: captionData?.source_info }, idx);
          // 백엔드에서 'document_images/' 접두사를 포함하여 경로를 보내는 문제에 대한 임시 해결책
          const cleanImagePath = imagePath.replace(/^document_images\//, '');
          const fullImagePath = imagePath.startsWith('http') ? imagePath : `/api/image-viewer/${cleanImagePath}`;
          
          return (
            <div key={idx} className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden hover:shadow-lg transition-shadow">
              <div className="bg-gray-50 dark:bg-gray-800 px-3 py-2 border-b border-gray-200 dark:border-gray-700">
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">{documentCaption}</p>
              </div>
              <div className="p-3">
                {/* 썸네일 이미지 */}
                <div className="relative group cursor-pointer" onClick={() => setSelectedImage({ src: fullImagePath, alt: documentCaption, caption: documentCaption })}>
                  <img 
                    src={fullImagePath} 
                    alt={documentCaption}
                    className="w-full h-48 object-contain bg-gray-100 dark:bg-gray-800 rounded-md transition-all duration-300 group-hover:scale-105"
                    loading="lazy"
                    onError={(e) => {
                       // 이미지 로딩 실패 로그 제거
                      e.target.style.display = 'none';
                      if (e.target.nextSibling) {
                        e.target.nextSibling.style.display = 'flex';
                      }
                    }}
                  />
                  {/* 이미지 로딩 실패 시 표시할 대체 요소 */}
                  <div className="hidden w-full h-48 bg-gray-100 dark:bg-gray-800 rounded-md items-center justify-center text-gray-500 dark:text-gray-400">
                    <div className="text-center">
                      <FiImage size={48} className="mx-auto mb-2" />
                      <p className="text-sm">이미지를 불러올 수 없습니다</p>
                      <p className="text-xs mt-1 text-gray-400">경로: {fullImagePath}</p>
                    </div>
                  </div>
                  {/* 호버 오버레이 */}
                  <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity rounded-md flex items-center justify-center">
                    <div className="bg-white/90 dark:bg-gray-800/90 rounded-full p-3">
                      <FiZoomIn size={24} className="text-gray-800 dark:text-gray-200" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {/* 이미지 모달 */}
      <ImageModal
        isOpen={!!selectedImage}
        onClose={() => setSelectedImage(null)}
        src={selectedImage?.src}
        alt={selectedImage?.alt}
        caption={selectedImage?.caption}
      />
    </>
  );
};

// 출처 컴포넌트 개선
const SourceItem = ({ source, onClick, isFiltered = false }) => {
  // source 객체가 없는 경우 기본값 설정
  if (!source) {
    return (
      <div className="flex items-center p-2 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300">
        <div className="w-8 h-8 rounded-lg bg-red-200 dark:bg-red-800/50 flex items-center justify-center text-red-600 dark:text-red-400 mr-3 flex-shrink-0">
          <FiAlertCircle size={16} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">
            출처 정보 없음
          </p>
        </div>
      </div>
    );
  }
  
  // source.path 또는 source.source 중 사용 가능한 것을 선택
  const sourcePath = source.path || source.source;
  
  // 소스 경로가 없는 경우
  if (!sourcePath) {
    return (
      <div className="flex items-center p-2 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300">
        <div className="w-8 h-8 rounded-lg bg-red-200 dark:bg-red-800/50 flex items-center justify-center text-red-600 dark:text-red-400 mr-3 flex-shrink-0">
          <FiAlertCircle size={16} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">
            출처 경로 정보 없음
          </p>
        </div>
      </div>
    );
  }
  
  // 파일명에서 UUID 접두사 제거 - 개선된 로직
  const getDisplayName = (path) => {
    if (!path) return '알 수 없는 출처';
    const fileName = path.split('/').pop();
    // UUID 패턴 제거 (UUID_파일명.확장자 형식)
    const uuidPattern = /^([a-f0-9]{8,}-?[a-f0-9-]*_|[a-f0-9]{8,}_)/i;
    return fileName.replace(uuidPattern, '');
  };
  
  const displayName = getDisplayName(sourcePath);
  
  // 표나 이미지 타입에 따라 아이콘 변경
  const getSourceIcon = () => {
    if (source.element_type === 'table_row') {
      return <FiList size={16} />;
    } else if (source.has_images) {
      return <FiImage size={16} />;
    } else {
      return <FiFileText size={16} />;
    }
  };
  
  // 추가 정보 표시
  const getAdditionalInfo = () => {
    const info = [];
    if (source.page) {
      info.push(`페이지 ${source.page}`);
    }
    if (source.element_type === 'table_row') {
      info.push('표 데이터');
    }
    if (source.has_images) {
      info.push(`이미지 ${source.images?.length || 1}개`);
    }
    return info.join(' • ');
  };
  
  return (
    <div 
      className={`flex items-center p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors ${
        isFiltered ? 'opacity-40' : ''
      }`}
      onClick={() => onClick(source)}
    >
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center mr-3 flex-shrink-0 ${
        source.element_type === 'table_row' 
          ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
          : source.has_images
          ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
          : 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400'
      }`}>
        {getSourceIcon()}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">
          {displayName}
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {getAdditionalInfo()}
        </p>
      </div>
      <div className="ml-2 text-gray-400 hover:text-indigo-500 dark:hover:text-indigo-400 transition-colors">
        <FiExternalLink size={16} />
      </div>
    </div>
  );
};

function ChatMessage({ message, searchTerm = "", isSearchMode, prevMessage, nextMessage, onAskFollowUp }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [star, setStar] = useState(0);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [currentFeedbackType, setCurrentFeedbackType] = useState(null);
  const [hasInteractedWithButtons, setHasInteractedWithButtons] = useState(false);
  const [toast, setToast] = useState({ visible: false, message: '', type: 'info' });
  
  // 출처 표시 관련 상태
  const [sourcesVisible, setSourcesVisible] = useState(false);
  const [showSourcePreview, setShowSourcePreview] = useState(false);
  const [previewSource, setPreviewSource] = useState(null);
  const [previewContent, setPreviewContent] = useState("");
  const [previewImage, setPreviewImage] = useState(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [sourceKeywords, setSourceKeywords] = useState([]);
  const [highlightKeywords, setHighlightKeywords] = useState([]);
  const [sourceFilterText, setSourceFilterText] = useState("");
  const [showTypeWriter, setShowTypeWriter] = useState(false);
  const [isTypingComplete, setIsTypingComplete] = useState(true);
  const [showImagePreview, setShowImagePreview] = useState(false);
  const [isGrouped, setIsGrouped] = useState(false);
  const [threadStartLine, setThreadStartLine] = useState(false);
  const [showThreadLine, setShowThreadLine] = useState(false);
  const [threadEndLine, setThreadEndLine] = useState(false);
  const [showTableOfContents, setShowTableOfContents] = useState(false);
  
  // 참조 추가
  const messageContainerRef = useRef(null);
  const contentRef = useRef(null);
  
  // 메시지 ID 생성
  const messageId = useMemo(() => `message-${message.timestamp || Date.now()}-${Math.random().toString(36).substr(2, 9)}`, [message.timestamp]);
  
  // 메시지 내용 준비 - 모든 관련 함수보다 먼저 실행
  // 봇 메시지인 경우 bot_response 사용, 사용자 메시지는 content 사용
  const messageContent = useMemo(() => {
    const rawContent = message.role === 'user' ? message.content : (message.bot_response || message.content || "");
    if (!rawContent) return "";
    
    // 마크다운 노드를 문자열로 변환하는 함수
    const mdNodeToString = (node) => {
      if (node == null) return '';
      if (typeof node === 'string') return node;
      if (Array.isArray(node)) return node.map(mdNodeToString).join('\n');
      
      if (typeof node === 'object' && node !== null) {
        switch (node.type) {
          case 'heading':
            return `${'#'.repeat(node.depth || 1)} ${mdNodeToString(node.children || node.value || '')}`;
          case 'listItem':
            return `- ${mdNodeToString(node.children || node.value || '')}`;
          case 'code':
            return `\`\`\`${node.lang || ''}\n${node.value || ''}\n\`\`\``;
          case 'paragraph':
            return mdNodeToString(node.children || node.value || '');
          case 'text':
            return node.value || '';
          default:
            return mdNodeToString(node.children || node.value || '');
        }
      }
      return String(node);
    };
    
    // 데이터 평탄화: 배열이나 객체를 문자열로 변환
    const content = mdNodeToString(rawContent);
    
    // 디버그 로그 추가
    console.log("messageContent type:", typeof content);
    console.log("messageContent isArray:", Array.isArray(content));
    console.log("messageContent includes [object Object]:", content.includes('[object Object]'));
    console.log("messageContent:", content);
    
    return content;
  }, [message.role, message.content, message.bot_response]);

  // 추천 질문 관련 상태
  const suggestedQuestions = useMemo(() => {
    return message.suggestedQuestions || [];
  }, [message.suggestedQuestions]);
  
  // 키워드 하이라이트 스타일 주입
  useEffect(() => {
    injectStyleOnce('highlight-animation-style', keyframesStyle);
  }, []);
  
  // 같은 화자의 연속 메시지인지 확인 - 위치 이동
  const isPrevSameSender = useMemo(() => {
    return prevMessage && prevMessage.role === message.role;
  }, [prevMessage, message.role]);
  
  const isNextSameSender = useMemo(() => {
    return nextMessage && nextMessage.role === message.role;
  }, [nextMessage, message.role]);
  
  // 스레드 라인 관련 상태 업데이트
  useEffect(() => {
    // 현재 메시지가 어시스턴트인 경우만 스레드 라인 표시
    if (!isUser) {
      // 앞 메시지가 다른 화자(사용자)면 스레드 시작
      setThreadStartLine(!isPrevSameSender);
      
      // 뒤 메시지가 같은 화자(어시스턴트)면 스레드 라인 표시
      setShowThreadLine(isNextSameSender);
      
      // 뒤 메시지가 다른 화자(사용자)면 스레드 종료
      setThreadEndLine(!isNextSameSender);
    } else {
      // 사용자 메시지는 스레드 라인 표시 안함
      setThreadStartLine(false);
      setShowThreadLine(false);
      setThreadEndLine(false);
    }
  }, [isUser, isPrevSameSender, isNextSameSender]);

  // isPrevSameSender에 따라 isGrouped 상태 업데이트
  useEffect(() => {
    setIsGrouped(isPrevSameSender);
  }, [isPrevSameSender]);
  
  // 제목 추출 및 목차 생성
  const headings = useMemo(() => {
    if (isUser || !message.content) return [];
    return extractHeadings(message.content);
  }, [isUser, message.content]);
  
  // 목차 표시 여부 업데이트
  useEffect(() => {
    setShowTableOfContents(!isUser && headings.length >= 2);
  }, [isUser, headings.length]);
  
  // 목차 클릭 시 해당 제목으로 스크롤
  const scrollToHeading = (headingId) => {
    if (!contentRef.current) return;
    
    const element = contentRef.current.querySelector(`#${headingId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // 강조 효과 추가
      element.classList.add('bg-indigo-500/20');
      setTimeout(() => {
        element.classList.remove('bg-indigo-500/20');
      }, 2000);
    }
  };
  
  // 메시지가 변경될 때마다 스크롤 이벤트 발생
  useEffect(() => {
    // 메시지가 렌더링된 후 스크롤 이벤트 발생
    const timer = setTimeout(() => {
      window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
    }, 100);
    
    return () => clearTimeout(timer);
  }, [message.content]);
  
  // 메시지 시간을 참조로 저장
  const messageTime = useMemo(() => {
    // 메시지가 타임스탬프를 가지고 있으면 그것을 사용, 아니면 현재 시간 생성
    return message.timestamp || new Date().getTime();
  }, [message.timestamp]);

  const formatMessageTime = (timestamp) => {
    if (!timestamp) return '방금 전';
    
    // timestamp가 숫자 또는 문자열인 경우 처리
    let dateObj;
    try {
      if (typeof timestamp === 'number' || !isNaN(parseInt(timestamp))) {
        dateObj = new Date(timestamp);
      } else if (typeof timestamp === 'string') {
        dateObj = new Date(timestamp);
      } else {
        return '방금 전';
      }
      
      // 올바른 날짜가 아닌 경우 (Invalid Date) 현재 시간으로 대체
      if (isNaN(dateObj.getTime())) {
        console.warn('Invalid date detected, using current time');
        dateObj = new Date();
      }
      
      // 현재 시간과의 차이 계산
      const now = new Date();
      const diffMs = now - dateObj;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);
      
      // 미래 시간인 경우 처리
      if (diffMs < 0) {
        return '방금 전';
      }
      
      // 상대적 시간 표시
      if (diffMins < 1) {
        return '방금 전';
      } else if (diffMins < 60) {
        return `${diffMins}분 전`;
      } else if (diffHours < 24) {
        return `${diffHours}시간 전`;
      } else if (diffDays < 7) {
        return `${diffDays}일 전`;
      } else {
        // 7일 이상 지난 경우 날짜 표시
        return dateObj.toLocaleDateString('ko-KR', {
          year: 'numeric',
          month: 'short',
          day: 'numeric'
        });
      }
    } catch (e) {
      console.error('Error formatting timestamp:', e);
      return '방금 전';
    }
  };

  const handleClosePreview = useCallback(() => {
    setPreviewSource(null);
    setPreviewContent("");
    setPreviewImage(null);
    setHighlightKeywords([]);
    setSourceKeywords([]);
    setSourceFilterText("");
  }, []);

  const handleCopy = useCallback(() => {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(message.content).then(
          () => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          },
          () => {
            console.error("Clipboard API failed");
            fallbackCopyTextToClipboard(message.content);
          }
        );
      } else {
        console.error("Clipboard API not supported");
        fallbackCopyTextToClipboard(message.content);
      }
    } catch (err) {
      console.error("Clipboard API not supported or failed:", err);
      fallbackCopyTextToClipboard(message.content);
    }
  }, [message.content]);

  // Clipboard API가 지원되지 않을 때 사용할 대체 복사 방법
  const fallbackCopyTextToClipboard = useCallback((text) => {
    try {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.top = "0";
      textArea.style.left = "0";
      textArea.style.position = "fixed";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      const successful = document.execCommand("copy");
      document.body.removeChild(textArea);
      if (successful) {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } else {
        alert("복사에 실패했습니다. 텍스트를 직접 선택해 복사해주세요.");
      }
    } catch (err) {
      console.error("Fallback copy failed:", err);
      alert("복사에 실패했습니다. 텍스트를 직접 선택해 복사해주세요.");
    }
  }, []);

  const handleFeedback = useCallback(
    async (type) => {
      // 이미 피드백을 보냈다면 토스트 메시지 표시
      if (feedbackSent) {
        setToast({
          visible: true,
          message: '이미 피드백을 제출하셨습니다',
          type: 'info'
        });
        return;
      }
      
      // 피드백 버튼 상태 변경
      setFeedback((current) => (current === type ? null : type));
      
      // 피드백 모달 열기
      setCurrentFeedbackType(type);
      setShowFeedbackModal(true);
      
      // 버튼 상호작용 상태 업데이트
      setHasInteractedWithButtons(true);
    },
    [feedbackSent]
  );
  
  // 피드백 제출 처리
  const handleSubmitFeedback = async (feedbackData) => {
    try {
      const response = await fetch("http://172.10.2.70:8000/api/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messageId: Date.now().toString(),
          feedbackType: feedbackData.feedbackType,
          rating: feedbackData.rating,
          content: message.content,
          reasons: feedbackData.reasons,
          comment: feedbackData.comment
        }),
      });
      
      if (!response.ok) {
        throw new Error(`피드백 전송 실패: ${response.status} ${response.statusText}`);
      }
      
      // 피드백 전송 성공
      setFeedbackSent(true);
      setToast({
        visible: true,
        message: '피드백이 제출되었습니다. 감사합니다!',
        type: 'success'
      });
      return true;
    } catch (err) {
      console.error("피드백 전송 중 오류 발생:", err);
      setToast({
        visible: true,
        message: '피드백 전송에 실패했습니다. 다시 시도해주세요.',
        type: 'error'
      });
      throw err;
    }
  };
  
  // 토스트 닫기 핸들러
  const handleCloseToast = () => {
    setToast(prev => ({ ...prev, visible: false }));
  };
  
  const handleStar = useCallback((n) => {
    setStar((current) => (current === n ? 0 : n));
  }, []);

  const extractKeywordsFromText = useCallback((text, maxKeywords = 12) => {
    if (!text || typeof text !== 'string') return [];
    
    try {
      // 한글, 영문, 숫자 단어 추출 (2글자 이상)
      const words = text.match(/[\wㄱ-ㅎ가-힣]{2,}/g) || [];
      
      // 불용어 제거 및 중복 제거
      const filteredWords = words
        .filter(word => word.length >= 2)
        .filter(word => !KOREAN_STOPWORDS.has(word))
        .map(word => word.toLowerCase());
      
      // 중복 제거 및 빈도수 계산
      const wordFreq = {};
      filteredWords.forEach(word => {
        wordFreq[word] = (wordFreq[word] || 0) + 1;
      });
      
      // 빈도수 기준 정렬 후 상위 N개 추출
      const uniqueKeywords = Object.entries(wordFreq)
        .sort((a, b) => b[1] - a[1])
        .slice(0, maxKeywords)
        .map(([word]) => word);
      
      return uniqueKeywords;
    } catch (error) {
      return [];
    }
  }, []);

  // 소스 미리보기 핸들러
  const handlePreviewSource = async (source) => {
    if (!source || !source.path) {
      // 오류 메시지 표시 방식 수정
      setPreviewContent("유효한 출처 정보가 없습니다.");
      setShowSourcePreview(true);
      return;
    }
    
    setShowSourcePreview(true);
    setPreviewSource(source);
    setPreviewContent("");
    setPreviewImage(null);
    setLoadingContent(true);
    
    try {
      // 키워드 추출 (응답 텍스트에서)
      let extractedKeywords = [];
      if (message.content) {
        extractedKeywords = extractKeywordsFromText(message.content);
      }
      
      // API 호출 - 출처 미리보기
      const response = await fetch('/api/source-preview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          path: source.path,
          page: source.page || 1,
          chunk_id: source.chunk_id || "",
          keywords: extractedKeywords,
          answer_text: message.content || ""
        }),
      });
      
      if (!response.ok) {
        throw new Error(`서버 응답 오류: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        // 이미지 또는 텍스트 컨텐츠 설정
        if (data.is_image) {
          setPreviewImage(data.image_data);
          setPreviewContent("");
        } else {
          setPreviewContent(data.content || "");
          setPreviewImage(null);
          
          // 하이라이트할 키워드 설정
          // API에서 제공한 키워드가 있으면 사용, 없으면 자체 추출한 키워드 사용
          const keywordsToUse = data.keywords && data.keywords.length > 0
            ? data.keywords
            : extractedKeywords;
            
          setHighlightKeywords(keywordsToUse);
        }
      } else {
        throw new Error(data.message || "출처 내용을 불러올 수 없습니다.");
      }
    } catch (error) {
      setPreviewContent(`출처 미리보기 오류: ${error.message}`);
    } finally {
      setLoadingContent(false);
    }
  };

  // 출처 목록 토글 함수
  const toggleSourcesVisible = useCallback(() => {
    setSourcesVisible((prevVisible) => !prevVisible);
  }, []);

  // 소스 콘텐츠 필터링 함수
  const filterSourceContent = useCallback((content, filterText) => {
    if (!filterText || !content) return content;
    
    const lines = content.split('\n');
    const filteredLines = lines.filter(line => 
      line.toLowerCase().includes(filterText.toLowerCase())
    );
    
    // 검색 결과가 없으면 원본 반환
    if (filteredLines.length === 0) return content;
    
    // 필터링된 라인에 하이라이트 적용
    return filteredLines.join('\n\n');
  }, []);

  // 소스 콘텐츠에 키워드 하이라이트 적용 함수
  const applyKeywordHighlighting = useCallback((content, keywords) => {
    if (!content || !keywords || keywords.length === 0) return content;
    
    // 문자열로 확실하게 변환
    let processedContent = String(content);
    
    // 정규식 특수문자 이스케이프
    const escapedKeywords = keywords.map(kw => String(kw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    
    // 정규식 패턴 생성
    const pattern = new RegExp(`(${escapedKeywords.join('|')})`, 'gi');
    
    // 마크다운 볼드체로 변환
    return processedContent.replace(pattern, '**$1**');
  }, []);

  // 마크다운으로 콘텐츠 렌더링 함수 - 출처 미리보기용
  const renderMarkdownContent = useCallback((content) => {
    if (!content) return null;
    
    // 스타일 주입 (한 번만)
    injectStyleOnce('highlight-style', keyframesStyle);
    
    // 필터 적용
    let displayContent = sourceFilterText ? filterSourceContent(content, sourceFilterText) : content;
    
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={{
            // 코드 블록 커스텀 렌더링
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              return !inline && match ? (
                <div className="relative">
                  <div className="absolute top-2 right-2 flex space-x-2">
                    <div className="text-xs text-gray-500 mr-2">
                      {match[1]}
                    </div>
                    <button
                      onClick={() => {
                        const code = String(children).replace(/\n$/, "");
                        navigator.clipboard.writeText(code);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      }}
                      className="p-1 rounded-md bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                      title="코드 복사"
                    >
                      {copied ? <FiCheck size={14} /> : <FiCopy size={14} />}
                    </button>
                  </div>
                  <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    className="rounded-md overflow-hidden !my-3"
                    showLineNumbers
                    wrapLines
                    {...props}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                </div>
              ) : (
                <code
                  className={`${className || ''} rounded-md bg-gray-800/80 dark:bg-gray-900/80 px-1.5 py-0.5 text-gray-200 dark:text-gray-200`}
                  {...props}
                >
                  {children}
                </code>
              );
            },
            // 키워드 하이라이트를 위한 텍스트 처리
            p({ node, children, ...props }) {
              // 소스 키워드가 있으면 하이라이트 적용
              if (highlightKeywords.length > 0 && typeof children === 'string') {
                // 정규식으로 키워드 하이라이트
                const parts = [];
                let lastIndex = 0;
                
                // 정규식 특수문자 이스케이프
                const escapedKeywords = highlightKeywords.map(kw => String(kw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
                // 더 정밀한 키워드 매칭을 위해 단어 경계 추가
                const pattern = new RegExp(`(${escapedKeywords.join('|')})`, 'gi');
                
                // 키워드 매칭 및 하이라이트 처리
                let match;
                const text = String(children);
                
                // 정규식 매칭 결과 처리
                while ((match = pattern.exec(text)) !== null) {
                  // 매치 이전 텍스트 추가
                  if (match.index > lastIndex) {
                    parts.push(text.substring(lastIndex, match.index));
                  }
                  
                  // 매치된 키워드를 하이라이트 처리
                  parts.push(
                    <span 
                      key={`kw-${match.index}`} 
                      className="highlight-keyword animate-highlight-pulse"
                    >
                      {match[0]}
                    </span>
                  );
                  
                  lastIndex = pattern.lastIndex;
                }
                
                // 마지막 텍스트 추가
                if (lastIndex < text.length) {
                  parts.push(text.substring(lastIndex));
                }
                
                return <p {...props}>{parts.length > 0 ? parts : children}</p>;
              }
              
              return <p {...props}>{children}</p>;
            },
            // 볼드체 텍스트 처리 (키워드 하이라이트용)
            strong({ node, children, ...props }) {
              return (
                <span 
                  className="highlight-keyword animate-highlight-pulse"
                  {...props}
                >
                  {children}
                </span>
              );
            },
            // mark 태그 처리 (line-highlight 스타일 적용)
            mark({ node, children, ...props }) {
              return (
                <mark 
                  className="bg-yellow-500/30 text-gray-800 dark:text-gray-200"
                  {...props}
                >
                  {children}
                </mark>
              );
            },
            // 일반 텍스트 노드에 대한 처리 추가
            text({ node, ...props }) {
              const { children } = props;
              
              // 텍스트 내용이 없으면 그대로 반환
              if (!children || typeof children !== 'string' || !highlightKeywords.length) {
                return <>{children}</>;
              }
              
              // 키워드 하이라이트 처리
              const parts = [];
              let lastIndex = 0;
              
              // 정규식 특수문자 이스케이프
              const escapedKeywords = highlightKeywords.map(kw => 
                String(kw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
              );
              
              // 키워드 패턴 생성
              const pattern = new RegExp(`(${escapedKeywords.join('|')})`, 'gi');
              
              // 텍스트 내에서 키워드 매칭
              let match;
              const text = children;
              
              while ((match = pattern.exec(text)) !== null) {
                // 매치 이전 텍스트 추가
                if (match.index > lastIndex) {
                  parts.push(text.substring(lastIndex, match.index));
                }
                
                // 하이라이트된 키워드 추가
                parts.push(
                  <span 
                    key={`kw-${match.index}`} 
                    className="highlight-keyword animate-highlight-pulse"
                  >
                    {match[0]}
                  </span>
                );
                
                lastIndex = pattern.lastIndex;
              }
              
              // 마지막 텍스트 추가
              if (lastIndex < text.length) {
                parts.push(text.substring(lastIndex));
              }
              
              return parts.length > 1 ? <>{parts}</> : <>{children}</>;
            }
          }}
        >
          {displayContent}
        </ReactMarkdown>
      </div>
    );
  }, [sourceFilterText, filterSourceContent, highlightKeywords, copied, injectStyleOnce, keyframesStyle]);

  // 원시 HTML 태그를 이스케이프 처리하는 함수
  const escapeHtmlTags = (content) => {
    if (!content) return '';
    
    // HTML 태그 감지 함수
    const detectMalformedTags = (text) => {
      const malformedTags = [];
      
      // 닫히지 않은 태그 찾기
      const openTagsRegex = /<([a-zA-Z][a-zA-Z0-9]*)[^>]*>/g;
      const closeTagsRegex = /<\/([a-zA-Z][a-zA-Z0-9]*)>/g;
      
      const openTags = {};
      let match;
      
      // 열린 태그 찾기
      while ((match = openTagsRegex.exec(text)) !== null) {
        const tagName = match[1].toLowerCase();
        openTags[tagName] = (openTags[tagName] || 0) + 1;
      }
      
      // 닫힌 태그 찾기
      while ((match = closeTagsRegex.exec(text)) !== null) {
        const tagName = match[1].toLowerCase();
        openTags[tagName] = (openTags[tagName] || 0) - 1;
      }
      
      // 균형이 맞지 않는 태그 찾기
      Object.entries(openTags).forEach(([tag, count]) => {
        if (count !== 0) {
          malformedTags.push(tag);
        }
      });
      
      return malformedTags.length > 0 ? malformedTags : null;
    };
    
    // 중국어 한자 감지 및 제거
    const detectChineseChars = (text) => {
      const chineseRegex = /[\u4e00-\u9fff]+/g;
      const match = text.match(chineseRegex);
      
      if (match) {
        // 중국어 한자를 공백으로 대체
        return text.replace(chineseRegex, ' ');
      }
      
      return text;
    };
    
    // 원본 콘텐츠 보존
    let processedContent = String(content);
    
    // 비정상 태그 패턴 감지 및 제거
    const malformedTags = detectMalformedTags(processedContent);
    if (malformedTags) {
      console.log('비정상 태그 감지:', malformedTags);
      // 비정상 태그 제거
      processedContent = processedContent.replace(new RegExp(`<${malformedTags.join('|')}>`, 'g'), '');
    }
    
    // 한자 제거 처리
    processedContent = detectChineseChars(processedContent);
    
    // ID 속성이 있는 span 태그 특별 처리 (이전 이미지의 주요 문제 패턴)
    processedContent = processedContent.replace(/<span\s+id=["']([^"']*)["']>(.*?)<\/span>/g, (match, id, content) => {
      console.log(`ID가 있는 span 태그 처리: id=${id}, content=${content}`);
      return content; // ID와 span 태그를 제거하고 내용만 유지
    });
    
    // ID를 포함한 모든 태그 속성 제거 (안전한 approach)
    processedContent = processedContent.replace(/<([a-zA-Z][a-zA-Z0-9]*)\s+[^>]*>/g, (match, tagName) => {
      // 허용된 태그만 남기고 모든 속성 제거
      const allowedTags = ['b', 'i', 'strong', 'em', 'u', 'code', 'pre', 'ul', 'ol', 'li', 'p', 'br', 'hr'];
      if (allowedTags.includes(tagName.toLowerCase())) {
        return `<${tagName}>`;
      }
      return ''; // 허용되지 않은 태그는 제거
    });
    
    // 마크다운 코드 블록 탐지 및 보호
    const codeBlockPattern = /```(\w*)\n([\s\S]*?)```/g;
    const codeBlocks = [];
    
    processedContent = processedContent.replace(codeBlockPattern, (match, language, code) => {
      const placeholder = `CODE_BLOCK_${codeBlocks.length}`;
      codeBlocks.push({placeholder, language, code});
      return placeholder;
    });
    
    // 마크다운 테이블 형식이 있는지 먼저 확인
    const hasMarkdownTable = /\|[\s-]+\|/.test(processedContent);
    
    // 테이블 변환 - 마크다운 테이블 형식으로 변환
    if (!hasMarkdownTable && /<table>|<tr>|<td>|<th>/.test(processedContent)) {
      try {
        // HTML 테이블을 마크다운으로 변환
        const tablePattern = /<table>([\s\S]*?)<\/table>/g;
        let matches;
        let lastIndex = 0;
        let result = '';
        
        while ((matches = tablePattern.exec(processedContent)) !== null) {
          // 테이블 앞 부분 추가
          result += processedContent.substring(lastIndex, matches.index);
          lastIndex = matches.index + matches[0].length;
          
          const tableContent = matches[1];
          // thead, tbody 태그 제거
          const cleanedContent = tableContent.replace(/<thead>|<\/thead>|<tbody>|<\/tbody>/g, '');
          
          // 행(row) 추출
          const rows = cleanedContent.match(/<tr>([\s\S]*?)<\/tr>/g);
          if (!rows) {
            // 추출 실패 시 원본 유지
            result += matches[0];
            continue;
          }
          
          const tableRows = [];
          let headerProcessed = false;
          
          // 각 행 처리
          for (const row of rows) {
            // 헤더(th) 및 데이터(td) 셀 추출
            const headerCells = row.match(/<th>([\s\S]*?)<\/th>/g);
            const cells = row.match(/<td>([\s\S]*?)<\/td>/g);
            
            // 헤더 행 처리
            if (headerCells && headerCells.length > 0) {
              const cellValues = headerCells.map(cell => 
                cell.replace(/<th>([\s\S]*?)<\/th>/g, '$1').trim());
              
              tableRows.push(`| ${cellValues.join(' | ')} |`);
              // 헤더와 본문 분리용 구분선 추가
              tableRows.push(`| ${cellValues.map(() => '---').join(' | ')} |`);
              headerProcessed = true;
            }
            // 데이터 행 처리
            else if (cells && cells.length > 0) {
              const cellValues = cells.map(cell => 
                cell.replace(/<td>([\s\S]*?)<\/td>/g, '$1').trim());
              
              // 헤더 행이 없었다면 첫 번째 데이터 행을 헤더로 사용
              if (!headerProcessed && tableRows.length === 0) {
                tableRows.push(`| ${cellValues.join(' | ')} |`);
                tableRows.push(`| ${cellValues.map(() => '---').join(' | ')} |`);
                headerProcessed = true;
              } else {
                tableRows.push(`| ${cellValues.join(' | ')} |`);
              }
            }
          }
          
          // 마크다운 테이블 추가
          if (tableRows.length > 0) {
            result += '\n' + tableRows.join('\n') + '\n';
          } else {
            // 변환에 실패한 경우 원본 텍스트 유지
            result += matches[0];
          }
        }
        
        // 테이블 이후 부분 추가
        result += processedContent.substring(lastIndex);
        processedContent = result;
      } catch (error) {
        console.error("테이블 변환 중 오류 발생:", error);
      }
    }
    
    // 닫히지 않은 br 태그 처리
    processedContent = processedContent.replace(/<br\s*\/?>/g, '\n');
    
    // p 태그 변환
    processedContent = processedContent.replace(/<p.*?>([\s\S]*?)<\/p>/g, '\n$1\n');
    
    // 기타 일반적인 태그 처리
    processedContent = processedContent
      .replace(/<b>([\s\S]*?)<\/b>/g, '**$1**')
      .replace(/<i>([\s\S]*?)<\/i>/g, '*$1*')
      .replace(/<strong>([\s\S]*?)<\/strong>/g, '**$1**')
      .replace(/<em>([\s\S]*?)<\/em>/g, '*$1*')
      .replace(/<ul>([\s\S]*?)<\/ul>/g, (match, content) => {
        return '\n' + content.replace(/<li>([\s\S]*?)<\/li>/g, '- $1\n');
      })
      .replace(/<ol>([\s\S]*?)<\/ol>/g, (match, content) => {
        return '\n' + content.replace(/<li>([\s\S]*?)<\/li>/g, '1. $1\n');
      });
    
    // 색상 태그 처리 (font color)
    processedContent = processedContent.replace(/<font color="([\s\S]*?)">([\s\S]*?)<\/font>/g, '**$2**');
    
    // 중복 마크다운 패턴 제거 (예: ****text**** -> **text**)
    processedContent = processedContent
      .replace(/\*{4,}([\s\S]*?)\*{4,}/g, '**$1**')
      .replace(/_{4,}([\s\S]*?)_{4,}/g, '__$1__');
    
    // 마크다운 헤더 중복 방지 (###, ##, # 등)
    processedContent = processedContent
      .replace(/^(#{1,6})\s*\1+\s*/gm, '$1 ');
    
    // 남은 모든 HTML 태그 제거 - 더 강력한 정규식 사용
    processedContent = processedContent.replace(/<[^>]*>?/gm, '');
    
    // 마크다운 코드 블록 복원
    codeBlocks.forEach(({placeholder, language, code}) => {
      processedContent = processedContent.replace(
        placeholder, 
        '```' + language + '\n' + code + '```'
      );
    });
    
    // 연속된 줄바꿈 정리 (3개 이상 -> 2개)
    processedContent = processedContent.replace(/\n{3,}/g, '\n\n');
    
    return processedContent;
  };

  // 마크다운 컴포넌트 생성 - 채팅 메시지용
  const MarkdownContent = useMemo(() => {
    // 헤딩에 ID 추가하는 함수
    const addHeadingIds = (content) => {
      return content.replace(/^(#{1,3})\s+(.+)$/gm, (match, hashes, text) => {
        const id = text.toLowerCase().replace(/[^\w\s가-힣]/g, '').replace(/\s+/g, '-');
        return `${hashes} ${text}`;  // ID 속성 제거, 단순 텍스트로 변환
      });
    };
    
    // 응답 텍스트가 비정상적인 패턴을 가지고 있는지 확인
    const hasAbnormalPatterns = (text) => {
      if (!text) return false;
      
      // 비정상 패턴 목록
      const abnormalPatterns = [
        /<span\s+id=/i,                // ID가 있는 span 태그
        /[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]/,  // 중국어 한자
        /<(\/?)([a-z]+)[^>]*>/i        // HTML 태그
      ];
      
      return abnormalPatterns.some(pattern => pattern.test(text));
    };
    
    // 콘텐츠 전처리 - 외부에서 정의된 messageContent 사용
    let processedContent = messageContent || "";
    
    // 객체가 문자열로 변환되지 않은 경우 처리
    if (typeof processedContent === 'object') {
      processedContent = JSON.stringify(processedContent);
    }
    
    // 비정상 패턴이 있는 경우에만 escapeHtmlTags 함수 적용 (성능 최적화)
    if (hasAbnormalPatterns(processedContent)) {
      processedContent = escapeHtmlTags(processedContent);
    }
    
    // Ensure content is a string before passing to ReactMarkdown
    const contentAsString = String(processedContent || '');
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none overflow-hidden">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          // rehypeRaw를 사용하여 HTML 직접 렌더링 허용
          rehypePlugins={[rehypeRaw, rehypeHighlight, rehypeKatex]} 
          components={{
            // ... (커스텀 컴포넌트들은 기존과 동일) ...
          }}
        >
          {contentAsString}
        </ReactMarkdown>
      </div>
    );
  }, [messageContent, escapeHtmlTags]); // messageContent 의존성으로 변경

  // 메시지 그룹핑 로직
  const isLastInGroup = !nextMessage || nextMessage.role !== message.role;
  
  // 이미지 URL 추출 및 처리
  const extractImageUrl = (content) => {
    const imageRegex = /!\[.*?\]\((.*?)\)/;
    const match = content.match(imageRegex);
    return match ? match[1] : null;
  };
  
  // 이미지 URL 추출 - 외부에서 정의된 messageContent 사용
  const imageUrl = extractImageUrl(messageContent);
  
  // 메시지 내용에서 이미지 마크다운 제거
  const cleanContent = messageContent.replace(/!\[.*?\]\(.*?\)/g, '').trim();
  
  // 검색어 하이라이트 처리
  const highlightSearchTerm = (text) => {
    if (!searchTerm || !text) return text;
    const regex = new RegExp(`(${searchTerm})`, 'gi');
    return text.split(regex).map((part, i) => 
      regex.test(part) ? <mark key={i} className="bg-yellow-500/30">{part}</mark> : part
    );
  };
  
  // 출처 정보 디버깅 로그 추가
  useEffect(() => {
    if (message.role === 'assistant') {
      console.log('출처 정보 상태:', JSON.stringify({
        hasSources: Boolean(message.sources),
        sourcesLength: message.sources?.length || 0,
        hasCitedSources: Boolean(message.cited_sources),
        citedSourcesLength: message.cited_sources?.length || 0,
        filteredSourcesLength: message.sources?.filter(s => s.is_cited)?.length || 0,
        sourcesVisible,
        isLastInGroup
      }, null, 2));
    }
  }, [message, sourcesVisible, isLastInGroup]);
  
  // 출처 표시 관련 로직 개선
  const filteredSources = useMemo(() => {
    if (!message.sources || !Array.isArray(message.sources) || message.sources.length === 0) {
      console.log("유효한 sources 배열이 없습니다:", message.sources);
      return [];
    }

    // source 속성이 없는 객체 필터링
    const validSources = message.sources.filter(source => {
      if (!source || typeof source !== 'object') {
        console.log("유효하지 않은 source 객체:", source);
        return false;
      }
      
      // source.path 또는 source.source 중 하나라도 있으면 유효한 소스로 간주
      return source.path || source.source;
    });

    // 중복 제거 (같은 파일 + 같은 페이지)
    const uniqueSources = validSources.filter((source, index, self) => {
      const sourcePath = source.path || source.source;
      return index === self.findIndex(s => 
        (s.path || s.source) === sourcePath && s.page === source.page
      );
    });

    // 필터링
    if (sourceFilterText) {
      return uniqueSources.filter(source => {
        const sourcePath = source.path || source.source;
        return sourcePath.toLowerCase().includes(sourceFilterText.toLowerCase());
      });
    }
    
    return uniqueSources;
  }, [message.sources, sourceFilterText]);
  
  // 출처 섹션 렌더링 함수
  const renderSourcesSection = () => {
    // 출처 정보 없는 경우 처리
    if (!message.sources || !Array.isArray(message.sources) || message.sources.length === 0) {
      return null;
    }
    
    // 인용된 출처만 필터링 (cited_sources가 있으면 우선 사용, 없으면 is_cited=true인 항목만 필터링)
    const citedSources = message.cited_sources && Array.isArray(message.cited_sources) && message.cited_sources.length > 0
      ? message.cited_sources 
      : message.sources.filter(s => s && s.is_cited === true);
    
    // 인용된 출처가 없는 경우 표시하지 않음
    if (citedSources.length === 0) {
      return null;
    }

    // 인용된 출처에 대해서만 필터링 적용
    const displaySources = sourceFilterText
      ? citedSources.filter(source => {
          const sourcePath = source.path || source.source || "";
          return sourcePath.toLowerCase().includes(sourceFilterText.toLowerCase());
        })
      : citedSources;

    return (
      <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <button
            onClick={() => setSourcesVisible(!sourcesVisible)}
            className="text-xs font-medium text-gray-500 dark:text-gray-400 flex items-center hover:text-indigo-500 dark:hover:text-indigo-400 transition-colors"
          >
            {sourcesVisible ? (
              <FiChevronDown className="mr-1" size={14} />
            ) : (
              <FiChevronRight className="mr-1" size={14} />
            )}
            출처 {displaySources.length}개
          </button>
          
          {sourcesVisible && displaySources.length > 3 && (
            <div className="relative">
              <input
                type="text"
                value={sourceFilterText}
                onChange={(e) => setSourceFilterText(e.target.value)}
                placeholder="출처 검색..."
                className="text-xs py-1 px-2 w-32 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
              {sourceFilterText && (
                <button
                  onClick={() => setSourceFilterText('')}
                  className="absolute right-1 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <FiX size={12} />
                </button>
              )}
            </div>
          )}
        </div>
        
        {sourcesVisible && (
          <div className="space-y-1 max-h-40 overflow-y-auto custom-scrollbar">
            {displaySources.length > 0 ? (
              displaySources.map((source, index) => (
                <SourceItem 
                  key={`${source.source || source.path}-${source.page}-${index}`}
                  source={source}
                  onClick={handlePreviewSource}
                />
              ))
            ) : (
              <p className="text-xs text-gray-500 dark:text-gray-400 py-2 text-center">
                {sourceFilterText ? '검색 결과가 없습니다.' : '출처 정보가 없습니다.'}
              </p>
            )}
          </div>
        )}
      </div>
    );
  };
  
  // 표와 이미지 데이터 추출 함수
  const extractTableAndImageData = () => {
    if (!message.sources || !Array.isArray(message.sources)) {
      return { tables: [], images: [] };
    }

    const tables = new Map(); // table_id로 그룹핑
    const images = [];

    message.sources.forEach(source => {
      // 표 데이터 처리
      if (source.element_type === 'table_row' && source.table_id && source.is_cited === true) {
        if (!tables.has(source.table_id)) {
          tables.set(source.table_id, {
            caption: typeof source.table_caption === 'string' ? source.table_caption : '',
            rows: [],
            source_info: {
              display_name: typeof source.display_name === 'string' ? source.display_name : '문서',
              page: source.page || ''
            }
          });
        }
        
        tables.get(source.table_id).rows.push({
          text: typeof source.text === 'string' ? source.text : '',
          row_index: source.row_index || 0
        });
      }
      
      // 이미지 데이터 처리 - 인용된 출처만 표시
      if (source.is_cited === true && source.has_images && source.images && Array.isArray(source.images)) {
        source.images.forEach((imagePath, idx) => {
          const caption = source.image_captions && source.image_captions[idx] 
            ? (typeof source.image_captions[idx].caption === 'string' ? source.image_captions[idx].caption : '')
            : '';
          
          images.push({
            path: imagePath,
            caption: caption,
            source_info: {
              display_name: typeof source.display_name === 'string' ? source.display_name : '문서',
              page: source.page || ''
            }
          });
        });
      }
    });

    // 표 데이터를 배열로 변환하고 행 정렬
    const tableArray = Array.from(tables.values()).map(table => ({
      ...table,
      rows: table.rows.sort((a, b) => (a.row_index || 0) - (b.row_index || 0))
    }));

    return { tables: tableArray, images };
  };

  // 메시지 내용 렌더링 함수
  const renderMessageContent = () => {
    if (showTypeWriter && !isUser) {
      return (
        <TypeWriter 
          text={messageContent || ""}
          speed={1} 
          onComplete={() => {
            setIsTypingComplete(true);
            setShowTypeWriter(false);
          }}
        />
      );
    }

    // 검색 모드에서 검색어 하이라이트
    const content = isSearchMode && searchTerm 
      ? highlightSearchTerm(messageContent || "")
      : messageContent || "";

    // 표와 이미지 데이터 추출 (어시스턴트 메시지인 경우에만)
    const { tables, images } = !isUser ? extractTableAndImageData() : { tables: [], images: [] };

    // Ensure content is a string before passing to ReactMarkdown
    const contentAsString = String(content || '');
    return (
      <div className="message-content-wrapper">
        {/* 기본 마크다운 콘텐츠 */}
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight, rehypeRaw]}
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              const language = match && match[1] ? match[1] : "";
              
              // 인라인 코드
              if (inline) {
                return (
                  <code
                    className="px-1.5 py-0.5 rounded-md bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 text-sm font-mono"
                    {...props}
                  >
                    {children}
                  </code>
                );
              }
              
              // 코드 블록
              return (
                <div className="relative my-0 rounded-lg overflow-hidden">
                  {language && (
                    <div className="absolute top-0 right-0 px-0.5 py-0 text-[18px] font-large text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 rounded-bl h-4 flex items-center">
                      {language}
                    </div>
                  )}
                  <SyntaxHighlighter
                    style={oneDark}
                    language={language}
                    PreTag="div"
                    className="!bg-gray-100 dark:!bg-gray-800 !rounded-xs !text-xs"
                    customStyle={{ lineHeight: '1.2', padding: '2px' }}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                </div>
              );
            },
            p({ node, children, ...props }) {
              return (
                <p className="mb-0 last:mb-0 text-base" style={{ lineHeight: '1.2', verticalAlign: 'baseline' }} {...props}>
                  {children}
                </p>
              );
            },
            a({ node, children, href, ...props }) {
              // 이미지 URL 처리
              if (href && /\.(jpg|jpeg|png|gif|webp)$/i.test(href)) {
                return (
                  <div className="my-4">
                    <img
                      src={href}
                      alt={children}
                      className="max-w-full h-auto rounded-lg shadow-md"
                      onClick={() => {
                        setImageUrl(href);
                        setShowImagePreview(true);
                      }}
                      style={{ cursor: 'pointer' }}
                    />
                  </div>
                );
              }
              
              // 일반 링크
              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-indigo-500 dark:text-indigo-400 hover:underline"
                  {...props}
                >
                  {children}
                  <FiExternalLink className="inline-block ml-1 mb-1" size={12} />
                </a>
              );
            }
          }}
        >
          {contentAsString}
        </ReactMarkdown>

        {/* 표 데이터 렌더링 */}
        {tables.length > 0 && (
          <div className="tables-section mt-4">
            <div className="section-header mb-3">
              <div className="flex items-center text-sm font-medium text-gray-600 dark:text-gray-400">
                <FiList className="mr-2" size={16} />
                관련 표 데이터 ({tables.length}개)
              </div>
            </div>
            {tables.map((table, idx) => (
              <div key={idx} className="mb-4">
                <TableDisplay 
                  tableData={table.rows} 
                  caption={table.caption || `${table.source_info.display_name} (페이지 ${table.source_info.page})`}
                />
              </div>
            ))}
          </div>
        )}

        {/* 이미지 데이터 렌더링 */}
        {images.length > 0 && (
          <div className="images-section mt-4">
            <div className="section-header mb-3">
              <div className="flex items-center text-sm font-medium text-gray-600 dark:text-gray-400">
                <FiImage className="mr-2" size={16} />
                관련 이미지 ({images.length}개)
              </div>
            </div>
            <ImageDisplay 
              images={images.map(img => img.path)} 
              captions={images.map(img => ({ caption: img.caption || `${img.source_info.display_name} (페이지 ${img.source_info.page})`, source_info: img.source_info }))}
            />
          </div>
        )}
      </div>
    );
  };
  
  return (
    <div
      id={messageId}
      className={`message-container ${isUser ? "user" : "assistant"} ${
        isGrouped ? "grouped" : ""
      } ${isSearchMode && searchTerm ? "search-mode" : ""}`}
    >
      {/* 스레드 라인 */}
      {threadStartLine && <div className="thread-start-line"></div>}
      {showThreadLine && <div className="thread-line"></div>}
      {threadEndLine && <div className="thread-end-line"></div>}
      
      <div
        className={`flex ${
          isUser ? "justify-end" : "justify-start"
        } relative mb-4 ${isGrouped ? "mt-1" : "mt-4"}`}
      >
        {/* 사용자 아바타 (왼쪽) */}
        {!isUser && (
          <div className="flex flex-col items-center mr-3">
            {!isGrouped && (
              <span className="message-time text-xs text-gray-400 mb-1">
                {formatMessageTime(messageTime)}
              </span>
            )}
            <ProfileAvatar role="assistant" isGrouped={isGrouped} />
          </div>
        )}
        
        {/* 메시지 말풍선 */}
        <div
          className={`message-bubble relative ${
            isUser
              ? "user-message bg-indigo-600 text-white"
              : "assistant-message bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200"
          } ${isGrouped ? "grouped" : ""}`}
        >
          {/* 메시지 내용 */}
          <div className={`message-content ${showTypeWriter ? "typing" : ""}`}>
            {/* 메시지 본문 */}
            <div className="message-body">
              {/* 목차 표시 */}
              {showTableOfContents && (
                <TableOfContents 
                  headings={headings} 
                  onClickHeading={scrollToHeading} 
                />
              )}
              
              {/* 마크다운 렌더링 */}
              <div ref={contentRef} className="markdown-content">
                {renderMessageContent()}
              </div>
              
              {/* 추천 질문 */}
              {suggestedQuestions && suggestedQuestions.length > 0 && (
                <div className="suggested-questions">
                  <p className="suggested-questions-title">
                    추천 질문:
                  </p>
                  <div className="suggested-questions-list">
                    {suggestedQuestions.map((question, index) => (
                      <button
                        key={index}
                        className="suggested-question-item"
                        onClick={() => onAskFollowUp(question)}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              
              {/* 출처 정보 */}
              {!isUser && renderSourcesSection()}
            </div>
            
            {/* 메시지 푸터 (피드백 버튼) */}
            {!isUser && (
              <div className="message-footer">
                <div className="feedback-buttons">
                  {/* 피드백 버튼들 */}
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* 사용자 아바타 (오른쪽) */}
        {isUser && (
          <div className="flex flex-col items-center ml-3">
            {!isGrouped && (
              <span className="message-time text-xs text-gray-400 mb-1">
                {formatMessageTime(messageTime)}
              </span>
            )}
            <ProfileAvatar role="user" isGrouped={isGrouped} />
          </div>
        )}
      </div>
      
      {/* 출처 미리보기 모달 */}
      <SourcePreviewModal
        isOpen={showSourcePreview}
        onClose={() => {
          setShowSourcePreview(false);
          handleClosePreview();
        }}
        source={previewSource}
        content={previewContent}
        image={previewImage}
        isLoading={loadingContent}
        keywords={highlightKeywords}
      />
      
      {/* 피드백 모달 */}
      <FeedbackModal
        isOpen={showFeedbackModal}
        onClose={() => setShowFeedbackModal(false)}
        messageContent={message.content}
        onSubmit={handleSubmitFeedback}
        feedbackType={currentFeedbackType}
      />
      
      {/* 피드백 토스트 */}
      <FeedbackToast
        isVisible={toast.visible}
        message={toast.message}
        type={toast.type}
        onClose={handleCloseToast}
      />
      
      {/* 이미지 미리보기 모달 */}
      {showImagePreview && imageUrl && (
        <ImagePreview
          src={imageUrl}
          alt="이미지"
          onClose={() => setShowImagePreview(false)}
        />
      )}
    </div>
  );
}

export default memo(ChatMessage);
