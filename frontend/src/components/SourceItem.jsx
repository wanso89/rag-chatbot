import React, { useState } from 'react';
import { FiBookmark, FiFileText, FiFile, FiExternalLink, FiImage, FiEye } from 'react-icons/fi';

/**
 * 출처 아이템 컴포넌트
 * @param {Object} props - 컴포넌트 속성
 * @param {Object} props.source - 출처 정보 객체
 * @param {Function} props.onClick - 클릭 핸들러 함수
 * @param {boolean} props.isFiltered - 필터링된 항목인지 여부
 * @param {boolean} props.isCited - 인용된 출처인지 여부
 * @param {boolean} props.isReference - 참고 문서인지 여부
 * @param {Function} props.onViewImages - 이미지 보기 핸들러 함수
 */
const SourceItem = ({ source, onClick, isFiltered = false, isCited = false, isReference = false, onViewImages }) => {
  const [showImagePreview, setShowImagePreview] = useState(false);
  // 파일명 정제 - display_name 우선 사용, 없으면 UUID 제거 후 사용
  const getCleanFileName = (filePath) => {
    if (!filePath) return '알 수 없는 출처';
    
    const fileName = filePath.split('/').pop();
    // UUID 패턴 제거 (UUID_파일명.확장자 형식)
    const uuidPattern = /^[a-f0-9]{8,}-?[a-f0-9-]*_/i;
    return fileName.replace(uuidPattern, '');
  };
  
  const displayName = source.display_name || 
    getCleanFileName(source.path || source.source) || '알 수 없는 출처';
  
  // 페이지 정보
  const pageInfo = source.page && source.page > 0 ? `p.${source.page}` : '';
  
  // 관련성 점수
  const relevanceScore = typeof source.score === 'number' ? 
    source.score > 0.7 ? '높음' : 
    source.score > 0.4 ? '중간' : '낮음' : '';
    
  // 이미지 정보
  const hasImages = source.has_images || false;
  const imageCount = source.image_count || (source.images && source.images.length) || 0;
  const processedImages = source.processed_images || [];
  
  // 이미지 버튼 클릭 핸들러
  const handleImageClick = (e) => {
    e.stopPropagation(); // 부모 클릭 이벤트 방지
    if (onViewImages && hasImages) {
      onViewImages(source);
    }
  };
  
  // 문서 보기 핸들러
  const handleDocumentClick = () => {
    if (onClick) {
      onClick(source);
    }
  };
  
  return (
    <div 
      className={`flex items-center px-3 py-2 rounded-md transition-all
        ${isCited ? 'bg-indigo-50 dark:bg-indigo-900/30 hover:bg-indigo-100 dark:hover:bg-indigo-800/40 shadow-sm' : 
          isReference ? 'bg-gray-50 dark:bg-gray-800/60 hover:bg-gray-100 dark:hover:bg-gray-800/80' : 
          'bg-gray-50 dark:bg-gray-800/40 hover:bg-gray-100 dark:hover:bg-gray-800/60'}
        ${isFiltered ? 'border-l-2 border-yellow-400' : ''}`}
    >
      <div className={`mr-2 p-1.5 rounded-full flex-shrink-0 
        ${isCited ? 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400' : 
          isReference ? 'bg-gray-100 dark:bg-gray-800/70 text-gray-500 dark:text-gray-400' : 
          'bg-gray-100 dark:bg-gray-800/50 text-gray-500'}`}>
        {isCited ? (
          <FiBookmark size={14} />
        ) : isReference ? (
          <FiFileText size={14} />
        ) : (
          <FiFile size={14} />
        )}
      </div>
      
      <div 
        className="flex-grow min-w-0 cursor-pointer"
        onClick={handleDocumentClick}
        title={`${isCited ? '인용 출처' : '참고 문서'} - ${displayName}${pageInfo ? ` (${pageInfo})` : ''}`}
      >
        <div className="flex items-center">
          <p className={`text-sm font-medium ${isCited ? 'text-gray-800 dark:text-gray-200' : 'text-gray-600 dark:text-gray-300'} truncate`}>
            {displayName}
          </p>
          {pageInfo && (
            <span className="ml-1 text-gray-500 dark:text-gray-500 flex-shrink-0">
              {pageInfo}
            </span>
          )}
          {hasImages && (
            <div className="ml-2 flex items-center">
              <FiImage size={12} className="text-green-500 dark:text-green-400 mr-1" />
              <span className="text-xs text-green-600 dark:text-green-400">
                {imageCount}개
              </span>
            </div>
          )}
        </div>
        
        {/* 이미지 미리보기 (간단히 표시) */}
        {hasImages && showImagePreview && processedImages.length > 0 && (
          <div className="mt-2 flex gap-1 overflow-x-auto">
            {processedImages.slice(0, 3).map((img, index) => (
              <div key={index} className="flex-shrink-0">
                <img 
                  src={img.url} 
                  alt={img.caption || `이미지 ${index + 1}`}
                  className="w-12 h-12 object-cover rounded border"
                  loading="lazy"
                />
              </div>
            ))}
            {processedImages.length > 3 && (
              <div className="flex-shrink-0 w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded border flex items-center justify-center">
                <span className="text-xs text-gray-500">+{processedImages.length - 3}</span>
              </div>
            )}
          </div>
        )}
      </div>
      
      <div className="ml-2 flex gap-1">
        {/* 이미지 보기 버튼 */}
        {hasImages && (
          <button
            onClick={handleImageClick}
            className="p-1.5 rounded-full bg-green-100 dark:bg-green-900/40 text-green-600 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-800/50 transition-colors"
            title={`이미지 보기 (${imageCount}개)`}
          >
            <FiEye size={14} />
          </button>
        )}
        
        {/* 문서 보기 버튼 */}
        <button
          onClick={handleDocumentClick}
          className="p-1.5 rounded-full bg-gray-100 dark:bg-gray-800/70 text-gray-500 dark:text-gray-400 hover:bg-indigo-100 dark:hover:bg-indigo-800/50 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
          title="문서 보기"
        >
          <FiExternalLink size={14} />
        </button>
      </div>
    </div>
  );
};

export default SourceItem;
