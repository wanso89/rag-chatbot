import React from 'react';
import { FiBookmark, FiFileText, FiFile, FiExternalLink } from 'react-icons/fi';

/**
 * 출처 아이템 컴포넌트
 * @param {Object} props - 컴포넌트 속성
 * @param {Object} props.source - 출처 정보 객체
 * @param {Function} props.onClick - 클릭 핸들러 함수
 * @param {boolean} props.isFiltered - 필터링된 항목인지 여부
 * @param {boolean} props.isCited - 인용된 출처인지 여부
 * @param {boolean} props.isReference - 참고 문서인지 여부
 */
const SourceItem = ({ source, onClick, isFiltered = false, isCited = false, isReference = false }) => {
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
  
  return (
    <div 
      onClick={() => onClick(source)}
      className={`flex items-center px-3 py-2 rounded-md cursor-pointer transition-all
        ${isCited ? 'bg-indigo-50 dark:bg-indigo-900/30 hover:bg-indigo-100 dark:hover:bg-indigo-800/40 shadow-sm' : 
          isReference ? 'bg-gray-50 dark:bg-gray-800/60 hover:bg-gray-100 dark:hover:bg-gray-800/80' : 
          'bg-gray-50 dark:bg-gray-800/40 hover:bg-gray-100 dark:hover:bg-gray-800/60'}
        ${isFiltered ? 'border-l-2 border-yellow-400' : ''}`}
      title={`${isCited ? '인용 출처' : '참고 문서'} - ${displayName}${pageInfo ? ` (${pageInfo})` : ''}`}
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
      <div className="flex-grow min-w-0">
        <p className={`text-sm font-medium ${isCited ? 'text-gray-800 dark:text-gray-200' : 'text-gray-600 dark:text-gray-300'} truncate`}>
          {displayName}
        </p>
        {pageInfo && (
          <span className="ml-1 text-gray-500 dark:text-gray-500">
            {pageInfo}
          </span>
        )}
      </div>
      <div className="ml-2 p-1.5 rounded-full bg-gray-100 dark:bg-gray-800/70 text-gray-500 dark:text-gray-400 hover:bg-indigo-100 dark:hover:bg-indigo-800/50 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">
        <FiExternalLink size={14} />
      </div>
    </div>
  );
};

export default SourceItem;
