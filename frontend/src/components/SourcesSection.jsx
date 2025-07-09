import React, { useState } from 'react';
import { FiChevronDown, FiChevronRight, FiX } from 'react-icons/fi';
import SourceItem from './SourceItem';

/**
 * 출처 문서 섹션 컴포넌트
 * @param {Object} props - 컴포넌트 속성
 * @param {Array} props.sources - 출처 문서 목록
 * @param {Array} props.citedSources - 인용된 출처 문서 목록
 * @param {Function} props.onViewSource - 출처 문서 보기 핸들러
 * @param {Function} props.onViewImages - 이미지 보기 핸들러
 */
const SourcesSection = ({ sources = [], citedSources = [], onViewSource, onViewImages }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [filterText, setFilterText] = useState("");
  
  // 인용된 출처만 필터링 (cited_sources가 있으면 우선 사용, 없으면 is_cited=true인 항목만 필터링)
  const displayCitedSources = citedSources.length > 0 
    ? citedSources 
    : sources.filter(s => s && s.is_cited === true);
  
  // 참고 문서 - 인용되지 않은 소스 중 상위 2개 (인용된 소스가 없을 때만)
  const referenceSources = [];
  if (displayCitedSources.length === 0 && sources.length > 0) {
    // 인용된 소스가 없을 때 상위 2개 문서를 참고 문서로 표시
    const sortedSources = [...sources]
      .filter(s => s && (s.path || s.source)) // 유효한 소스만 필터링
      .sort((a, b) => (b.score || 0) - (a.score || 0)) // 점수 기준 내림차순 정렬
      .slice(0, 2); // 상위 2개만 선택
    
    referenceSources.push(...sortedSources);
  }
  
  // 출처 정보 없는 경우 처리
  if (displayCitedSources.length === 0 && referenceSources.length === 0) {
    return null;
  }
  
  // 필터링 처리
  const filteredCitedSources = filterText
    ? displayCitedSources.filter(source => {
        const sourcePath = source.path || source.source || "";
        return sourcePath.toLowerCase().includes(filterText.toLowerCase());
      })
    : displayCitedSources;
    
  const filteredReferenceSources = filterText
    ? referenceSources.filter(source => {
        const sourcePath = source.path || source.source || "";
        return sourcePath.toLowerCase().includes(filterText.toLowerCase());
      })
    : referenceSources;
  
  const hasSourcesToDisplay = filteredCitedSources.length > 0 || filteredReferenceSources.length > 0;
  
  return (
    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={() => setIsVisible(!isVisible)}
          className="sources-button"
        >
          {isVisible ? (
            <FiChevronDown className="mr-1" size={14} />
          ) : (
            <FiChevronRight className="mr-1" size={14} />
          )}
          {filteredCitedSources.length > 0 ? `출처 문서 (${filteredCitedSources.length})` : 
           filteredReferenceSources.length > 0 ? `참고 문서 (${filteredReferenceSources.length})` : 
           '출처 정보'}
        </button>
        
        {isVisible && (filteredCitedSources.length + filteredReferenceSources.length) > 3 && (
          <div className="relative">
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="출처 검색..."
              className="text-xs py-1 px-2 w-32 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            {filterText && (
              <button
                onClick={() => setFilterText('')}
                className="absolute right-1 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <FiX size={12} />
              </button>
            )}
          </div>
        )}
      </div>
      
      {isVisible && (
        <div className="sources-list">
          {/* 인용 출처 표시 */}
          {filteredCitedSources.length > 0 && (
            <>
              {filteredCitedSources.length > 0 && filteredReferenceSources.length > 0 && (
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 mt-1 px-2">
                  인용 출처:
                </div>
              )}
              {filteredCitedSources.map((source, index) => (
                <SourceItem 
                  key={`cited-${source.source || source.path}-${source.page}-${index}`}
                  source={source}
                  onClick={onViewSource}
                  onViewImages={onViewImages}
                  isCited={true}
                />
              ))}
            </>
          )}
          
          {/* 참고 문서 표시 */}
          {filteredReferenceSources.length > 0 && (
            <>
              {filteredCitedSources.length > 0 && filteredReferenceSources.length > 0 && (
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 mt-2 px-2">
                  참고 문서:
                </div>
              )}
              {filteredReferenceSources.map((source, index) => (
                <SourceItem 
                  key={`ref-${source.source || source.path}-${source.page}-${index}`}
                  source={source}
                  onClick={onViewSource}
                  onViewImages={onViewImages}
                  isReference={true}
                />
              ))}
            </>
          )}
          
          {/* 필터링 결과가 없는 경우 */}
          {!hasSourcesToDisplay && (
            <p className="text-xs text-gray-500 dark:text-gray-400 py-2 text-center">
              {filterText ? '검색 결과가 없습니다.' : '출처 정보가 없습니다.'}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default SourcesSection;
