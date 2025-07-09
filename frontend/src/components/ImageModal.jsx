import React, { useState, useEffect } from 'react';
import { FiX, FiChevronLeft, FiChevronRight, FiDownload, FiMaximize2, FiMinimize2 } from 'react-icons/fi';

/**
 * 이미지 모달 컴포넌트
 * @param {Object} props - 컴포넌트 속성
 * @param {boolean} props.isOpen - 모달 열림 상태
 * @param {Function} props.onClose - 모달 닫기 핸들러
 * @param {Object} props.source - 출처 정보 객체
 * @param {Array} props.images - 이미지 목록
 */
const ImageModal = ({ isOpen, onClose, source, images = [] }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [sourceImages, setSourceImages] = useState([]);

  // 모달이 열릴 때 이미지 데이터 가져오기
  useEffect(() => {
    if (isOpen && source) {
      fetchSourceImages();
    }
  }, [isOpen, source]);

  // 키보드 이벤트 처리
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isOpen) return;
      
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          navigateImage(-1);
          break;
        case 'ArrowRight':
          navigateImage(1);
          break;
        case 'f':
        case 'F':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            setIsFullScreen(!isFullScreen);
          }
          break;
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, isFullScreen]);

  // 출처 이미지 가져오기
  const fetchSourceImages = async () => {
    if (!source || (!source.path && !source.source)) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/source-images', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          source_path: source.path || source.source,
          page: source.page || 1,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        setSourceImages(data.images || []);
        setCurrentIndex(0);
      } else {
        setError(data.message || '이미지를 가져오는 중 오류가 발생했습니다.');
      }
    } catch (err) {
      console.error('이미지 가져오기 오류:', err);
      setError('이미지를 가져오는 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  // 이미지 네비게이션
  const navigateImage = (direction) => {
    const totalImages = sourceImages.length;
    if (totalImages === 0) return;
    
    setCurrentIndex((prev) => {
      const newIndex = prev + direction;
      if (newIndex < 0) return totalImages - 1;
      if (newIndex >= totalImages) return 0;
      return newIndex;
    });
  };

  // 이미지 다운로드
  const handleDownload = async (imageUrl, fileName) => {
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName || 'image.jpg';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('이미지 다운로드 오류:', err);
    }
  };

  // 모달이 열려있지 않으면 렌더링하지 않음
  if (!isOpen) return null;

  const currentImage = sourceImages[currentIndex];
  const hasMultipleImages = sourceImages.length > 1;

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center ${isFullScreen ? 'bg-black' : 'bg-black bg-opacity-75'}`}>
      <div className={`relative ${isFullScreen ? 'w-full h-full' : 'max-w-4xl max-h-[90vh] w-full mx-4'} bg-white dark:bg-gray-900 rounded-lg overflow-hidden`}>
        {/* 헤더 */}
        <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white truncate">
              {source?.display_name || '이미지 보기'}
            </h3>
            {source?.page && (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                페이지 {source.page}
              </p>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            {/* 이미지 네비게이션 정보 */}
            {hasMultipleImages && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {currentIndex + 1} / {sourceImages.length}
              </span>
            )}
            
            {/* 전체화면 토글 */}
            <button
              onClick={() => setIsFullScreen(!isFullScreen)}
              className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              title={isFullScreen ? '전체화면 해제' : '전체화면'}
            >
              {isFullScreen ? (
                <FiMinimize2 size={20} className="text-gray-600 dark:text-gray-400" />
              ) : (
                <FiMaximize2 size={20} className="text-gray-600 dark:text-gray-400" />
              )}
            </button>
            
            {/* 닫기 버튼 */}
            <button
              onClick={onClose}
              className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              title="닫기"
            >
              <FiX size={20} className="text-gray-600 dark:text-gray-400" />
            </button>
          </div>
        </div>

        {/* 이미지 콘텐츠 */}
        <div className="relative flex-1 overflow-hidden">
          {isLoading && (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
              <span className="ml-2 text-gray-600 dark:text-gray-400">이미지 로딩 중...</span>
            </div>
          )}

          {error && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <p className="text-red-500 dark:text-red-400 mb-2">{error}</p>
                <button
                  onClick={fetchSourceImages}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
                >
                  다시 시도
                </button>
              </div>
            </div>
          )}

          {!isLoading && !error && sourceImages.length === 0 && (
            <div className="flex items-center justify-center h-64">
              <p className="text-gray-500 dark:text-gray-400">이미지가 없습니다.</p>
            </div>
          )}

          {!isLoading && !error && currentImage && (
            <div className="relative h-full flex items-center justify-center">
              {/* 이미지 */}
              <img
                src={currentImage.url}
                alt={currentImage.caption || `이미지 ${currentIndex + 1}`}
                className={`max-w-full max-h-full object-contain ${isFullScreen ? 'w-full h-full' : ''}`}
                loading="lazy"
              />
              
              {/* 네비게이션 버튼 */}
              {hasMultipleImages && (
                <>
                  <button
                    onClick={() => navigateImage(-1)}
                    className="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-black bg-opacity-50 hover:bg-opacity-75 text-white transition-all"
                    title="이전 이미지"
                  >
                    <FiChevronLeft size={24} />
                  </button>
                  
                  <button
                    onClick={() => navigateImage(1)}
                    className="absolute right-4 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-black bg-opacity-50 hover:bg-opacity-75 text-white transition-all"
                    title="다음 이미지"
                  >
                    <FiChevronRight size={24} />
                  </button>
                </>
              )}
              
              {/* 다운로드 버튼 */}
              <button
                onClick={() => handleDownload(currentImage.url, currentImage.path?.split('/').pop() || 'image.jpg')}
                className="absolute bottom-4 right-4 p-2 rounded-full bg-black bg-opacity-50 hover:bg-opacity-75 text-white transition-all"
                title="이미지 다운로드"
              >
                <FiDownload size={20} />
              </button>
            </div>
          )}
        </div>

        {/* 이미지 정보 */}
        {!isLoading && !error && currentImage && (
          <div className="p-4 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                {currentImage.caption && (
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {currentImage.caption}
                  </p>
                )}
                
                {currentImage.ocr_text && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    OCR: {currentImage.ocr_text.substring(0, 100)}
                    {currentImage.ocr_text.length > 100 && '...'}
                  </p>
                )}
                
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                  페이지 {currentImage.page || 1}
                </p>
              </div>
              
              {hasMultipleImages && (
                <div className="flex items-center gap-1 ml-4">
                  {sourceImages.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => setCurrentIndex(index)}
                      className={`w-2 h-2 rounded-full transition-colors ${
                        index === currentIndex 
                          ? 'bg-indigo-600' 
                          : 'bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500'
                      }`}
                      title={`이미지 ${index + 1}`}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageModal;
