import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import { useRef, useEffect, useState, useMemo, useCallback } from "react";
// FiExternalLink 아이콘 추가
import { FiLoader, FiArrowUp, FiList, FiX, FiExternalLink, FiTrash2, FiHardDrive, FiFile, FiFolder, FiSearch, FiMessageSquare, FiBookmark, FiUploadCloud, FiPlus, FiCornerDownRight, FiCommand, FiMessageCircle, FiDatabase, FiBarChart2, FiSquare, FiStopCircle } from "react-icons/fi"; 
import { FiAlertCircle, FiFileText, FiHelpCircle } from "react-icons/fi";

// 로딩 인디케이터 컴포넌트 추가
const LoadingIndicator = ({ active }) => {
  // 로딩 시간이 길어질 경우 보여줄 다양한 메시지들
  const loadingMessages = [
    "문서를 검색하고 답변을 작성하고 있습니다",
    "관련 정보를 분석하고 있습니다",
    "최적의 답변을 생성하고 있습니다",
    "조금만 더 기다려주세요"
  ];
  
  // 메시지를 주기적으로 변경하기 위한 상태
  const [messageIndex, setMessageIndex] = useState(0);
  
  // 활성화되면 메시지 변경 타이머 시작
  useEffect(() => {
    if (!active) return;
    
    const interval = setInterval(() => {
      setMessageIndex(prev => (prev + 1) % loadingMessages.length);
    }, 3000);
    
    return () => clearInterval(interval);
  }, [active, loadingMessages.length]);
  
  if (!active) return null;
  
  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm flex flex-col items-center justify-center z-[60] animate-fade-in">
      <div className="relative flex flex-col items-center p-10 rounded-2xl bg-gray-800/40 backdrop-blur-lg shadow-2xl">
        {/* 메인 로딩 원형 */}
        <div className="w-24 h-24 rounded-full border-4 border-indigo-500/20 relative flex items-center justify-center">
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin"></div>
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500/10 border-b-transparent animate-ping opacity-30"></div>
          
          {/* 내부 원형 */}
          <div className="absolute inset-5 rounded-full bg-indigo-500/10 flex items-center justify-center">
            <FiMessageSquare className="text-indigo-400 animate-pulse" size={20} />
          </div>
          
          {/* 주변 빛나는 효과 */}
          <div className="absolute -inset-2 rounded-full bg-indigo-500/5 blur-xl"></div>
        </div>
        
        {/* 텍스트 */}
        <div className="mt-8 text-center space-y-2">
          <p className="text-gray-200 font-medium">응답 생성 중...</p>
          <p className="text-gray-400 text-sm min-h-[20px] transition-all duration-500">
            {loadingMessages[messageIndex]}
          </p>
        </div>
        
        {/* 로딩 점 애니메이션 */}
        <div className="flex items-center space-x-2 mt-4">
          <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '200ms' }}></div>
          <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '400ms' }}></div>
        </div>
      </div>
    </div>
  );
};

// 전역 스타일 (CSS-in-JS 방식으로 변경)
const globalStyles = `
  .custom-scrollbar::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
    border-radius: 8px;
    margin: 2px;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background-color: rgba(156, 163, 175, 0.3);
    border-radius: 8px;
    border: 2px solid transparent;
    background-clip: padding-box;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background-color: rgba(156, 163, 175, 0.5);
  }
  
  /* Dark mode scrollbar */
  @media (prefers-color-scheme: dark) {
    .custom-scrollbar::-webkit-scrollbar-thumb {
      background-color: rgba(156, 163, 175, 0.2);
    }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover {
      background-color: rgba(156, 163, 175, 0.4);
    }
  }
  
  @keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  @keyframes slide-up {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  @keyframes fade-in-up {
    from { transform: translateY(10px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  @keyframes shrink-fade-out {
    from { transform: scale(1); opacity: 1; }
    to { transform: scale(0.9); opacity: 0; }
  }
  
  .animate-fade-in {
    animation: fade-in 0.2s ease-out forwards;
  }
  
  .animate-slide-up {
    animation: slide-up 0.3s ease-out forwards;
  }
  
  .animate-shrink-fade-out {
    animation: shrink-fade-out 0.3s ease-out forwards;
  }
`;

// 스타일 요소를 DOM에 주입하는 함수
const injectGlobalStyles = () => {
  // 이미 존재하는지 확인
  if (!document.getElementById('custom-chat-styles')) {
    const styleElement = document.createElement('style');
    styleElement.id = 'custom-chat-styles';
    styleElement.innerHTML = globalStyles;
    document.head.appendChild(styleElement);
    
    // 추가 애니메이션 스타일 주입
    const additionalStyles = `
      @keyframes fade-in-up {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
      }
      
      .animate-fade-in-up {
        animation: fade-in-up 0.5s ease-out forwards;
      }
    `;
    
    // 이미 존재하는지 확인
    if (!document.getElementById('additional-animation-styles')) {
      const animStyleElement = document.createElement('style');
      animStyleElement.id = 'additional-animation-styles';
      animStyleElement.innerHTML = additionalStyles;
      document.head.appendChild(animStyleElement);
    }
  }
};

// 삭제 확인 모달 컴포넌트 추가
const DeleteConfirmModal = ({ isOpen, onClose, onConfirm, fileName, isDeleting }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[60] p-4 animate-fade-in">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-5 sm:p-6 max-w-md w-full shadow-2xl animate-slide-up">
        <div className="flex items-start mb-4">
          <div className="flex-shrink-0 text-red-500">
            <FiAlertCircle size={24} />
          </div>
          <div className="ml-3">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              파일 삭제 확인
            </h3>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              <span className="font-medium text-gray-800 dark:text-gray-300">{fileName}</span> 파일을 삭제하시겠습니까?
            </p>
            <p className="mt-1 text-xs text-red-500">
              이 작업은 되돌릴 수 없으며, 파일과 관련된 모든 인덱스 데이터가 삭제됩니다.
            </p>
          </div>
        </div>
        <div className="flex justify-end space-x-3 mt-5">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-300 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-400"
            disabled={isDeleting}
          >
            취소
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50 transition flex items-center"
            disabled={isDeleting}
          >
            {isDeleting ? (
              <>
                <FiLoader className="animate-spin mr-2" size={16} />
                삭제 중...
              </>
            ) : (
              <>삭제</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// 인덱싱된 파일 목록 컴포넌트 추가
const IndexedFilesModal = ({ isOpen, onClose }) => {
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [deleteAnimation, setDeleteAnimation] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoadingRef, setIsLoadingRef] = useState(false); // 로딩 상태 참조 추가
  const [showDeleteAllConfirm, setShowDeleteAllConfirm] = useState(false); // 전체 삭제 확인 모달 상태
  
  const modalRef = useRef(null);
  const hasLoadedRef = useRef(false); // 이미 로드했는지 체크하는 참조 추가

  // 파일 목록 가져오기 (최적화된 버전)
  useEffect(() => {
    // 모달이 열릴 때만 파일 목록을 가져오고, 한 번만 가져오도록 수정
    if (isOpen && !hasLoadedRef.current && !isLoadingRef) {
      loadFiles();
      hasLoadedRef.current = true; // 로드 완료 표시
    } else if (!isOpen) {
      // 모달이 닫힐 때 참조값 초기화
      hasLoadedRef.current = false;
    }
  }, [isOpen, isLoadingRef]);
  
  // 외부에서 파일 목록 강제 새로고침 이벤트 처리 (최적화)
  useEffect(() => {
    const handleForceRefresh = () => {
      // 모달이 열려있고 로딩 중이 아닐 때만 로드
      if (isOpen && !isLoadingRef) {
        hasLoadedRef.current = false; // 강제 새로고침 시 참조값 초기화
        loadFiles();
      }
    };
    
    window.addEventListener('forceRefreshFiles', handleForceRefresh);
    
    return () => {
      window.removeEventListener('forceRefreshFiles', handleForceRefresh);
    };
  }, [isOpen, isLoadingRef]);

  // 모달 바깥 클릭 시 닫기
  useEffect(() => {
    function handleClickOutside(event) {
      if (modalRef.current && !modalRef.current.contains(event.target)) {
        onClose();
      }
    }
    
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => {
        document.removeEventListener("mousedown", handleClickOutside);
      };
    }
  }, [isOpen, onClose]);

  // 성공/에러 메시지 자동 제거
  useEffect(() => {
    let timer;
    if (successMessage || deleteError) {
      timer = setTimeout(() => {
        setSuccessMessage(null);
        setDeleteError(null);
      }, 3000);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [successMessage, deleteError]);

  const loadFiles = async () => {
    // 이미 로딩 중인 경우 중복 요청 방지
    if (isLoadingRef) {
      return;
    }
    
    setIsLoading(true);
    setIsLoadingRef(true); // 로딩 상태 참조 업데이트
    setError(null);
    
    try {
      const controller = new AbortController(); // 요청 취소 컨트롤러
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10초 타임아웃
      
      const response = await fetch('/api/indexed-files', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId); // 타임아웃 제거
      
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        setFiles(data.files || []);
        hasLoadedRef.current = true; // 로드 완료 표시
      } else {
        throw new Error(data.detail || data.message || '파일 목록을 불러오는데 실패했습니다.');
      }
    } catch (err) {
      // 오류 타입에 따른 메시지 설정
      let errorMsg = '파일 목록을 불러오는데 실패했습니다.';
      
      if (err.name === 'AbortError') {
        errorMsg = '서버 응답 시간이 너무 깁니다. 서버 상태를 확인해주세요.';
      } else if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        errorMsg = '서버에 연결할 수 없습니다. 네트워크 연결 또는 서버 상태를 확인해주세요.';
      } else if (err.message) {
        errorMsg = `${errorMsg} ${err.message}`;
      }
      
      setError(errorMsg);
    } finally {
      setIsLoading(false);
      setIsLoadingRef(false); // 로딩 상태 참조 업데이트
    }
  };

  const handleDeleteClick = async (filename) => {
    try {
      setIsDeleting(true);
      setDeleteAnimation(filename);
      
      // UI를 먼저 업데이트하여 사용자에게 즉각적인 피드백 제공
      setFiles(prev => prev.filter(file => file !== filename));
      
      // 서버에 삭제 요청
      const response = await fetch(`/api/delete-file?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        // 성공 메시지만 표시 (UI는 이미 업데이트됨)
        setSuccessMessage(`${cleanFilename(filename)} 파일이 삭제되었습니다.`);
        setDeleteAnimation(null);
      } else {
        // 실패 시 파일 목록 복원
        loadFiles(); // 파일 목록 다시 불러오기
        throw new Error(data.message || '파일 삭제 중 오류가 발생했습니다.');
      }
    } catch (err) {
      console.error('파일 삭제 오류:', err);
      setDeleteError('파일 삭제에 실패했습니다. 다시 시도해주세요.');
      setDeleteAnimation(null);
      
      // 실패 시 파일 목록 다시 불러오기
      loadFiles();
    } finally {
      setIsDeleting(false);
    }
  };

  const cleanFilename = (filename) => {
    // UUID와 같은 접두사가 있다면 제거
    const parts = filename.split('_');
    if (parts.length > 1 && parts[0].length >= 20) {
      return parts.slice(1).join('_');
    }
    return filename; 
  };

  const getFileIcon = (filename) => {
    const ext = filename.split('.').pop().toLowerCase();
    
    // 파일 확장자에 따라 아이콘 결정
    switch (ext) {
      case 'pdf':
        return <div className="w-8 h-8 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center text-red-600 dark:text-red-400">PDF</div>;
      case 'doc':
      case 'docx':
        return <div className="w-8 h-8 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-400">DOC</div>;
      case 'xls':
      case 'xlsx':
        return <div className="w-8 h-8 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center text-green-600 dark:text-green-400">XLS</div>;
      case 'ppt':
      case 'pptx':
        return <div className="w-8 h-8 rounded-lg bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center text-orange-600 dark:text-orange-400">PPT</div>;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
        return <div className="w-8 h-8 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 dark:text-purple-400">IMG</div>;
      case 'txt':
        return <div className="w-8 h-8 rounded-lg bg-gray-100 dark:bg-gray-700 flex items-center justify-center text-gray-600 dark:text-gray-400">TXT</div>;
      default:
        return <div className="w-8 h-8 rounded-lg bg-gray-100 dark:bg-gray-700 flex items-center justify-center text-gray-600 dark:text-gray-400">FILE</div>;
    }
  };

  const handleFileClick = (prefixedFilename) => {
    // 파일 뷰어 URL 생성
    const viewerUrl = `/api/file-viewer/${encodeURIComponent(prefixedFilename)}`;
    // 새 탭에서 열기
    window.open(viewerUrl, '_blank');
  };

  // 검색어로 필터링된 파일 목록
  const filteredFiles = searchQuery 
    ? files.filter(file => cleanFilename(file).toLowerCase().includes(searchQuery.toLowerCase()))
    : files;

  // 파일 전체 삭제 함수 추가
  const handleDeleteAllFiles = async () => {
    try {
      setIsDeleting(true);
      setShowDeleteAllConfirm(false); // 확인 모달 닫기
      
      // 서버에 전체 삭제 요청 - 상대 경로로 변경
      const response = await fetch(`/api/delete-all-files`, {
        method: 'DELETE',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        // 타임아웃 설정 추가
        signal: AbortSignal.timeout(30000) // 30초 타임아웃
      });
      
      // 응답 데이터 가져오기 (JSON이 아닐 경우 대비)
      let data;
      let responseText;
      
      try {
        responseText = await response.text(); // 먼저 텍스트로 가져옴
        data = JSON.parse(responseText); // 그 다음 JSON으로 파싱 시도
      } catch (jsonError) {
        throw new Error('서버 응답을 처리할 수 없습니다.');
      }
      
      if (response.ok && (data.status === 'success' || data.status === 'partial_success')) {
        // 파일 목록 강제 새로고침을 위해 상태 초기화
        hasLoadedRef.current = false;
        
        // 전체 파일 목록 비우기
        setFiles([]);
        
        // 성공 메시지 설정
        if (data.status === 'success') {
          setSuccessMessage(`모든 파일이 삭제되었습니다. (총 ${data.deleted_count || 0}개)`);
        } else {
          setSuccessMessage(`${data.deleted_count || 0}개 파일 삭제 성공, ${data.failed_count || 0}개 파일 삭제 실패`);
        }
        
        // 파일 목록 다시 로드
        await loadFiles();
        
        // 부모 컴포넌트에 삭제 완료 알림 (필요시)
        if (onUploadSuccess) {
          onUploadSuccess([]);
        }
      } else {
        throw new Error(data.message || '파일 전체 삭제 중 오류가 발생했습니다.');
      }
    } catch (err) {
      setDeleteError(`파일 전체 삭제에 실패했습니다: ${err.message || '알 수 없는 오류'}`);
      // 실패 시 파일 목록 다시 불러오기
      loadFiles();
    } finally {
      setIsDeleting(false);
    }
  };

  // 삭제 확인 모달
  const DeleteAllConfirmModal = () => {
    if (!showDeleteAllConfirm) return null;
    
    return (
      <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[70] p-4 animate-fade-in">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-5 sm:p-6 max-w-md w-full shadow-2xl animate-slide-up">
          <div className="flex items-start mb-4">
            <div className="flex-shrink-0 text-red-500">
              <FiAlertCircle size={24} />
            </div>
            <div className="ml-3">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                모든 파일 삭제 확인
              </h3>
              <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                <span className="font-medium text-gray-800 dark:text-gray-300">총 {files.length}개</span>의 모든 파일을 삭제하시겠습니까?
              </p>
              <p className="mt-1 text-xs text-red-500">
                이 작업은 되돌릴 수 없으며, 모든 파일과 인덱스 데이터가 영구적으로 삭제됩니다.
              </p>
            </div>
          </div>
          <div className="flex justify-end space-x-3 mt-5">
            <button
              onClick={() => setShowDeleteAllConfirm(false)}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-300 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-400"
              disabled={isDeleting}
            >
              취소
            </button>
            <button
              onClick={handleDeleteAllFiles}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50 transition flex items-center"
              disabled={isDeleting}
            >
              {isDeleting ? (
                <>
                  <FiLoader className="animate-spin mr-2" size={16} />
                  삭제 중...
                </>
              ) : (
                <>모두 삭제</>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      {/* 전체 삭제 확인 모달 */}
      <DeleteAllConfirmModal />
      
      <div 
        ref={modalRef}
        className="bg-white dark:bg-gray-800 rounded-2xl max-w-2xl w-full max-h-[80vh] shadow-2xl overflow-hidden animate-slide-up"
      >
        {/* 헤더 */}
        <div className="p-5 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-800 dark:text-white flex items-center">
            <FiFolder className="mr-2 text-indigo-500" size={20} /> 
            <span>인덱싱된 파일 목록</span>
          </h2>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <FiX className="text-gray-500 dark:text-gray-400" size={20} />
          </button>
        </div>
        
        {/* 검색 바와 전체 삭제 버튼 */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <div className="relative flex-1">
              <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
                <FiSearch className="text-gray-400" size={16} />
              </div>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="파일 검색..."
                className="w-full pl-10 pr-4 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400 text-gray-800 dark:text-gray-200"
              />
            </div>
          </div>
        </div>
        
        {/* 파일 목록 */}
        <div className="overflow-y-auto custom-scrollbar" style={{ maxHeight: 'calc(80vh - 150px)' }}>
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <FiLoader size={36} className="animate-spin text-indigo-500 mb-4" />
              <p className="text-gray-500 dark:text-gray-400">파일 목록을 불러오는 중...</p>
            </div>
          ) : error ? (
            <div className="p-8 text-center">
              <FiAlertCircle size={36} className="mx-auto text-red-500 mb-4" />
              <p className="text-red-500 font-medium">{error}</p>
                    <button 
                onClick={loadFiles}
                className="mt-4 px-4 py-2 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg transition-colors"
                    >
                다시 시도
                    </button>
            </div>
          ) : filteredFiles.length === 0 ? (
            <div className="p-8 text-center">
              {searchQuery ? (
                <p className="text-gray-500 dark:text-gray-400">검색 결과가 없습니다.</p>
              ) : (
                <>
                  <p className="text-gray-500 dark:text-gray-400 mb-2">인덱싱된 파일이 없습니다.</p>
                  <p className="text-gray-400 dark:text-gray-500 text-sm">파일을 업로드하여 대화에 활용해보세요.</p>
                </>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 divide-y divide-gray-200 dark:divide-gray-700">
              {filteredFiles.map((filename) => (
                <div 
                  key={filename}
                  className={`p-4 hover:bg-gray-50 dark:hover:bg-gray-750 transition-all ${
                    deleteAnimation === filename ? 'animate-shrink-fade-out' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div 
                      className="flex items-center flex-1 cursor-pointer"
                      onClick={() => handleFileClick(filename)}
                    >
                      {getFileIcon(filename)}
                      <div className="ml-3 flex-1 min-w-0">
<p className="text-sm font-medium text-gray-800 dark:text-white overflow-x-auto">
  {cleanFilename(filename)}
</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button 
                        onClick={() => handleFileClick(filename)}
                        className="p-2 text-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 rounded-full transition-colors"
                        title="새 탭에서 보기"
                      >
                        <FiExternalLink size={16} />
                      </button>
                      <button 
                        onClick={() => handleDeleteClick(filename)}
                        className="p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-full transition-colors"
                        title="파일 삭제"
                        disabled={isDeleting}
                      >
                        {isDeleting && deleteAnimation === filename ? (
                          <FiLoader size={16} className="animate-spin" />
                        ) : (
                          <FiTrash2 size={16} />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* 상태 메시지 */}
        {(successMessage || deleteError) && (
          <div className={`p-3 m-4 rounded-lg text-sm font-medium ${
            successMessage 
              ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' 
              : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
          }`}>
            {successMessage || deleteError}
        </div>
        )}
      </div>
    </div>
  );
};

// 새 대화 시작 모달 컴포넌트 추가
const NewChatModal = ({ isOpen, onClose, onStart }) => {
  const [topic, setTopic] = useState('');
  const [category, setCategory] = useState('메뉴얼');
  const modalRef = useRef(null);
  
  const categories = ['메뉴얼', '장애보고서', '기술문서', '기타'];
  
  // 모달 바깥 클릭 시 닫기
  useEffect(() => {
    function handleClickOutside(event) {
      if (modalRef.current && !modalRef.current.contains(event.target)) {
        onClose();
      }
    }
    
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => {
        document.removeEventListener("mousedown", handleClickOutside);
      };
    }
  }, [isOpen, onClose]);
  
  const handleSubmit = (e) => {
    e.preventDefault();
    onStart(topic, category);
    setTopic('');
    onClose();
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div
        ref={modalRef}
        className="bg-gray-800 rounded-xl w-full max-w-md overflow-hidden shadow-xl border border-gray-700/50 animate-slide-up"
      >
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <h2 className="text-lg font-medium text-gray-100 flex items-center">
            <FiMessageSquare className="mr-2 text-indigo-500" size={18} />
            새 대화 시작
          </h2>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-700 text-gray-400 hover:text-gray-300 transition-colors"
          >
            <FiX size={20} />
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">주제 (선택사항)</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="대화 주제를 입력하세요"
              className="w-full px-4 py-2.5 bg-gray-700 border border-gray-600 rounded-lg text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <p className="mt-1 text-xs text-gray-500">주제를 설정하면 더 명확한 컨텍스트로 대화를 시작할 수 있습니다.</p>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">카테고리</label>
            <div className="flex flex-wrap gap-2">
              {categories.map((cat) => (
                <button
                  key={cat}
                  type="button"
                  onClick={() => setCategory(cat)}
                  className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                    category === cat
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>
          
          <div className="pt-2 flex justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 mr-2"
            >
              취소
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
            >
              대화 시작
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

function ChatContainer({
  scrollLocked,
  activeConversationId,
  messages,
  searchTerm,
  filteredMessages,
  onUpdateMessages,
  isEmbedding,
  onUploadSuccess,
  onNewConversation,
  fileManagerOpen,
  setFileManagerOpen,
  sidebarOpen,
  setMode, // onToggleMode 대신 setMode를 받습니다
  currentMode, // 현재 모드 props 추가
  isStreaming,
  setIsStreaming,
  onStopGeneration // 응답 중단 함수 추가
}) {
  const [showSettings, setShowSettings] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [fileToDelete, setFileToDelete] = useState(null);
  const [fileDeleteLoading, setFileDeleteLoading] = useState(false);
  const [showDeleteAllModal, setShowDeleteAllModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [files, setFiles] = useState([]);
  const [filesLoading, setFilesLoading] = useState(false);
  const containerRef = useRef(null);
  const chatInputRef = useRef(null);
  const messagesEndRef = useRef(null); // 메시지 끝 참조 추가
  const dropZoneRef = useRef(null); // dropZone 참조 추가
  const [atBottom, setAtBottom] = useState(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [showScrollToTop, setShowScrollToTop] = useState(false); // 위로 스크롤 버튼 상태 추가
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [indexedFilesModalOpen, setIndexedFilesModalOpen] = useState(false);
  const [newChatModalOpen, setNewChatModalOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [selectedSourceIndex, setSelectedSourceIndex] = useState(null);
  const [selectedSourceContent, setSelectedSourceContent] = useState("");
  const [selectedSourceHighlights, setSelectedSourceHighlights] = useState([]);
  const prevMessageLengthRef = useRef(messages.length);
  const [filesToUpload, setFilesToUpload] = useState([]);
  
  // 응답 스트리밍 중지 컨트롤러
  const abortControllerRef = useRef(new AbortController());
  
  // 스크롤 관련 함수들 - 최상단에 정의
  // 스크롤을 하단으로 이동시키는 함수
  const scrollToBottom = useCallback(() => {
    if (!containerRef.current || scrollLocked) return;

    // 스크롤을 최하단으로 이동하기 위한 다양한 방법 조합
    
    // 1. 컨테이너의 scrollTop 직접 조작
    if (containerRef.current) {
      const scrollHeight = containerRef.current.scrollHeight;
      containerRef.current.scrollTop = scrollHeight;
    }
    
    // 2. messagesEndRef를 이용한 scrollIntoView
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "auto" });
    }
    
    // 3. 약간의 지연 후 한번 더 스크롤 시도
    setTimeout(() => {
      if (containerRef.current) {
        const scrollHeight = containerRef.current.scrollHeight;
        containerRef.current.scrollTop = scrollHeight;
      }
      
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "auto" });
      }
    }, 100);
  }, [scrollLocked]);
  
  // 스트리밍 중지 함수
  const stopResponseGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      console.log("ChatContainer: 응답 생성 중지");
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsStreaming(false);
      
      // App.jsx의 스트리밍 중단 처리도 함께 호출
      if (onStopGeneration) {
        console.log("ChatContainer: App의 중단 함수 호출");
        onStopGeneration();
      }
    }
  }, [onStopGeneration]);

  // 스타일 주입 및 초기 포커스 설정
  useEffect(() => {
    injectGlobalStyles();
    
    // 컴포넌트 마운트 시 입력 필드에 포커스
    console.log('ChatContainer: 컴포넌트 마운트 시 포커스 이벤트 발생');
    setTimeout(() => {
      if (chatInputRef.current) {
        chatInputRef.current.focus();
      }
      window.dispatchEvent(new CustomEvent('chatInputFocus'));
    }, 800);
  }, []);
  
  // 메시지 변경 시 스크롤 이벤트
  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);
  
  // 메시지 배열 변경 시 스크롤 강제 이동
  useEffect(() => {
    // 메시지가 변경되면 (새 메시지 추가, 메시지 로드 등) 스크롤 강제 이동
    if (!messages.length) return;
    
    // 즉시 실행
    scrollToBottom();
    
  }, [messages, scrollToBottom]);

  // activeConversationId가 변경될 때마다 자동 스크롤
  useEffect(() => {
    if (!activeConversationId) return;
    
    // 초기 로딩 시에만 스크롤 자동 이동 - 한 번만 실행
    if (containerRef.current) {
      const scrollHeight = containerRef.current.scrollHeight;
      containerRef.current.scrollTop = scrollHeight;
    }
    
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "auto" });
    }
    
    // 사용자 스크롤 감지 변수 추가
    let userHasScrolled = false;
    
    // 스크롤 이벤트 리스너 추가
    const handleScroll = () => {
      if (containerRef.current) {
        // 스크롤이 맨 아래가 아니면 사용자가 스크롤한 것으로 간주
        const isAtBottom = 
          containerRef.current.scrollHeight - containerRef.current.scrollTop <= 
          containerRef.current.clientHeight + 100; // 100px 여유 추가
        
        if (!isAtBottom) {
          userHasScrolled = true;
        }
      }
    };
    
    // 스크롤 이벤트 리스너 등록
    if (containerRef.current) {
      containerRef.current.addEventListener('scroll', handleScroll);
    }
    
    // 초기 타이밍에만 스크롤 자동 이동 (첫 500ms 동안만)
    const initialScrollTimers = [100, 300, 500];
    initialScrollTimers.forEach(delay => {
      setTimeout(() => {
        // 사용자가 스크롤하지 않은 경우에만 자동 스크롤 수행
        if (!userHasScrolled && containerRef.current) {
          const scrollHeight = containerRef.current.scrollHeight;
          containerRef.current.scrollTop = scrollHeight;
          
          if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: "auto" });
          }
        }
      }, delay);
    });
    
    // 이벤트 리스너 정리
    return () => {
      if (containerRef.current) {
        containerRef.current.removeEventListener('scroll', handleScroll);
      }
    };
  }, [activeConversationId]);

  // 스크롤 이벤트 감지 및 자동 스크롤 제어
  useEffect(() => {
    let userHasScrolled = false;
    let scrollTimer = null;
    
    // 스크롤 이벤트 핸들러
    const handleScroll = () => {
      if (containerRef.current) {
        // 스크롤이 맨 아래가 아니면 사용자가 스크롤한 것으로 간주
        const isAtBottom = 
          containerRef.current.scrollHeight - containerRef.current.scrollTop <= 
          containerRef.current.clientHeight + 100; // 100px 여유 추가
        
        if (!isAtBottom) {
          userHasScrolled = true;
          
          // 스크롤 버튼 표시 로직
          if (containerRef.current.scrollTop < containerRef.current.scrollHeight - containerRef.current.clientHeight - 200) {
            setShowScrollToTop(true);
          } else {
            setShowScrollToTop(false);
          }
        } else {
          // 맨 아래에 도달한 경우에만 스크롤 상태 초기화
          setShowScrollToTop(false);
          
          // 맨 아래에 도달해도 사용자 스크롤 상태는 유지 (강제 스크롤 방지)
          // userHasScrolled = false; - 이 부분 제거
        }
      }
    };
    
    // 커스텀 스크롤 이벤트 핸들러
    const handleChatScrollToBottom = (e) => {
      // 대화 전환 시에만 스크롤 이동 허용 (이벤트에 forceSroll 필드가 있는 경우)
      if (e && e.detail && e.detail.forceScroll) {
        scrollToBottom();
        return;
      }
      
      // 사용자가 수동으로 스크롤 중이면 자동 스크롤 방지
      if (userHasScrolled) {
        console.log('사용자가 스크롤 중이므로 자동 스크롤 방지');
        return;
      }
      
      scrollToBottom();
    };
    
    // 사용자 스크롤 감지를 위한 추가 이벤트 리스너
    const handleUserScroll = () => {
      userHasScrolled = true;
      
      // 디바운스로 로그 출력 (디버깅용)
      if (scrollTimer) clearTimeout(scrollTimer);
      scrollTimer = setTimeout(() => {
        console.log('사용자 스크롤 감지됨');
      }, 100);
    };
    
    // 이벤트 리스너 등록
    if (containerRef.current) {
      containerRef.current.addEventListener('scroll', handleScroll);
      containerRef.current.addEventListener('wheel', handleUserScroll);
      containerRef.current.addEventListener('touchmove', handleUserScroll);
    }
    
    window.addEventListener('chatScrollToBottom', handleChatScrollToBottom);
    
    // 스크롤 상태 리셋 함수 (대화 전환 시 사용)
    const resetScrollState = () => {
      userHasScrolled = false;
    };
    
    // 대화 ID 변경 시 스크롤 상태 리셋
    resetScrollState();
    
    // 이벤트 리스너 정리
    return () => {
      if (containerRef.current) {
        containerRef.current.removeEventListener('scroll', handleScroll);
        containerRef.current.removeEventListener('wheel', handleUserScroll);
        containerRef.current.removeEventListener('touchmove', handleUserScroll);
      }
      window.removeEventListener('chatScrollToBottom', handleChatScrollToBottom);
      if (scrollTimer) clearTimeout(scrollTimer);
    };
  }, [scrollToBottom, activeConversationId]);
  
  // 스크롤을 맨 위로 이동하는 함수 추가
  const scrollToTop = () => { 
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: 0,
        behavior: "smooth"
      });
    }
  };
  
  // 빈 채팅 화면 워터마크 컴포넌트
  const EmptyChatWatermark = () => {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <div className="flex flex-col items-center justify-center max-w-2xl px-6 py-8 text-center">
          {/* 물결 애니메이션 배경 효과 */}
          <div className="absolute inset-0 overflow-hidden opacity-10">
            <div className="absolute inset-x-0 top-1/4 w-full h-64 bg-gradient-to-r from-indigo-600/30 to-purple-600/30 rounded-full filter blur-3xl transform -translate-y-20 scale-150 animate-pulse" style={{ animationDuration: '8s' }}></div>
            <div className="absolute inset-x-0 top-1/3 w-full h-64 bg-gradient-to-r from-blue-600/30 to-cyan-600/30 rounded-full filter blur-3xl transform translate-y-16 scale-125 animate-pulse" style={{ animationDuration: '10s', animationDelay: '1s' }}></div>
          </div>
          
          {/* 아이콘 컨테이너 - 문서 아이콘으로 변경 */}
          <div className="relative mb-8">
            <div className="absolute -inset-0 rounded-full bg-gradient-to-r from-blue-500/20 to-indigo-600/20 animate-ping opacity-30" style={{ animationDuration: '3s' }}></div>
            <div className="absolute -inset-4 rounded-full bg-gradient-to-r from-blue-500/10 to-indigo-600/10 animate-ping opacity-20" style={{ animationDuration: '3.5s' }}></div>
            <div className="absolute -inset-8 rounded-full bg-gradient-to-r from-blue-500/5 to-indigo-600/5 animate-ping opacity-10" style={{ animationDuration: '4s' }}></div>
            
            <div className="relative w-24 h-24 bg-gradient-to-br from-blue-500/20 to-indigo-600/20 rounded-full flex items-center justify-center backdrop-blur-sm">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-500/30 to-indigo-600/30 rounded-full flex items-center justify-center">
                <div className="w-16 h-16 bg-gray-900/80 rounded-full flex items-center justify-center ring-2 ring-blue-500/30">
                  <FiFileText className="text-blue-400 animate-pulse" style={{ animationDuration: '2s' }} size={30} />
                </div>
              </div>
            </div>
          </div>
          
          {/* 텍스트 영역 - 부드러운 페이드인 애니메이션 */}
          <div className="space-y-4 animate-fade-in-up">
            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-200 to-indigo-300 mb-2">
              문서 기반 질의응답 챗봇
            </h2>
            <p className="text-base text-gray-400 mb-6 max-w-lg leading-relaxed">
              업로드한 문서에 관련된 질문을 해보세요. <br />정확한 정보와 함께 답변해 드립니다.
            </p>
          </div>
          
          {/* 기능 설명 카드 영역 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-md text-left mt-4 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
            <div className="flex items-start p-3 rounded-lg bg-gray-800/40 border border-gray-700/50 backdrop-blur-sm hover:bg-gray-800/60 hover:border-blue-500/30 transition-all duration-300 transform hover:-translate-y-1">
              <div className="flex-shrink-0 bg-gradient-to-br from-blue-500/20 to-indigo-600/20 p-2 rounded-md mr-3">
                <FiSearch className="text-blue-400" size={18} />
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-200">정보 검색</h3>
                <p className="text-xs text-gray-500 mt-1">문서 내 관련 정보를 검색해 답변합니다</p>
              </div>
            </div>
            
            <div className="flex items-start p-3 rounded-lg bg-gray-800/40 border border-gray-700/50 backdrop-blur-sm hover:bg-gray-800/60 hover:border-blue-500/30 transition-all duration-300 transform hover:-translate-y-1">
              <div className="flex-shrink-0 bg-gradient-to-br from-blue-500/20 to-indigo-600/20 p-2 rounded-md mr-3">
                <FiFileText className="text-blue-400" size={18} />
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-200">문서 요약</h3>
                <p className="text-xs text-gray-500 mt-1">주요 내용을 간결하게 요약해 드립니다</p>
              </div>
            </div>
            
            <div className="flex items-start p-3 rounded-lg bg-gray-800/40 border border-gray-700/50 backdrop-blur-sm hover:bg-gray-800/60 hover:border-blue-500/30 transition-all duration-300 transform hover:-translate-y-1">
              <div className="flex-shrink-0 bg-gradient-to-br from-blue-500/20 to-indigo-600/20 p-2 rounded-md mr-3">
                <FiMessageCircle className="text-blue-400" size={18} />
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-200">질의응답</h3>
                <p className="text-xs text-gray-500 mt-1">문서 기반 질문에 정확히 답변합니다</p>
              </div>
            </div>
            
            <div className="flex items-start p-3 rounded-lg bg-gray-800/40 border border-gray-700/50 backdrop-blur-sm hover:bg-gray-800/60 hover:border-blue-500/30 transition-all duration-300 transform hover:-translate-y-1">
              <div className="flex-shrink-0 bg-gradient-to-br from-blue-500/20 to-indigo-600/20 p-2 rounded-md mr-3">
                <FiBookmark className="text-blue-400" size={18} />
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-200">출처 제공</h3>
                <p className="text-xs text-gray-500 mt-1">답변의 정확한 출처를 함께 제공합니다</p>
              </div>
            </div>
          </div>
          
          {/* 시작 도움말 - 하단 안내 문구 */}
          <div className="mt-10 flex items-center text-gray-500 text-sm animate-fade-in-up" style={{ animationDelay: '400ms' }}>
            <FiCornerDownRight className="mr-2 text-blue-400" size={16} />
            <span>입력창에 질문을 입력하면 워터마크가 사라지고 대화가 시작됩니다</span>
          </div>
        </div>
      </div>
    );
  };
  
  // 메시지 렌더링 함수 추가
  const renderMessages = () => {
    // 필터된 메시지 또는 전체 메시지를 기준으로 렌더링
    const messagesToRender = searchTerm ? filteredMessages : messages;
    
    // 메시지 목록과 워터마크를 함께 표시하는 방식으로 변경
    // 사용자 메시지가 없는 경우에만 워터마크 표시
    const hasUserMessage = messages.some(msg => msg.role === 'user');
    
    // 메시지가 없거나, 유일한 메시지가 "안녕하세요! 무엇을 도와드릴까요?"인 경우 워터마크 표시
    const showWatermark = !hasUserMessage;
    
    return (
      <>
        {/* 메시지 목록 출력 */}
        {messagesToRender.map((message, index) => {
          const prevMessage = index > 0 ? messagesToRender[index - 1] : null;
          const nextMessage = index < messagesToRender.length - 1 ? messagesToRender[index + 1] : null;
          
          // 고유한 key 생성 (timestamp + index 조합)
          const messageKey = `${message.timestamp || Date.now()}-${index}`;
          
          return (
            <ChatMessage
              key={messageKey}
              message={message}
              searchTerm={searchTerm}
              isSearchMode={!!searchTerm}
              prevMessage={prevMessage}
              nextMessage={nextMessage}
              onAskFollowUp={handleAskFollowUp}
            />
          );
        })}
        
        {/* 사용자 메시지가 없을 때만 워터마크 표시 */}
        {showWatermark && <EmptyChatWatermark />}
      </>
    );
  };
  
  // 파일 관리 버튼 클릭 핸들러
  const handleFileManager = () => {
    if (setFileManagerOpen) {
      setFileManagerOpen(true);
    }
  };
  
  // 후속 질문 핸들러
  const handleAskFollowUp = (question) => {
    if (!question || !chatInputRef.current) return;
    
    // 채팅 입력창에 질문 설정 후 자동 포커스
    chatInputRef.current.clear();
    setTimeout(() => {
      if (chatInputRef.current) {
        chatInputRef.current.setMessage(question);
        chatInputRef.current.focus();
      }
    }, 50);
  };
  
  // 새 대화 시작 핸들러 - 모달 없이 바로 새 대화 생성으로 수정
  const handleStartNewChat = () => {
    if (onNewConversation) {
      // 모달 대신 바로 기본 제목("새 대화" 또는 "대화 N")으로 대화 생성
      onNewConversation(null, "메뉴얼");
      
      // 새 대화 생성 후 입력 필드에 포커스 (추가 지연 적용)
      console.log('ChatContainer: 새 대화 시작 후 포커스 이벤트 발생');
      setTimeout(() => {
        if (chatInputRef.current) {
          chatInputRef.current.focus();
        }
        window.dispatchEvent(new CustomEvent('chatInputFocus'));
      }, 800);
    }
  };

  // 모드 전환 토글 컴포넌트 추가
  const ModeToggleSwitch = () => {
    // 사용자 상호작용 시 토글 상태 변경
    const handleToggleMode = (newMode) => (e) => {
      // 이벤트 버블링 방지
      e.preventDefault();
      e.stopPropagation();
      
      // 같은 모드를 다시 클릭한 경우 무시
      if (currentMode === newMode) return;
      
      // setMode 함수 호출하여 모드 전환 (App.jsx에서 전달된 props)
      console.log('모드 전환 요청:', newMode, '현재 모드:', currentMode);
      if (typeof setMode === 'function') {
        setMode(newMode);
      } else {
        console.error('setMode 함수가 없거나 올바르지 않습니다.');
      }
    };

    // props로 전달받은 현재 모드 사용 (fallback으로 window.currentAppMode도 확인)
    const mode = currentMode || (typeof window !== 'undefined' && window.currentAppMode) || 'chat';

    return (
      <div className="fixed right-6 top-1/2 transform -translate-y-1/2 z-20">
        <div className="bg-gray-800/90 backdrop-blur-md rounded-full p-2 shadow-lg border border-gray-700/50 flex flex-col gap-3">
          {/* 배경 효과 - 활성화된 모드에 따라 움직임 */}
          <div className="absolute inset-x-1.5 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500/20 to-indigo-600/20 filter blur-sm transition-all duration-300 ease-in-out pointer-events-none" 
               style={{ 
                 top: mode === 'chat' ? '0.4rem' : mode === 'sql' ? '2.9rem' : '5.4rem',
                 opacity: 0.7
               }}>
          </div>
          
          {/* 챗봇 모드 버튼 */}
          <button 
            onClick={handleToggleMode('chat')}
            className={`relative transition-all duration-300 w-8 h-8 rounded-full flex items-center justify-center ${
              mode === 'chat' 
                ? 'bg-gradient-to-br from-blue-500/80 to-indigo-600/80 text-white shadow-md shadow-blue-500/20' 
                : 'bg-gray-800/80 text-gray-400 hover:text-gray-200 hover:bg-gray-700/60'
            }`}
            title="챗봇 모드로 전환"
            data-testid="chat-mode-toggle"
          >
            {/* 활성화 효과 - 고리 애니메이션 */}
            {mode === 'chat' && (
              <>
                <div className="absolute inset-0 rounded-full border border-blue-400/30 animate-ping opacity-30"></div>
                <div className="absolute inset-0 rounded-full bg-gradient-to-br from-blue-500/5 to-indigo-600/5 animate-pulse"></div>
              </>
            )}
            
            <FiMessageCircle size={14} className="transition-all duration-300" />
          </button>
          
          {/* SQL 모드 버튼 */}
          <button 
            onClick={handleToggleMode('sql')}
            className={`relative transition-all duration-300 w-8 h-8 rounded-full flex items-center justify-center ${
              mode === 'sql' 
                ? 'bg-gradient-to-br from-indigo-500/80 to-purple-600/80 text-white shadow-md shadow-indigo-500/20' 
                : 'bg-gray-800/80 text-gray-400 hover:text-gray-200 hover:bg-gray-700/60'
            }`}
            title="SQL 질의 모드로 전환"
            data-testid="sql-mode-toggle"
          >
            {/* 활성화 효과 - 고리 애니메이션 */}
            {mode === 'sql' && (
              <>
                <div className="absolute inset-0 rounded-full border border-indigo-400/30 animate-ping opacity-30"></div>
                <div className="absolute inset-0 rounded-full bg-gradient-to-br from-indigo-500/5 to-purple-600/5 animate-pulse"></div>
              </>
            )}
            
            <FiDatabase size={14} className="transition-all duration-300" />
          </button>
          
          {/* 대시보드 모드 버튼 */}
          <button 
            onClick={handleToggleMode('dashboard')}
            className={`relative transition-all duration-300 w-8 h-8 rounded-full flex items-center justify-center ${
              mode === 'dashboard' 
                ? 'bg-gradient-to-br from-emerald-500/80 to-teal-600/80 text-white shadow-md shadow-emerald-500/20' 
                : 'bg-gray-800/80 text-gray-400 hover:text-gray-200 hover:bg-gray-700/60'
            }`}
            title="대시보드 모드로 전환"
            data-testid="dashboard-mode-toggle"
          >
            {/* 활성화 효과 - 고리 애니메이션 */}
            {mode === 'dashboard' && (
              <>
                <div className="absolute inset-0 rounded-full border border-emerald-400/30 animate-ping opacity-30"></div>
                <div className="absolute inset-0 rounded-full bg-gradient-to-br from-emerald-500/5 to-teal-600/5 animate-pulse"></div>
              </>
            )}
            
            <FiBarChart2 size={14} className="transition-all duration-300" />
          </button>
        </div>
      </div>
    );
  };

  const handleStopResponse = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = new AbortController();
      setIsStreaming(false);
    }
  };

  // 메시지 전송 핸들러 추가
  const handleSubmit = async (input, selectedCategory) => {
    if (!input.trim()) return;

    // 사용자 메시지 추가
    const userMessage = {
      role: "user",
      content: input,
      timestamp: new Date().getTime(),
    };
    
    const updatedMessages = [...messages, userMessage];
    onUpdateMessages(updatedMessages);
    
    // 로딩 상태 활성화 - 약간의 지연 추가 (UI가 부드럽게 전환되도록)
    setTimeout(() => {
      setLoading(true);
    }, 100);
    setError(null);
    
    try {
      // 이전 스트리밍 중인 경우 중단
      if (abortControllerRef.current) {
        console.log("ChatContainer: 이전 요청 중단");
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      
      // 새 AbortController 생성
      abortControllerRef.current = new AbortController();
      console.log("ChatContainer: 새 AbortController 생성");
      
      // history 배열 처리 - role과 content만 포함
      const history = messages.slice(-10).map(m => ({
        role: m.role,
        content: m.content
      }));
      
      // category가 문자열인지 확인하고 기본값 설정
      const categoryValue = typeof selectedCategory === 'string' && selectedCategory ? selectedCategory : "메뉴얼";
      
      // 응답 메시지 객체 미리 생성 (스트리밍용)
      const botMessage = {
        role: "assistant",
        content: "",
        sources: [],
        cited_sources: [],
        timestamp: new Date().getTime(),
      };
      
      // 빈 봇 메시지 추가 (스트리밍 시작 전)
      onUpdateMessages([...updatedMessages, botMessage]);
      
      // 스트리밍 시작 상태로 설정
      setIsStreaming(true);
      
      // 스트리밍 응답 가져오기
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: input,
          history: history,
          category: categoryValue,
        }),
        signal: abortControllerRef.current.signal, // AbortController 신호 연결
      });
      
      if (!response.ok) {
        throw new Error(`서버 응답 오류: ${response.status}`);
      }
      
      // 스트리밍 응답 처리
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // 스트리밍 시작 시 로딩 상태 비활성화 (중요: 스트리밍이 시작되면 로딩 인디케이터 제거)
      let isFirstChunk = true;
      let accumulatedContent = "";
      let sources = [];
      let cited_sources = [];
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        // 청크 디코딩
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        
        for (const line of lines) {
          if (line.trim() === "") continue;
          
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              
              // 첫 번째 청크가 도착하면 로딩 상태 비활성화
              if (isFirstChunk) {
                setLoading(false);
                isFirstChunk = false;
              }
              
              // 토큰 처리
              if (data.token) {
                accumulatedContent += data.token;
                
                // 메시지 업데이트
                const updatedBotMessage = {
                  role: "assistant",
                  content: accumulatedContent,
                  sources: sources,
                  cited_sources: cited_sources,
                  timestamp: new Date().getTime(),
                };
                
                onUpdateMessages([...updatedMessages, updatedBotMessage]);
              }
              
              // 출처 정보 처리
              if (data.sources) {
                sources = data.sources;
                
                // 메시지 업데이트 (출처 포함)
                const updatedBotMessage = {
                  role: "assistant",
                  content: accumulatedContent,
                  sources: sources,
                  cited_sources: cited_sources,
                  timestamp: new Date().getTime(),
                };
                
                onUpdateMessages([...updatedMessages, updatedBotMessage]);
              }
              
              // 인용된 출처 정보 처리
              if (data.cited_sources) {
                cited_sources = data.cited_sources;
                
                // 메시지 업데이트 (인용된 출처 포함)
                const updatedBotMessage = {
                  role: "assistant",
                  content: accumulatedContent,
                  sources: sources,
                  cited_sources: cited_sources,
                  timestamp: new Date().getTime(),
                };
                
                onUpdateMessages([...updatedMessages, updatedBotMessage]);
              }
              
              // eos 이벤트 처리 (스트리밍 종료)
              if (data.event === 'eos') {
                // 스트리밍 상태 비활성화
                setIsStreaming(false);
                abortControllerRef.current = null;
              }
            } catch (e) {
              console.error("스트리밍 데이터 파싱 오류:", e);
            }
          }
        }
      }
      
      // 스트리밍 완료 후 최종 메시지 업데이트
      const finalBotMessage = {
        role: "assistant",
        content: accumulatedContent,
        sources: sources,
        cited_sources: cited_sources,
        timestamp: new Date().getTime(),
      };
      
      onUpdateMessages([...updatedMessages, finalBotMessage]);
      
      // 스트리밍 완료 후 포커스 설정
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('chatInputFocus'));
      }, 100);
      
    } catch (error) {
      // AbortError는 사용자가 의도적으로 중단한 경우이므로 일반 오류로 처리하지 않음
      if (error.name === 'AbortError') {
        // 사용자가 응답 생성을 중단했습니다.
        console.log("ChatContainer: 사용자에 의해 응답이 중단되었습니다.");
        // App.jsx의 handleStopGeneration에서 메시지 처리를 담당합니다.
      } else {
        console.error("채팅 요청 오류:", error);
        setError(`서버 응답을 처리할 수 없습니다: ${error.message}`);
      }
    } finally {
      // 로딩 상태 비활성화 (오류 발생 시나 모든 처리 완료 후)
      setLoading(false);
      // setIsStreaming(false); // App.jsx에서 관리하므로 주석 처리 또는 삭제
      // abortControllerRef.current = null; // App.jsx에서 관리하므로 주석 처리 또는 삭제
      console.log("ChatContainer: handleSubmit finally - 로딩 상태 비활성화");
      
      // 응답 완료 후 포커스 설정
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('chatInputFocus'));
      }, 100);
    }
  };

  return (
    <div
      id="chat-container"
      className={`flex flex-col h-full overflow-hidden relative ${
        sidebarOpen ? "ml-0 md:ml-[var(--sidebar-width)]" : "ml-0"
      } transition-all duration-300 ease-in-out`}
    >
      {/* 로딩 인디케이터 추가 */}
      <LoadingIndicator active={loading} />
      
      {/* 파일 드래그 오버레이 */}
      {isDragging && (
        <div className="absolute inset-0 bg-indigo-900/50 backdrop-blur-sm z-50 flex items-center justify-center border-2 border-dashed border-indigo-400 animate-pulse">
          <div className="bg-gray-800/80 backdrop-blur-md p-8 rounded-2xl text-center shadow-2xl">
            <div className="w-20 h-20 bg-indigo-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
              <FiUploadCloud size={40} className="text-indigo-400" />
            </div>
            <h3 className="text-xl font-bold text-white mb-1">파일을 이곳에 놓으세요</h3>
            <p className="text-gray-300">파일을 업로드하여 대화에 활용하세요</p>
          </div>
        </div>
      )}
      
      {/* 헤더 - 스타일 업데이트 */}
      <div className="h-16 flex items-center justify-between px-6 bg-gray-900 border-b border-gray-800 shadow-sm z-10">
        <div className={`flex items-center ${!sidebarOpen ? 'ml-12' : ''} transition-all duration-300`}>
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 mr-4 flex items-center justify-center shadow-md">
            <FiMessageSquare size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
              지식 QnA 어시스턴트
            </h1>
            <p className="text-xs text-gray-400 mt-0.5">AI 기반 업무 지능화 솔루션</p>
          </div>
        </div>

        {/* 헤더 버튼 그룹 */}
        <div className="flex gap-2 items-center">
          {/* 파일 관리 버튼 */}
          <button
            onClick={handleFileManager}
            className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
              fileManagerOpen ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
            title="인덱싱된 파일 관리"
            disabled={isEmbedding}
          >
            <FiHardDrive size={16} />
            <span>파일</span>
          </button>
        </div>
      </div>

      {/* 메시지 목록 */}
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto overflow-x-hidden bg-gray-900 px-4 pt-2 pb-4 custom-scrollbar"
      >
        <div className="flex flex-col space-y-2 max-w-4xl mx-auto">
          {/* 메시지가 없을 때 안내 메시지 */}
          {renderMessages().length === 0 && (
            <div className="flex flex-col items-center justify-center h-full py-10 text-center">
              <div className="w-16 h-16 rounded-full bg-indigo-900/30 flex items-center justify-center mb-4">
                <FiMessageSquare size={28} className="text-indigo-400" />
              </div>
              <h3 className="text-xl font-medium text-gray-300 mb-2">대화를 시작해보세요</h3>
              <p className="text-gray-500 max-w-md">
                질문을 입력하시면 인덱싱된 문서 기반으로 답변해 드립니다.
              </p>
            </div>
          )}
          
          {/* 메시지 목록 */}
          {renderMessages()}
          <div ref={messagesEndRef} />
        </div>

        {/* 스크롤 위로 이동 버튼 */}
        {showScrollToTop && (
          <button
            className="absolute bottom-24 right-6 p-3 bg-indigo-600 text-white rounded-full shadow-lg hover:bg-indigo-700 hover:shadow-glow-sm transition-all transform hover:scale-105 active:scale-95"
            onClick={scrollToTop}
          >
            <FiArrowUp size={20} />
          </button>
        )}
      </div>

      {/* 모드 전환 스위치 추가 */}
      <ModeToggleSwitch />

      {/* 입력 영역 - SQL 질의 화면과 스타일 통일 */}
      <div className="px-4 py-3 border-t border-gray-800/50 bg-gray-900/70 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto">
          <ChatInput
            ref={chatInputRef}
            onSend={handleSubmit}
            disabled={loading || isEmbedding}
            onTyping={(isTyping) => {}}
            onUploadSuccess={onUploadSuccess}
            isEmbedding={isEmbedding}
            isStreaming={isStreaming}
            onStopGeneration={stopResponseGeneration}
          />
          
          {/* 로딩 상태 표시 */}
          {loading && (
            <div className="absolute top-0 left-0 w-full h-1 overflow-hidden bg-gray-800">
              <div className="h-full bg-gradient-to-r from-indigo-500 via-blue-500 to-indigo-500 animate-shine bg-[length:200%_100%]" />
            </div>
          )}
          
          {/* 에러 표시 */}
          {error && (
            <div className="mt-1 text-red-500 text-sm animate-fade-in">
              <div className="flex items-center bg-red-950/30 p-2 rounded-lg">
                <FiAlertCircle className="mr-2 flex-shrink-0" size={14} />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 인덱싱된 파일 관리 모달 */}
      {fileManagerOpen && (
        <IndexedFilesModal 
          isOpen={fileManagerOpen}
          onClose={() => setFileManagerOpen(false)}
        />
      )}
    </div>
  );
}

export default ChatContainer;
