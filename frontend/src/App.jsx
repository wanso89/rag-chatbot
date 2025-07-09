import React, { useRef, useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatContainer from "./components/ChatContainer";
import SQLQueryPage from "./components/SQLQueryPage";
import Dashboard from "./components/dashboard/Dashboard";
import ImageModal from "./components/ImageModal";
import {
  FiChevronLeft,
  FiChevronRight,
  FiMessageSquare,
  FiPlus,
  FiX,
  FiLoader,
  FiCheckCircle,
  FiFile,
  FiMenu,
  FiAlignLeft,
  FiDatabase,
  FiAlertCircle,
  FiBarChart2
} from "react-icons/fi";

// API 기본 URL 상수 정의
const API_BASE_URL = 'http://172.10.2.70:8000';

const SIDEBAR_WIDTH = 320;
const SIDEBAR_MIN = 280;
const SIDEBAR_MAX = 450;

// 임베딩 알림 오버레이 컴포넌트 추가
const EmbeddingOverlay = ({ isActive, status, files }) => {
  if (!isActive && !status) return null;
  
  const isCompleted = status === '완료';
  
  return (
    <div className="fixed inset-0 bg-gray-900/90 backdrop-blur-sm flex flex-col items-center justify-center z-[100] animate-fade-in">
      <div className="max-w-lg w-full px-6 py-8 rounded-2xl bg-gray-800/70 backdrop-blur-md text-center space-y-6">
        <div className="relative mx-auto w-24 h-24">
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500/20"></div>
          
          {!isCompleted ? (
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin"></div>
          ) : (
            <div className="absolute inset-0 rounded-full border-4 border-green-500 flex items-center justify-center animate-pulse">
              <FiCheckCircle className="text-green-400" size={36} />
            </div>
          )}
          
          <div className="absolute inset-4 rounded-full bg-indigo-500/20 flex items-center justify-center">
            <FiFile className={isCompleted ? "text-green-200" : "text-indigo-200"} size={24} />
          </div>
        </div>
        
        <div className="space-y-2">
          <h3 className="text-xl font-bold text-white">
            {isCompleted ? "파일 임베딩 완료!" : (status || "파일 임베딩 처리 중...")}
          </h3>
          <p className="text-gray-300 text-sm">
            {isCompleted ? (
              <>파일이 성공적으로 임베딩되었습니다.<br/>이제 챗봇과의 대화에 활용할 수 있습니다.</>
            ) : (
              <>이 과정은 파일 크기와 내용에 따라 몇 분 정도 소요될 수 있습니다.<br/>
              임베딩이 완료될 때까지 기다려주세요.</>
            )}
          </p>
        </div>
        
        {files && files.length > 0 && !isCompleted && (
          <div className="bg-gray-900/50 rounded-xl p-4 max-h-40 overflow-y-auto">
            <p className="text-gray-400 text-xs mb-2">{files.length}개 파일 처리 중:</p>
            <div className="space-y-1.5">
              {files.map((file, index) => (
                <div key={index} className="flex items-center">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></div>
                  <span className="text-gray-200 text-sm truncate">{file.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {isCompleted && (
          <button 
            className="mt-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            onClick={() => { window.dispatchEvent(new CustomEvent('refreshIndexedFiles')); }}
          >
            파일 목록 보기
          </button>
        )}
      </div>
    </div>
  );
};

function App() {
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_WIDTH);
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768); // md: 이상에서 기본적으로 열림
  const [userName, setUserName] = useState("사용자");
  const [scrollLocked, setScrollLocked] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const isResizing = useRef(false);
  const [userId, setUserId] = useState("user1"); // 임시 사용자 ID, 실제로는 인증 기반 ID 사용
  const [theme, setTheme] = useState(() => {
    // 항상 다크모드를 기본값으로 설정
    localStorage.setItem("theme", "dark");
    return "dark";
  });
  const [defaultCategory, setDefaultCategory] = useState("메뉴얼"); // 기본 카테고리 상태
  const [saveError, setSaveError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentMessages, setCurrentMessages] = useState([
    {
      role: "assistant",
      content: "안녕하세요! 무엇을 도와드릴까요?",
      sources: [],
      timestamp: new Date().getTime(),
    },
  ]);
  const [filteredMessages, setFilteredMessages] = useState(currentMessages);
  // 임베딩 상태 관련 변수 추가
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [embeddingStatus, setEmbeddingStatus] = useState(null);
  const [embeddedFiles, setEmbeddedFiles] = useState([]);
  
  // SQL 쿼리 페이지 표시 여부 상태 추가
  const [showSQLPage, setShowSQLPage] = useState(false);
  // 모드 상태 추가 (chat, sql 또는 dashboard)
  const [mode, setMode] = useState('chat');
  // 파일 매니저 상태 추가
  const [fileManagerOpen, setFileManagerOpen] = useState(false);
  
  // 이미지 모달 상태 추가
  const [imageModalOpen, setImageModalOpen] = useState(false);
  const [selectedImageSource, setSelectedImageSource] = useState(null);

  // 대시보드 관련 상태
  const [dashboardStats, setDashboardStats] = useState({
    totalQueries: 0,
    totalChats: 0,
    activeUsers: 0,
    averageQueriesPerDay: 0,
    queryCountByDate: []
  });
  
  // SQL 모드 관련 상태
  const [recentQueries, setRecentQueries] = useState([]);
  const [dbSchema, setDbSchema] = useState({});

  // 응답 스트리밍 상태 추가
  const [isStreaming, setIsStreaming] = useState(false);
  // 응답 중단을 위한 AbortController 레퍼런스 추가
  const abortControllerRef = useRef(null);

  console.log('===============================================');
  console.log('앱 컴포넌트 초기화');
  console.log('showSQLPage 초기값:', showSQLPage);
  console.log('mode 초기값:', mode);
  
  // 브라우저 콘솔에서 직접 상태를 변경할 수 있는 디버깅 함수 추가
  useEffect(() => {
    // 전역 함수로 등록
    window.setAppMode = (newMode) => {
      if (newMode === 'sql' || newMode === 'chat' || newMode === 'dashboard') {
        setMode(newMode);
        return true;
      }
      return false;
    };

    window.toggleAppMode = () => {
      const modes = ['chat', 'sql', 'dashboard'];
      const currentIndex = modes.indexOf(mode);
      const nextIndex = (currentIndex + 1) % modes.length;
      const newMode = modes[nextIndex];
      
      setMode(newMode);
      return newMode;
    };
    
    window.debugAppState = () => {
      // 디버깅 함수는 유지
    };
    
    return () => {
      // 정리 함수
      delete window.setAppMode;
      delete window.toggleAppMode;
      delete window.debugAppState;
    };
  }, [mode]);
  
  // mode 상태가 변경될 때 toast 메시지 표시
  useEffect(() => {
    console.log('모드 변경됨:', mode);
    // 전역 변수에 현재 모드 저장 (ModeToggleSwitch 컴포넌트에서 참조)
    if (typeof window !== 'undefined') {
      window.currentAppMode = mode;
    }
    
    // 모드 변경 시 토스트 메시지 표시
    if (mode === 'sql') {
      showModeChangeToast('SQL 질의 모드로 전환했습니다');
    } else if (mode === 'chat') {
      showModeChangeToast('챗봇 모드로 전환했습니다');
    } else if (mode === 'dashboard') {
      showModeChangeToast('대시보드 모드로 전환했습니다');
    }
  }, [mode]);

  // showSQLPage 상태 변경 감지 useEffect 추가
  useEffect(() => {
    console.log('showSQLPage 상태 변경됨:', showSQLPage);
  }, [showSQLPage]);

  // 테마 변경 함수
  const toggleTheme = useCallback(() => {
    // 다크모드만 유지하도록 수정
    setTheme("dark");
    localStorage.setItem("theme", "dark");
    document.documentElement.classList.add("dark");
  }, []);
  
  // 테마 적용
  useEffect(() => {
    // 항상 다크모드 적용
    document.documentElement.classList.add("dark");
  }, [theme]);

  // 초기 대화 목록 로드 (로컬 스토리지 또는 백엔드)
  useEffect(() => {
    console.log("초기 대화 목록 로드 시작...");
    
    const savedConversations = localStorage.getItem("conversations");
    const savedActiveId = localStorage.getItem("activeConversationId");
    
    console.log("로컬 스토리지 데이터:", { 
      savedConversations: savedConversations ? "있음" : "없음", 
      savedActiveId: savedActiveId ? savedActiveId : "없음" 
    });
    
    let initialConvs = [];
    
    if (savedConversations) {
      try {
        // 저장된 대화 파싱 시도
        initialConvs = JSON.parse(savedConversations);
        
        // 데이터 구조 유효성 검증 및 보완
        if (Array.isArray(initialConvs)) {
          console.log(`로컬 스토리지에서 ${initialConvs.length}개의 대화 불러옴`);
          
          // 각 대화 객체 구조 보강
          initialConvs = initialConvs.map((conv) => {
            // 메시지가 없거나 빈 배열인 경우 초기 메시지 추가
            const messages = Array.isArray(conv.messages) && conv.messages.length > 0 
              ? conv.messages.map(msg => ({
                  role: msg.role || "assistant",
                  content: msg.content || "",
                  timestamp: msg.timestamp || Date.now(),
                  sources: Array.isArray(msg.sources) ? msg.sources : [],
                }))
              : [{
                  role: "assistant",
                  content: "안녕하세요! 무엇을 도와드릴까요?",
                  sources: [],
                  timestamp: Date.now(),
                }];
            
            // 대화 객체 구조 보강
            return {
              id: conv.id || Date.now().toString(),
              title: conv.title || "제목 없음",
              timestamp: conv.timestamp || Date.now(),
              messages: messages,
              pinned: !!conv.pinned,
              category: conv.category || defaultCategory,
              metadata: conv.metadata || {
                messageCount: messages.length,
                firstMessageTimestamp: Date.now(),
                lastActivity: Date.now()
              }
            };
          });
        } else {
          console.warn("로컬 스토리지의 대화 데이터가 배열이 아님:", typeof initialConvs);
          initialConvs = [];
        }
      } catch (e) {
        console.error("로컬 스토리지에서 대화 파싱 중 오류 발생:", e);
        initialConvs = [];
        // 손상된 데이터 제거
        localStorage.removeItem("conversations");
      }
    } else {
      console.log("로컬 스토리지에 저장된 대화 없음");
    }
    
    // conversations가 0개면 새 대화 생성
    if (initialConvs.length === 0) {
      console.log("초기 대화가 없어 새 대화 생성");
      const now = new Date();
      const newConv = {
        id: Date.now().toString(),
        title: `대화 1`,
        timestamp: now.getTime(),
        messages: [
          {
            role: "assistant",
            content: "안녕하세요! 무엇을 도와드릴까요?",
            sources: [],
            timestamp: Date.now(),
          },
        ],
        pinned: false,
        category: defaultCategory,
        metadata: {
          messageCount: 1,
          firstMessageTimestamp: Date.now(),
          lastActivity: Date.now()
        }
      };
      initialConvs = [newConv];
      
      // 상태 업데이트
      setConversations(initialConvs);
      setActiveConversationId(newConv.id);
      setCurrentMessages(newConv.messages);
      
      // 신규 대화 생성 시 localStorage에 즉시 저장 (문자열 따옴표 없이 저장)
      localStorage.setItem("conversations", JSON.stringify(initialConvs));
      localStorage.setItem("activeConversationId", newConv.id);
      
      console.log("새 대화가 생성되고 로컬 스토리지에 저장됨:", newConv.id);
      
      // 백엔드 저장 시도 (비동기로 실행)
      saveConversationToBackend(userId, newConv.id, newConv.messages)
        .catch(err => console.warn("백엔드 저장 실패, 로컬에만 저장됨:", err));
    } else {
      // 기존 대화 목록 있는 경우 - 상태 업데이트
      setConversations(initialConvs);
      
      // 활성 대화 ID 결정
      let initialActiveId = null;
      
      // 저장된 활성 ID가 있는지 확인
      if (savedActiveId) {
        try {
          // JSON 형식으로 저장되어 있을 수 있으므로 파싱 시도
          initialActiveId = savedActiveId.startsWith('"') 
            ? JSON.parse(savedActiveId) 
            : savedActiveId;
          
          // 활성 ID가 실제 존재하는 대화인지 확인
          if (!initialConvs.some((conv) => conv.id === initialActiveId)) {
            console.log("저장된 활성 대화 ID가 존재하지 않아 마지막 대화로 설정");
            initialActiveId = initialConvs[initialConvs.length - 1].id;
          }
        } catch (e) {
          console.warn("활성 대화 ID 파싱 중 오류, 마지막 대화로 설정:", e);
          initialActiveId = initialConvs[initialConvs.length - 1].id;
        }
      } else {
        console.log("저장된 활성 대화 ID가 없어 마지막 대화로 설정");
        initialActiveId = initialConvs[initialConvs.length - 1].id;
      }
      
      // 활성 대화 ID 설정
      setActiveConversationId(initialActiveId);
      
      // 활성 대화의 메시지 설정
      const activeConv = initialConvs.find(c => c.id === initialActiveId);
      if (activeConv && activeConv.messages) {
        setCurrentMessages(activeConv.messages);
      }
      
      // localStorage에 활성 ID 저장 (문자열 따옴표 없이 저장)
      localStorage.setItem("activeConversationId", initialActiveId);
      
      console.log("기존 대화 목록 로드 완료, 활성 대화 ID:", initialActiveId);
    }
    
    // 컴포넌트 마운트 시 입력 필드에 포커스
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('chatInputFocus'));
    }, 800);
  }, []);

  // 초기 설정 로드 (로컬 스토리지 또는 백엔드) 수정
  useEffect(() => {
    // 다크모드로 항상 설정
    const savedTheme = "dark";
    const savedCategory = localStorage.getItem("defaultCategory");
    setTheme(savedTheme);
    document.documentElement.classList.add("dark");
    if (savedCategory) {
      setDefaultCategory(savedCategory);
    }
    // 백엔드에서 설정 불러오기
    loadUserSettingsFromBackend(userId);
  }, [userId]);

  // 검색어 변경 핸들러
  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  // searchTerm 또는 currentMessages가 변경될 때 필터링 로직 적용
  useEffect(() => {
    if (searchTerm && currentMessages.length > 0) {
      const filtered = currentMessages.filter(
        (msg) =>
          typeof msg.content === "string" &&
          msg.content.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredMessages(filtered);
    } else {
      setFilteredMessages(currentMessages);
    }
  }, [searchTerm, currentMessages]);

  const isDark = theme === "dark";

  // 인덱싱된 파일 목록 새로고침을 위한 이벤트 리스너 추가
  useEffect(() => {
    const handleRefreshFilesEvent = () => {
      setFileManagerOpen(true);
    };
    
    window.addEventListener('refreshIndexedFiles', handleRefreshFilesEvent);
    
    return () => {
      window.removeEventListener('refreshIndexedFiles', handleRefreshFilesEvent);
    };
  }, []);

  // 파일 목록 새로고침 함수 추가
  const handleRefreshFiles = () => {
    setFileManagerOpen(true);
    console.log('파일 목록 새로고침');
  };

  // 파일 업로드 완료 핸들러 추가
  const handleUploadSuccess = (files) => {
    console.log('업로드 성공:', files);
    setIsEmbedding(true);
    setEmbeddedFiles(files);

    // 5초 후 임베딩 완료 처리 (실제로는 서버에서 완료 신호를 받아야 함)
    setTimeout(() => {
      setIsEmbedding(false);
      // 임베딩 완료 상태를 설정하고 UI로 표시
      setEmbeddingStatus('완료');
      // 1초 후 임베딩 상태 초기화 (기존 5초에서 1초로 변경)
      setTimeout(() => {
        setEmbeddingStatus(null);
        setEmbeddedFiles([]);
      }, 1000); // 5000에서 1000으로 변경
    }, 5000);
  };

  // 백엔드에 사용자 설정 저장
  const saveUserSettingsToBackend = async (userId, settings) => {
    try {
      const response = await fetch(
        "/api/settings/save",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: userId,
            settings: settings,
          }),
        }
      );
      if (!response.ok) {
        throw new Error(
          `사용자 설정 저장 실패: ${response.status} ${response.statusText}`
        );
      }
      console.log("사용자 설정이 백엔드에 저장되었습니다.");
    } catch (err) {
      console.error("사용자 설정 저장 중 오류 발생:", err);
    }
  };

  // 백엔드에서 사용자 설정 불러오기
  const loadUserSettingsFromBackend = async (userId) => {
    try {
      // 백엔드 연결 시도 전 로컬 기본값 설정
      const savedTheme = localStorage.getItem("theme") || "dark";
      const savedDefaultCategory = localStorage.getItem("defaultCategory") || "메뉴얼";
      
      // 로컬 저장소의 값 적용
      setTheme(savedTheme);
      setDefaultCategory(savedDefaultCategory);
      document.documentElement.classList.toggle("light", savedTheme === "light");
      
      console.log("로컬 설정 적용됨:", { theme: savedTheme, defaultCategory: savedDefaultCategory });
      
      // 백엔드 요청 시도 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3초 타임아웃
      
      const response = await fetch(
        "/api/settings/load",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: userId,
          }),
          signal: controller.signal
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`사용자 설정 불러오기 실패: ${response.status} ${response.statusText}`);
        return; // 이미 로컬 설정이 적용되었으므로 종료
      }
      
      const data = await response.json();
      if (data.status === "success" && data.settings) {
        if (data.settings.theme) {
          setTheme(data.settings.theme);
          localStorage.setItem("theme", data.settings.theme);
          document.documentElement.classList.toggle(
            "light",
            data.settings.theme === "light"
          );
        }
        if (data.settings.defaultCategory) {
          setDefaultCategory(data.settings.defaultCategory);
          localStorage.setItem(
            "defaultCategory",
            data.settings.defaultCategory
          );
        }
        console.log("사용자 설정이 백엔드에서 불러와졌습니다.");
      } else if (data.status === "not_found") {
        console.log("사용자 설정이 없습니다. 로컬 설정을 유지합니다.");
      }
    } catch (err) {
      // 오류 발생 시 콘솔에 경고만 표시하고 앱은 계속 실행
      console.warn("사용자 설정 불러오기 중 오류 발생:", err.message);
      console.log("로컬 설정을 사용합니다.");
    }
  };

  // activeConversationId 변경 시 localStorage에 저장
  useEffect(() => {
    if (activeConversationId) {
      // 문자열 직접 저장 - JSON.stringify 사용하지 않음
      localStorage.setItem("activeConversationId", activeConversationId);
      console.log("활성 대화 ID 저장:", activeConversationId);
    }
  }, [activeConversationId]);

  // 대화 메시지 변경 시 자동 저장
  useEffect(() => {
    // 대화가 변경되면 로컬 스토리지에 자동 저장
    if (conversations.length > 0) {
      try {
        // 직렬화 전에 DOM 요소 참조를 제거하기 위해 순수 객체만 추출
        const cleanConversations = conversations.map(conv => ({
          id: conv.id,
          title: conv.title,
          timestamp: conv.timestamp || Date.now(),
          pinned: !!conv.pinned,
          category: conv.category || defaultCategory,
          messages: Array.isArray(conv.messages) ? conv.messages.map(msg => ({
            role: msg.role || "assistant",
            content: msg.content || "",
            timestamp: msg.timestamp || Date.now(),
            sources: Array.isArray(msg.sources) ? msg.sources : []
          })) : [],
          metadata: conv.metadata || {
            messageCount: Array.isArray(conv.messages) ? conv.messages.length : 0,
            firstMessageTimestamp: Date.now(),
            lastActivity: Date.now()
          }
        }));
        
        // 유효성 검사 후 저장
        if (Array.isArray(cleanConversations) && cleanConversations.every(conv => conv.id)) {
          localStorage.setItem("conversations", JSON.stringify(cleanConversations));
          console.log("로컬 스토리지에 대화 저장 완료 - 대화 수:", cleanConversations.length);
        } else {
          console.error("저장 시도 중 유효하지 않은 대화 데이터 발견:", 
            cleanConversations.filter(conv => !conv.id).length);
        }
      } catch (error) {
        console.error("대화 저장 중 오류 발생:", error);
      }
      
      // 활성화된 대화에 대해서만 백엔드에 저장
      if (activeConversationId) {
        const activeConv = conversations.find(
          (c) => c.id === activeConversationId
        );
        if (activeConv && activeConv.messages && activeConv.messages.length > 0) {
          // 백엔드 저장 비활성화 (422 오류 문제 해결을 위함)
          // saveConversationToBackend(userId, activeConversationId, activeConv.messages);
        }
      }
    }
  }, [conversations, activeConversationId, userId, defaultCategory]);

  // 전체 대화 삭제 기능 추가
  const deleteAllConversations = () => {
    try {
      // 로컬 상태 초기화
      setConversations([]);
      setActiveConversationId(null);
      setCurrentMessages([
        // 시스템 시작 메시지 추가하여 챗봇이 먼저 인사하도록 함
        {
          role: "assistant",
          content: "안녕하세요! 무엇을 도와드릴까요?",
          sources: [],
          timestamp: Date.now(),
        }
      ]);
      
      // 로컬 스토리지 데이터 삭제
      localStorage.removeItem('conversations');
      localStorage.removeItem('activeConversationId');
      
      // 백엔드에 요청 (옵션)
      fetch("/api/conversations/delete-all", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ userId: "user123" }),
      }).catch(err => {
        console.warn("백엔드에 전체 삭제 요청 실패:", err);
        // 백엔드 실패해도 로컬에서는 삭제된 상태로 유지
      });
      
      // 성공 메시지 표시
      showModeChangeToast("모든 대화가 삭제되었습니다");
      
      // 새 대화 시작 - 새 대화 생성 안함 (이미 초기 메시지 설정됨)
      // handleNewConversation(null, defaultCategory, true);
    } catch (err) {
      console.error("대화 전체 삭제 중 오류 발생:", err);
    }
  };

  // 새 대화 생성
  const handleNewConversation = (topic, category, forceFirst = false) => {
    try {
      console.log('새 대화 시작...', topic, category);
    
      // 초기 제목 설정 (forceFirst가 true이면 항상 "대화 1"로 설정)
      const initialTitle = topic || (forceFirst ? "제목 없음" : `대화 ${conversations.length + 1}`);
      
      // 고유 ID 생성
      const newId = `conv_${Date.now()}`;
      
      // 새 대화 객체 생성 (구조 개선 - 메타데이터 추가)
      const newConversation = {
        id: newId,
        title: initialTitle,
        messages: [
          // 시스템 시작 메시지 추가하여 챗봇이 먼저 인사하도록 함
          {
            role: "assistant",
            content: "안녕하세요! 무엇을 도와드릴까요?",
            sources: [],
            timestamp: Date.now(),
          }
        ],
        timestamp: Date.now(),
        category: category || defaultCategory,
        pinned: false,
        metadata: {
          messageCount: 1,
          firstMessageTimestamp: Date.now(),
          lastActivity: Date.now()
        }
      };
    
      let updatedConversations;
      
      // 새 대화 추가 (맨 앞 또는 맨 뒤)
      if (forceFirst) {
        // 가장 최근 대화로 추가 (맨 앞)
        updatedConversations = [newConversation, ...conversations];
        setConversations(updatedConversations);
      } else {
        // 새 대화 추가 (맨 뒤)
        updatedConversations = [...conversations, newConversation];
        setConversations(updatedConversations);
      }
      
      // 새 대화를 현재 활성 대화로 설정
      setActiveConversationId(newId);
      
      // 메시지 초기화
      setCurrentMessages([
        // 시스템 시작 메시지 추가하여 챗봇이 먼저 인사하도록 함
        {
          role: "assistant",
          content: "안녕하세요! 무엇을 도와드릴까요?",
          sources: [],
          timestamp: Date.now(),
        }
      ]);
      
      // 검색어 초기화
      setSearchTerm("");
      
      // 사이드바 모바일에서 자동으로 닫기
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      }
      
      // 새 대화 생성 즉시 localStorage에 저장
      try {
        // 직렬화 전에 DOM 요소 참조를 제거하기 위해 순수 객체만 추출
        const cleanConversations = updatedConversations.map(conv => ({
          id: conv.id,
          title: conv.title,
          timestamp: conv.timestamp,
          pinned: !!conv.pinned,
          messages: conv.messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp || Date.now(),
            sources: Array.isArray(msg.sources) ? msg.sources : []
          }))
        }));
        
        localStorage.setItem("conversations", JSON.stringify(cleanConversations));
        localStorage.setItem("activeConversationId", JSON.stringify(newId));
        console.log("새 대화 생성 후 로컬 스토리지에 즉시 저장 완료");
        
        // 백엔드 저장 시도 (비동기로 실행)
        if (newConversation.messages.length > 0) {
          saveConversationToBackend(userId, newId, newConversation.messages)
            .catch(err => console.warn("백엔드 저장 실패, 로컬에만 저장됨:", err));
        }
      } catch (error) {
        console.error("대화 저장 중 오류 발생:", error);
      }
      
      // 새 대화 생성 후 입력 필드에 포커스
      console.log('App: 새 대화 생성 후 포커스 이벤트 발생');
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('chatInputFocus'));
      }, 500);
      
      return newId;
    } catch (error) {
      console.error('새 대화 생성 중 오류 발생:', error);
      return null;
    }
  };

  // 대화 선택
  const handleSelectConversation = (id) => {
    console.log(`대화 선택: ${id} - 메시지 로딩 시작`);
    
    // 먼저 activeConversationId 업데이트
    setActiveConversationId(id);
    
    // 로컬 스토리지에서 최신 대화 데이터 직접 불러오기
    try {
      const savedConversations = localStorage.getItem("conversations");
      if (savedConversations) {
        const parsedConversations = JSON.parse(savedConversations);
        const freshConversation = parsedConversations.find(conv => conv.id === id);
        
        if (freshConversation && freshConversation.messages && freshConversation.messages.length > 0) {
          console.log(`로컬 스토리지에서 대화 로드 - ID: ${id}, 메시지 수: ${freshConversation.messages.length}`);
          
          // 완전한 대화 데이터 발견시 현재 메시지 상태 업데이트
          setCurrentMessages(freshConversation.messages);
          
          // conversations 상태도 업데이트
          setConversations(prev => {
            // 이미 존재하는 대화만 업데이트
            return prev.map(conv => 
              conv.id === id ? {...conv, messages: freshConversation.messages} : conv
            );
          });
        }
      }
    } catch (error) {
      console.error("로컬 스토리지에서 대화 로드 중 오류:", error);
    }
    
    // 메모리(React 상태)에서 현재 선택된 대화 찾기
    const selectedConv = conversations.find(conv => conv.id === id);
    
    // 메모리에서 찾은 대화가 있고, 로컬 스토리지에서 로딩하지 않았다면 메모리 데이터 사용
    if (selectedConv && selectedConv.messages && selectedConv.messages.length > 0) {
      console.log(`메모리에서 대화 데이터 로드 - 메시지 수: ${selectedConv.messages.length}`);
      setCurrentMessages(selectedConv.messages);
    }
    
    // 백엔드에서도 최신 데이터 불러오기 (항상 실행하여 가장 최신 데이터 확보)
    loadConversationFromBackend(userId, id);
    
    // 모바일에서 대화 선택 시 사이드바 닫기
    if (window.innerWidth < 768) {
      setSidebarOpen(false);
    }
    
    // 대화창 전환 시 스크롤을 최신 메시지로 이동 (forceScroll 플래그 추가)
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('chatScrollToBottom', { 
        detail: { forceScroll: true }
      }));
    }, 300);
    
    // 대화 선택 후 입력 필드에 포커스
    console.log('App: 대화 선택 후 포커스 이벤트 발생');
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('chatInputFocus'));
    }, 500);
  };

  // 대화 삭제
  const handleDeleteConversation = (id) => {
    // 확인 창 없이 바로 삭제 처리
    setConversations((prev) => {
      const updated = prev.filter((conv) => conv.id !== id);
      let newActive = activeConversationId;
      if (id === activeConversationId) {
        newActive = updated.length > 0 ? updated[updated.length - 1].id : null;
        setActiveConversationId(newActive);
      }
      // conversations가 0개가 되면 새 대화 자동 생성
      if (updated.length === 0) {
        const now = new Date();
        const newConv = {
          id: Date.now().toString(),
          title: "대화 1",
          timestamp: now.toLocaleString(),
          messages: [
            {
              role: "assistant",
              content: "안녕하세요! 무엇을 도와드릴까요?",
              sources: [],
            },
          ],
          pinned: false,
        };
        setActiveConversationId(newConv.id);
        return [newConv];
      }
      return updated;
    });
  };

  // 대화 제목 변경
  const handleRenameConversation = (id, newTitle) => {
    let updatedConversations;
    setConversations((prev) => {
      const updated = prev.map((conv) => (conv.id === id ? { ...conv, title: newTitle } : conv));
      updatedConversations = updated;
      return updated;
    });
    
    // 제목 변경 즉시 localStorage에 저장
    try {
      if (updatedConversations) {
        const cleanConversations = updatedConversations.map(conv => ({
          id: conv.id,
          title: conv.title,
          timestamp: conv.timestamp,
          pinned: !!conv.pinned,
          messages: conv.messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp || Date.now(),
            sources: Array.isArray(msg.sources) ? msg.sources : []
          }))
        }));
        
        localStorage.setItem("conversations", JSON.stringify(cleanConversations));
        console.log("제목 변경 후 로컬 스토리지에 즉시 저장 완료");
        
        // 백엔드 저장 시도 (비동기로 실행)
        const conversation = updatedConversations.find(c => c.id === id);
        if (conversation && conversation.messages.length > 0) {
          saveConversationToBackend(userId, id, conversation.messages)
            .catch(err => console.warn("백엔드 저장 실패, 로컬에만 저장됨:", err));
        }
      }
    } catch (error) {
      console.error("대화 저장 중 오류 발생:", error);
    }
  };

  // 대화 즐겨찾기 토글
  const handleTogglePinConversation = (id) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === id ? { ...conv, pinned: !conv.pinned } : conv
      )
    );
  };

  // 백엔드에 대화 저장
  const saveConversationToBackend = async (
    userId,
    conversationId,
    messages
  ) => {
    try {
      // 백엔드 요청 시도 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3초 타임아웃
      
      const response = await fetch(
        "/api/conversations/save",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: userId,
            conversationId: conversationId,
            messages: messages.map((msg) => ({
              role: msg.role,
              content: msg.content,
              sources: msg.sources || [],
            })),
          }),
          signal: controller.signal
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`대화 저장 실패: ${response.status} ${response.statusText}`);
        setSaveError("대화 저장에 실패했습니다. 로컬에만 저장됩니다.");
        setTimeout(() => setSaveError(null), 5000);
        return;
      }
      
      console.log("대화가 백엔드에 저장되었습니다.");
      setSaveError(null); // 성공 시 오류 메시지 초기화
    } catch (err) {
      console.warn("대화 저장 중 오류 발생:", err.message);
      setSaveError("대화 저장에 실패했습니다. 로컬에만 저장됩니다."); // 오류 메시지 설정
      // 로컬 저장은 이미 conversations 변경 시 useEffect에서 보장됨
      setTimeout(() => setSaveError(null), 5000); // 5초 후 알림 사라짐
    }
  };

  // 백엔드에서 대화 불러오기
  const loadConversationFromBackend = async (userId, conversationId) => {
    try {
      // 현재 메모리에 있는 대화 데이터 찾기 (비교용)
      const currentConversation = conversations.find(c => c.id === conversationId);
      const currentMessageCount = currentConversation?.messages?.length || 0;
      
      console.log(`백엔드 대화 로드 시작 - ID: ${conversationId}, 현재 메시지 수: ${currentMessageCount}`);
      
      // 백엔드 요청 시도 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3초 타임아웃
      
      const response = await fetch(
        "/api/conversations/load",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: userId,
            conversationId: conversationId,
          }),
          signal: controller.signal
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`대화 불러오기 실패: ${response.status} ${response.statusText}`);
        return; // 로컬 데이터 유지
      }
      
      const data = await response.json();
      if (data.status === "success" && data.conversation) {
        // 백엔드에서 받은 메시지가 있고 유효한지 확인
        if (data.conversation.messages && Array.isArray(data.conversation.messages)) {
          const backendMessageCount = data.conversation.messages.length;
          console.log(`백엔드 응답 성공 - 메시지 수: ${backendMessageCount}`);
          
          // 백엔드 데이터가 더 많은 메시지를 가지고 있거나 현재 데이터가 없는 경우에만 업데이트
          if (backendMessageCount > currentMessageCount || currentMessageCount === 0) {
            console.log(`백엔드 데이터가 더 많은 메시지를 포함하여 업데이트 진행`);
            
            // conversations 상태 업데이트
            setConversations((prev) =>
              prev.map((conv) =>
                conv.id === conversationId
                  ? { ...conv, messages: data.conversation.messages }
                  : conv
              )
            );
            
            // 현재 활성 대화의 메시지 상태도 즉시 업데이트
            if (activeConversationId === conversationId) {
              setCurrentMessages(data.conversation.messages);
              
              // 로컬 스토리지에도 즉시 업데이트 (다음 로드 시 최신 상태 유지)
              try {
                // 전체 conversations 가져와서 특정 대화만 업데이트
                const savedConversations = localStorage.getItem("conversations");
                if (savedConversations) {
                  let parsedConversations = JSON.parse(savedConversations);
                  parsedConversations = parsedConversations.map(conv => 
                    conv.id === conversationId 
                      ? {...conv, messages: data.conversation.messages}
                      : conv
                  );
                  localStorage.setItem("conversations", JSON.stringify(parsedConversations));
                  console.log("백엔드 데이터로 로컬 스토리지 업데이트 완료");
                }
              } catch (err) {
                console.warn("로컬 스토리지 업데이트 중 오류:", err);
              }
            }
            
            console.log("대화가 백엔드에서 불러와져 업데이트 되었습니다.");
          } else {
            console.log("백엔드 데이터가 현재 데이터보다 적거나 동일하여 업데이트 생략");
          }
        } else {
          console.warn("백엔드에서 받은 메시지가 비어있거나 유효하지 않습니다.");
        }
      } else {
        console.log("백엔드에서 대화를 찾을 수 없거나 응답 형식이 잘못되었습니다.");
      }
    } catch (err) {
      console.warn("대화 불러오기 중 오류 발생:", err.message);
      console.log("로컬 대화 데이터를 사용합니다.");
    }
  };

  // conversations와 activeConversationId가 초기화된 경우에만 메시지 가져오기
  useEffect(() => {
    if (conversations.length > 0 && activeConversationId) {
      const activeConv = conversations.find(
        (conv) => conv.id === activeConversationId
      );
      if (activeConv) {
        console.log("대화 선택: 메시지 개수", activeConv.messages.length);
        setCurrentMessages(activeConv.messages);
        
        // 메시지가 있는지 확인하고 백엔드에서 다시 로드
        // 만약 messages가 비어있거나 첫 메시지만 있다면 백엔드에서 다시 로드
        if (!activeConv.messages || activeConv.messages.length <= 1) {
          console.log("불완전한 메시지 감지: 백엔드에서 전체 대화 다시 로드");
          loadConversationFromBackend(userId, activeConversationId);
        }
      }
    } else {
      setCurrentMessages([
        {
          role: "assistant",
          content: "안녕하세요! 무엇을 도와드릴까요?",
          sources: [],
        },
      ]);
    }
  }, [conversations, activeConversationId, userId]);

  // 반응형: 화면 크기 변경 시 사이드바 상태 업데이트
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768 && sidebarOpen) {
        setSidebarOpen(false);
      } else if (window.innerWidth >= 1200 && !sidebarOpen) {
        setSidebarOpen(true);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [sidebarOpen]);

  // 드래그 핸들러
  const handleMouseDown = (e) => {
    isResizing.current = true;
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };
  const handleMouseMove = (e) => {
    if (!isResizing.current) return;
    const newWidth = Math.max(
      SIDEBAR_MIN,
      Math.min(SIDEBAR_MAX, e.clientX)
    );
    setSidebarWidth(newWidth);
  };
  const handleMouseUp = () => {
    isResizing.current = false;
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  // 토글(여닫이) 버튼
  const handleToggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // 메시지 업데이트
  const handleUpdateMessages = (updatedMessages) => {
    if (!activeConversationId) return;
    
    // 대화 업데이트
    let updatedConversations;
    setConversations((prev) => {
      const updated = prev.map((conv) => {
        if (conv.id === activeConversationId) {
          // 기존 대화 업데이트
          return { ...conv, messages: updatedMessages, timestamp: new Date().getTime() };
        }
        return conv;
      });
      updatedConversations = updated;
      return updated;
    });
    setCurrentMessages(updatedMessages);
    
    // 메시지 업데이트 즉시 localStorage에 저장
    try {
      // 직렬화 전에 DOM 요소 참조를 제거하기 위해 순수 객체만 추출
      // 상태 업데이트가 비동기적이므로 방금 업데이트한 conversations를 사용할 수 없음
      // 직접 업데이트된 대화 목록을 생성하여 저장
      if (updatedConversations) {
        const cleanConversations = updatedConversations.map(conv => ({
          id: conv.id,
          title: conv.title,
          timestamp: conv.timestamp,
          pinned: !!conv.pinned,
          messages: conv.messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp || Date.now(),
            sources: Array.isArray(msg.sources) ? msg.sources : []
          }))
        }));
        
        localStorage.setItem("conversations", JSON.stringify(cleanConversations));
        console.log("메시지 업데이트 후 로컬 스토리지에 즉시 저장 완료");
      }
    } catch (error) {
      console.error("대화 저장 중 오류 발생:", error);
    }
    
    // 자동 제목 생성 로직: 사용자 메시지가 있을 때 모든 제목에 대해 동작
    const userMessages = updatedMessages.filter(msg => msg.role === "user");
    
    // 첫 번째 사용자 메시지가 추가됐을 때만 자동 제목 생성을 수행
    const activeConv = conversations.find(c => c.id === activeConversationId);
    const prevUserMessages = activeConv?.messages?.filter(msg => msg.role === "user") || [];
    
    if (userMessages.length > 0 && prevUserMessages.length === 0) {
      console.log("첫 질문 감지: 자동 제목 생성 시도");
      // 비동기로 제목 생성 API 호출
      generateTitleForConversation(activeConversationId, updatedMessages);
    }
  };

  // 대화 제목 자동 생성 함수
  const generateTitleForConversation = async (conversationId, messages) => {
    try {
      console.log("제목 생성 API 호출 - 메시지:", messages);
      
      // 백엔드 요청 시도 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5초 타임아웃
      
      const response = await fetch("/api/generate-title", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ messages }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`제목 생성 실패: ${response.status} ${response.statusText}`);
        // 백엔드 API 호출 실패 시 대화 메시지 기반으로 제목 생성
        generateFallbackTitle(conversationId, messages);
        return;
      }
      
      const data = await response.json();
      console.log("제목 생성 응답:", data);
      
      if (data.status === "success" && data.title) {
        // 백엔드에서 제공한 제목으로 업데이트
        handleRenameConversation(conversationId, data.title);
      } else {
        // 실패하면 폴백 제목 생성
        generateFallbackTitle(conversationId, messages);
      }
    } catch (err) {
      console.warn("제목 생성 중 오류 발생:", err.message);
      // 오류 발생 시 폴백 제목 생성
      generateFallbackTitle(conversationId, messages);
    }
  };

  // 오류 시 폴백 제목 생성 함수
  const generateFallbackTitle = (conversationId, messages) => {
    try {
      // 첫 사용자 메시지를 기반으로 제목 생성
      const userMessage = messages.find(msg => msg.role === "user");
      
      if (userMessage) {
        // 첫 질문의 처음 15자를 추출하고 말줄임표 추가
        let title = userMessage.content.trim().substring(0, 15);
        if (userMessage.content.length > 15) {
          title += "...";
        }
        
        // 제목이 너무 짧으면 기본 제목 사용
        if (title.length < 5) {
          const defaultTitle = `대화 ${new Date().toLocaleDateString('ko-KR')}`;
          handleRenameConversation(conversationId, defaultTitle);
        } else {
          // 추출한 제목으로 업데이트
          handleRenameConversation(conversationId, title);
        }
      }
    } catch (error) {
      console.warn("폴백 제목 생성 실패:", error);
      // 최종 폴백: 현재 날짜/시간 기반 제목
      const timestamp = new Date().toLocaleDateString('ko-KR', { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit' 
      });
      handleRenameConversation(conversationId, `대화 ${timestamp}`);
    }
  };

  // 모드 전환 함수 - 챗봇 <-> SQL 질의 모드 전환
  const onToggleMode = (mode) => {
    console.log('onToggleMode 호출됨:', mode, '현재 상태:', showSQLPage);
    
    try {
      if (mode === 'sql') {
        console.log('SQL 모드로 전환 중...');
        if (!showSQLPage) {  // 현재 SQL 모드가 아닐 때만 전환
          setShowSQLPage(true);
          // 모드 전환 토스트 메시지 표시
          showModeChangeToast('SQL 질의 모드로 전환했습니다');
        }
      } else if (mode === 'chat') {
        console.log('챗봇 모드로 전환 중...');
        if (showSQLPage) {  // 현재 SQL 모드일 때만 전환
          setShowSQLPage(false);
          // 모드 전환 토스트 메시지 표시
          showModeChangeToast('챗봇 모드로 전환했습니다');
        }
      } else {
        // 모드가 지정되지 않은 경우 토글
        const newMode = !showSQLPage;
        console.log('모드 토글 중...', newMode ? 'SQL로' : '챗봇으로');
        setShowSQLPage(newMode);
        // 모드 전환 토스트 메시지 표시
        showModeChangeToast(newMode ? 'SQL 질의 모드로 전환했습니다' : '챗봇 모드로 전환했습니다');
      }
      console.log('모드 전환 요청 완료. 현재 showSQLPage 상태:', showSQLPage, '(실제 변경은 리렌더링 후 적용됨)');
    } catch (error) {
      console.error('모드 전환 중 오류 발생:', error);
    }
  };

  // 모드 전환 토스트 메시지 표시 함수
  const showModeChangeToast = (message) => {
    // 토스트 메시지 비활성화
    console.log('토스트 메시지 비활성화됨:', message);
    return; // 함수 실행 중단
  };

  // 대시보드 통계 데이터 가져오기
  const fetchDashboardStats = useCallback(async () => {
    try {
      const response = await fetch("/api/dashboard/stats", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          startDate: '2023-01-01',
          endDate: '2023-12-31',
          timeRange: 'month'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.stats) {
          setDashboardStats(data.stats);
        }
      } else {
        console.error('대시보드 통계 데이터를 불러오는데 실패했습니다.');
        
        // 목데이터 설정 (API 응답 실패 시)
        setDashboardStats({
          totalQueries: 1284,
          totalChats: 3562,
          activeUsers: 215,
          averageQueriesPerDay: 42.8,
          queryCountByDate: [
            { date: '1월', value: 130 },
            { date: '2월', value: 245 },
            { date: '3월', value: 375 },
            { date: '4월', value: 480 },
            { date: '5월', value: 520 },
            { date: '6월', value: 620 }
          ]
        });
      }
    } catch (error) {
      console.error('대시보드 통계 데이터 가져오기 오류:', error);
      
      // 목데이터 설정 (오류 발생 시)
      setDashboardStats({
        totalQueries: 1284,
        totalChats: 3562,
        activeUsers: 215,
        averageQueriesPerDay: 42.8,
        queryCountByDate: [
          { date: '1월', value: 130 },
          { date: '2월', value: 245 },
          { date: '3월', value: 375 },
          { date: '4월', value: 480 },
          { date: '5월', value: 520 },
          { date: '6월', value: 620 }
        ]
      });
    }
  }, []);
  
  // 최근 SQL 쿼리 가져오기
  const fetchRecentQueries = useCallback(async () => {
    try {
      const response = await fetch("/api/sql/recent-queries");
      
      if (response.ok) {
        const data = await response.json();
        if (data.queries) {
          setRecentQueries(data.queries);
        }
      } else {
        console.error('최근 쿼리 데이터를 불러오는데 실패했습니다.');
        
        // 목데이터 설정
        setRecentQueries([
          {
            id: 1,
            question: "지난 달 매출이 가장 높은 상품은?",
            sql: "SELECT product_name, SUM(amount) as total FROM sales WHERE date >= '2023-05-01' GROUP BY product_name ORDER BY total DESC LIMIT 1",
            timestamp: Date.now() - 1000 * 60 * 30 // 30분 전
          },
          {
            id: 2,
            question: "현재 재고가 10개 미만인 상품 목록",
            sql: "SELECT product_name, stock FROM inventory WHERE stock < 10 ORDER BY stock ASC",
            timestamp: Date.now() - 1000 * 60 * 60 * 2 // 2시간 전
          },
          {
            id: 3,
            question: "각 지역별 고객 수 통계",
            sql: "SELECT region, COUNT(*) as customer_count FROM customers GROUP BY region ORDER BY customer_count DESC",
            timestamp: Date.now() - 1000 * 60 * 60 * 24 // 어제
          }
        ]);
      }
    } catch (error) {
      console.error('최근 쿼리 데이터 가져오기 오류:', error);
      
      // 목데이터 설정
      setRecentQueries([
        {
          id: 1,
          question: "지난 달 매출이 가장 높은 상품은?",
          sql: "SELECT product_name, SUM(amount) as total FROM sales WHERE date >= '2023-05-01' GROUP BY product_name ORDER BY total DESC LIMIT 1",
          timestamp: Date.now() - 1000 * 60 * 30 // 30분 전
        },
        {
          id: 2,
          question: "현재 재고가 10개 미만인 상품 목록",
          sql: "SELECT product_name, stock FROM inventory WHERE stock < 10 ORDER BY stock ASC",
          timestamp: Date.now() - 1000 * 60 * 60 * 2 // 2시간 전
        },
        {
          id: 3,
          question: "각 지역별 고객 수 통계",
          sql: "SELECT region, COUNT(*) as customer_count FROM customers GROUP BY region ORDER BY customer_count DESC",
          timestamp: Date.now() - 1000 * 60 * 60 * 24 // 어제
        }
      ]);
    }
  }, []);
  
  // DB 스키마 정보 가져오기
  const fetchDbSchema = useCallback(async () => {
    try {
      const response = await fetch("/api/sql/schema");
      
      if (response.ok) {
        const data = await response.json();
        if (data.schema) {
          setDbSchema(data.schema);
        }
      } else {
        console.error('DB 스키마 정보를 불러오는데 실패했습니다.');
        
        // 목데이터 설정
        setDbSchema({
          'customers': ['id', 'name', 'email', 'phone', 'region', 'created_at'],
          'products': ['id', 'product_name', 'category', 'price', 'description'],
          'sales': ['id', 'customer_id', 'product_id', 'date', 'amount', 'quantity'],
          'inventory': ['id', 'product_id', 'stock', 'updated_at']
        });
      }
    } catch (error) {
      console.error('DB 스키마 정보 가져오기 오류:', error);
      
      // 목데이터 설정
      setDbSchema({
        'customers': ['id', 'name', 'email', 'phone', 'region', 'created_at'],
        'products': ['id', 'product_name', 'category', 'price', 'description'],
        'sales': ['id', 'customer_id', 'product_id', 'date', 'amount', 'quantity'],
        'inventory': ['id', 'product_id', 'stock', 'updated_at']
      });
    }
  }, []);

  // 모드가 변경될 때 필요한 데이터 로딩
  useEffect(() => {
    if (mode === 'dashboard') {
      fetchDashboardStats();
    } else if (mode === 'sql') {
      fetchRecentQueries();
      fetchDbSchema();
    }
  }, [mode, fetchDashboardStats, fetchRecentQueries, fetchDbSchema]);

  const handleSubmit = async (input, selectedCategory) => {
    if (!input.trim()) return;

    // 사용자 메시지 추가
    const userMessage = {
      role: "user",
      content: input,
      timestamp: new Date().getTime(),
    };

    // 임시 응답 메시지 추가
    const tempResponseMessage = {
      role: "assistant",
      content: "",
      timestamp: new Date().getTime(),
      isStreaming: true, // 스트리밍 중임을 표시
    };

    // 메시지 목록 업데이트
    setCurrentMessages((prevMessages) => [...prevMessages, userMessage, tempResponseMessage]);

    // 대화 기록 구성
    const conversationHistory = currentMessages
      .filter((msg) => msg.role === "user" || msg.role === "assistant")
      .map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

    try {
      // 새로운 AbortController 생성
      abortControllerRef.current = new AbortController();
      const { signal } = abortControllerRef.current;

      // 스트리밍 상태 활성화
      setIsStreaming(true);
      
      console.log("App: 요청 시작, AbortController 생성됨");

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: input,
          category: selectedCategory,
          history: conversationHistory,
        }),
        signal, // AbortController 시그널 추가
      });

      if (!response.ok) {
        throw new Error(`서버 오류: ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let responseText = "";

      // 응답 스트리밍 처리
      while (!done) {
        // 중단 여부 확인 (매 반복마다)
        if (signal.aborted) {
          console.log("App: 스트리밍 중 중단 신호 감지됨");
          break;
        }

        const { value, done: readerDone } = await reader.read();
        done = readerDone;

        if (value) {
          const chunk = decoder.decode(value, { stream: !done });
          responseText += chunk;

          // 메시지 업데이트
          setCurrentMessages((prevMessages) => {
            const lastMessage = prevMessages[prevMessages.length - 1];
            if (lastMessage && lastMessage.role === "assistant") {
              return [
                ...prevMessages.slice(0, -1),
                { ...lastMessage, content: responseText },
              ];
            }
            return prevMessages;
          });
        }
      }

      // 스트리밍 완료 후 처리
      if (!signal.aborted) {
        console.log("App: 응답 스트리밍 완료");
        
        // 메시지에 최종 응답 설정 및 스트리밍 상태 제거
        setCurrentMessages((prevMessages) => {
          const lastMessage = prevMessages[prevMessages.length - 1];
          if (lastMessage && lastMessage.role === "assistant") {
            return [
              ...prevMessages.slice(0, -1),
              { 
                ...lastMessage, 
                content: responseText,
                isStreaming: false
              },
            ];
          }
          return prevMessages;
        });
        
        // 응답 완료 후 입력 필드에 포커스
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('chatInputFocus'));
        }, 100);
      }
    } catch (error) {
      // 중단 요청에 의한 에러인 경우 특별 처리
      if (error.name === 'AbortError') {
        console.log('App: 사용자에 의해 응답이 중단되었습니다.');
        // 중단 메시지 처리는 handleStopGeneration에서 일괄적으로 수행하므로 여기서는 추가 작업 없음
      } else {
        console.error("App: 응답 처리 중 오류 발생:", error);
        // 기존 에러 처리 로직 유지
        const errorMessage = {
          role: "assistant",
          content: "죄송합니다. 응답 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
          timestamp: new Date().getTime(),
        };
        setCurrentMessages(prevMessages => {
          // 마지막 메시지가 빈 어시스턴트 메시지라면 교체, 아니면 추가
          const lastMsg = prevMessages[prevMessages.length - 1];
          if (lastMsg && lastMsg.role === "assistant" && !lastMsg.content) {
            return [...prevMessages.slice(0, -1), errorMessage];
          } else {
            return [...prevMessages, errorMessage];
          }
        });
        
        // 오류 발생 후에도 입력 필드에 포커스
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('chatInputFocus'));
        }, 100);
      }
    } finally {
      // 스트리밍 상태 비활성화 및 AbortController 참조 해제
      // handleStopGeneration 함수에서도 setIsStreaming(false) 및 abortControllerRef.current = null을 호출하므로,
      // 여기서의 호출은 스트리밍이 정상적으로 완료되었거나, AbortError 외의 다른 오류로 종료된 경우를 담당합니다.
      // AbortError의 경우 handleStopGeneration에서 이미 처리되었으므로, 여기서는 추가적인 상태 변경이 발생하지 않습니다.
      if (abortControllerRef.current !== null) { // handleStopGeneration에서 null로 설정되지 않은 경우에만 실행
        setIsStreaming(false);
        abortControllerRef.current = null;
        console.log("App: handleSubmit finally - 스트리밍 상태 비활성화 및 컨트롤러 해제");
      }
    }
  };

  // 응답 생성 중단 함수
  const handleStopGeneration = () => {
    console.log("App: 응답 생성 중단 함수 실행");
    
    // 이미 스트리밍이 중지된 상태라면 함수 종료
    if (!isStreaming) {
      console.log("App: 이미 스트리밍이 중단되어 있음");
      return;
    }
    
    // 디버그 정보 출력
    console.log("App: 현재 AbortController 상태:", {
      exists: !!abortControllerRef.current,
      isStreaming: isStreaming
    });
    
    // 스트리밍 상태 즉시 비활성화 (가장 먼저 실행)
    setIsStreaming(false);
    
    if (abortControllerRef.current) {
      console.log("App: AbortController 중단 신호 발생");
      
      try {
        // 명시적으로 abort 메서드 호출
        abortControllerRef.current.abort("사용자가 응답 생성을 중지함");
        console.log("App: abort() 메서드 호출 성공");
      } catch (abortError) {
        console.error("App: abort() 호출 중 오류:", abortError);
      }
      
      // 즉시 참조 초기화
      abortControllerRef.current = null;
      
      // 마지막 메시지가 어시스턴트 메시지인 경우에만 중단 표시 추가
      const lastMessage = currentMessages[currentMessages.length - 1];
      
      if (lastMessage && lastMessage.role === "assistant") {
        console.log("App: 응답 중단 메시지 추가");
        
        // 기존 메시지에 중단 표시 추가
        const updatedLastMessage = {
          ...lastMessage,
          content: lastMessage.content + "\n\n(응답이 중단되었습니다)",
          isStopped: true // 중단 플래그 추가
        };
        
        // 마지막 메시지를 업데이트된 메시지로 교체
        setCurrentMessages(prevMessages => [
          ...prevMessages.slice(0, -1),
          updatedLastMessage
        ]);
        
        // 응답 중단 후 입력 필드에 포커스
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('chatInputFocus'));
        }, 100);
      }
      
      console.log("App: 응답 중단 처리 완료");
    } else {
      console.log("App: 중단할 AbortController가 없음");
    }
  };

  // 이미지 보기 핸들러
  const handleViewImages = (source) => {
    console.log("이미지 보기 요청:", source);
    setSelectedImageSource(source);
    setImageModalOpen(true);
  };

  // 이미지 모달 닫기 핸들러
  const handleCloseImageModal = () => {
    setImageModalOpen(false);
    setSelectedImageSource(null);
  };

  return (
    <div className="flex flex-col md:flex-row h-screen bg-gray-900 text-gray-100 overflow-hidden relative">
      {/* 임베딩 처리 오버레이 */}
      <EmbeddingOverlay 
        isActive={isEmbedding} 
        status={embeddingStatus} 
        files={embeddedFiles}
      />
      
      {/* 사이드바 */}
      <div
        className={`absolute md:relative inset-y-0 left-0 z-20 transform ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        } md:translate-x-0 transition-transform duration-300 ease-in-out flex-shrink-0 w-${sidebarWidth}px flex flex-col border-r border-gray-800 h-full bg-gray-900`}
        style={{ width: `${sidebarWidth}px` }}
      >
        <Sidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onRenameConversation={handleRenameConversation}
          onDeleteConversation={handleDeleteConversation}
          onTogglePinConversation={handleTogglePinConversation}
          onToggleTheme={toggleTheme}
          isDarkMode={theme === "dark"}
          onToggleMode={setMode}
          currentMode={mode}
          onDeleteAllConversations={deleteAllConversations}
          recentQueries={recentQueries || []}
          dbSchema={dbSchema || {}}
          dashboardStats={dashboardStats || {}}
        />
        <div
          className="absolute top-0 -right-3 h-full w-3 cursor-ew-resize z-10"
          onMouseDown={handleMouseDown}
        ></div>
      </div>

      {/* 채팅 컨테이너, SQL 쿼리 페이지 또는 대시보드 */}
      <div className="flex-grow relative overflow-hidden">
        {mode === 'sql' ? (
          // SQL 쿼리 페이지 렌더링
          <div className="w-full h-full">
            <SQLQueryPage 
              setMode={setMode} 
              key="sql-page-component" 
            />
          </div>
        ) : mode === 'dashboard' ? (
          // 대시보드 렌더링
          <div className="w-full h-full">
            <Dashboard 
              setMode={setMode} 
              key="dashboard-component" 
            />
          </div>
        ) : (
          // 챗봇 컨테이너 렌더링
          <div className="w-full h-full">
            {!sidebarOpen && (
              <button
                className="absolute left-4 top-4 z-10 md:hidden p-2 rounded-full bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
                onClick={handleToggleSidebar}
              >
                <FiMenu size={20} />
              </button>
            )}
            <ChatContainer
              scrollLocked={scrollLocked}
              activeConversationId={activeConversationId}
              messages={currentMessages}
              searchTerm={searchTerm}
              filteredMessages={filteredMessages}
              onUpdateMessages={handleUpdateMessages}
              isEmbedding={isEmbedding}
              onUploadSuccess={handleUploadSuccess}
              onNewConversation={handleNewConversation}
              fileManagerOpen={fileManagerOpen}
              setFileManagerOpen={setFileManagerOpen}
              sidebarOpen={sidebarOpen}
              setMode={setMode}
              currentMode={mode}
              isStreaming={isStreaming}
              setIsStreaming={setIsStreaming}
              onStopGeneration={handleStopGeneration}
              onViewImages={handleViewImages}
              key="chat-container-component"
            />
          </div>
        )}
      </div>
      
      {/* 이미지 모달 */}
      <ImageModal
        isOpen={imageModalOpen}
        onClose={handleCloseImageModal}
        source={selectedImageSource}
      />
    </div>
  );
}

export default App;
