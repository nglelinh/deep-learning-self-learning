/**
 * Search functionality sử dụng Lunr.js
 * Hỗ trợ tìm kiếm đa ngôn ngữ và hiển thị kết quả theo thời gian thực
 */

class OptimizationSearch {
  constructor() {
    this.searchIndex = null;
    this.searchData = [];
    this.currentLang = this.detectLanguage();
    this.isInitialized = false;
    
    // DOM elements
    this.searchInput = null;
    this.searchResults = null;
    this.searchContainer = null;
    this.searchOverlay = null;
    
    this.init();
  }

  /**
   * Phát hiện ngôn ngữ hiện tại từ URL hoặc config
   */
  detectLanguage() {
    const path = window.location.pathname;
    if (path.includes('/vi/')) return 'vi';
    if (path.includes('/en/')) return 'en';
    
    // Mặc định là tiếng Anh
    return 'en';
  }

  /**
   * Khởi tạo search functionality
   */
  async init() {
    try {
      await this.loadSearchData();
      this.createSearchIndex();
      this.createSearchUI();
      this.bindEvents();
      this.isInitialized = true;
      console.log('🔍 Search đã được khởi tạo thành công');
    } catch (error) {
      console.error('❌ Lỗi khởi tạo search:', error);
    }
  }

  /**
   * Tải dữ liệu search từ JSON file
   */
  async loadSearchData() {
    const baseUrl = window.location.origin + (window.location.pathname.includes('/optimization-for-data-science-iuh-2025') 
      ? '/optimization-for-data-science-iuh-2025' : '');
    
    const filename = this.currentLang === 'en' ? 'search-index.json' : `search-index-${this.currentLang}.json`;
    const url = `${baseUrl}/${filename}`;
    
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      this.searchData = await response.json();
      console.log(`📚 Đã tải ${this.searchData.length} mục cho ngôn ngữ ${this.currentLang}`);
    } catch (error) {
      console.error('❌ Không thể tải dữ liệu search:', error);
      this.searchData = [];
    }
  }

  /**
   * Tạo Lunr search index
   */
  createSearchIndex() {
    if (this.searchData.length === 0) {
      console.warn('⚠️ Không có dữ liệu để tạo search index');
      return;
    }

    const searchData = this.searchData; // Lưu reference
    const currentLang = this.currentLang; // Lưu reference

    this.searchIndex = lunr(function() {
      // Cấu hình cho tiếng Việt
      if (currentLang === 'vi') {
        // Lunr không hỗ trợ tiếng Việt native, nhưng ta có thể tùy chỉnh
        this.pipeline.remove(lunr.stemmer);
        this.searchPipeline.remove(lunr.stemmer);
      }
      
      // Định nghĩa các trường có thể tìm kiếm
      this.ref('id');
      this.field('title', { boost: 10 });      // Tiêu đề có trọng số cao nhất
      this.field('content', { boost: 5 });     // Nội dung có trọng số cao
      this.field('chapter', { boost: 3 });     // Chapter có trọng số trung bình
      this.field('owner', { boost: 2 });       // Tác giả có trọng số thấp hơn
      this.field('lesson_type');               // Loại bài học
      
      // Thêm tất cả documents vào index
      searchData.forEach(doc => {
        this.add(doc);
      });
    });

    console.log('🔍 Search index đã được tạo thành công');
  }

  /**
   * Tạo giao diện tìm kiếm
   */
  createSearchUI() {
    // Tạo search container
    this.searchContainer = document.createElement('div');
    this.searchContainer.className = 'search-container';
    this.searchContainer.innerHTML = `
      <div class="search-box">
        <div class="search-input-wrapper">
          <input type="text" 
                 class="search-input" 
                 placeholder="${this.currentLang === 'vi' ? 'Tìm kiếm nội dung...' : 'Search content...'}"
                 autocomplete="off">
          <button class="search-clear" title="${this.currentLang === 'vi' ? 'Xóa' : 'Clear'}">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
        <div class="search-results"></div>
      </div>
    `;

    // Tạo overlay
    this.searchOverlay = document.createElement('div');
    this.searchOverlay.className = 'search-overlay';
    this.searchOverlay.appendChild(this.searchContainer);
    
    // Thêm vào body
    document.body.appendChild(this.searchOverlay);

    // Lưu references
    this.searchInput = this.searchContainer.querySelector('.search-input');
    this.searchResults = this.searchContainer.querySelector('.search-results');

    // Thêm search button vào header (với retry logic)
    this.addSearchButton();
  }

  /**
   * Thêm search button vào header với retry logic
   */
  addSearchButton() {
    const addButton = () => {
      const headerActions = document.querySelector('.header-actions');
      if (headerActions && !document.querySelector('.search-toggle')) {
        const searchButton = document.createElement('button');
        searchButton.className = 'search-toggle';
        searchButton.innerHTML = `
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
          </svg>
        `;
        searchButton.title = this.currentLang === 'vi' ? 'Tìm kiếm' : 'Search';
        
        // Thêm search button trước language toggle
        headerActions.insertBefore(searchButton, headerActions.firstChild);
        
        console.log('✅ Search button đã được thêm vào header');
        return true;
      }
      return false;
    };

    // Thử thêm ngay lập tức
    if (!addButton()) {
      // Nếu không thành công, thử lại sau 500ms
      setTimeout(() => {
        if (!addButton()) {
          console.warn('⚠️ Không thể tìm thấy .header-actions để thêm search button');
        }
      }, 500);
    }
  }

  /**
   * Bind events cho search UI
   */
  bindEvents() {
    // Bind search toggle với event delegation
    document.addEventListener('click', (e) => {
      if (e.target.closest('.search-toggle')) {
        e.preventDefault();
        this.toggleSearch();
      }
    });

    // Clear search
    const searchClear = this.searchContainer.querySelector('.search-clear');
    if (searchClear) {
      searchClear.addEventListener('click', () => {
        this.clearSearch();
      });
    }

    // Close overlay khi click outside
    this.searchOverlay.addEventListener('click', (e) => {
      if (e.target === this.searchOverlay) {
        this.closeSearch();
      }
    });

    // Search input events
    if (this.searchInput) {
      // Debounced search
      let searchTimeout;
      this.searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
          this.performSearch(e.target.value);
        }, 300);
      });

      // Keyboard navigation
      this.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.closeSearch();
        } else if (e.key === 'Enter') {
          e.preventDefault();
          this.selectFirstResult();
        }
      });
    }

    // Keyboard shortcut (Ctrl/Cmd + K)
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        this.toggleSearch();
      }
    });

    console.log('✅ Search events đã được bind');
  }

  /**
   * Toggle search overlay
   */
  toggleSearch() {
    if (this.searchOverlay.classList.contains('active')) {
      this.closeSearch();
    } else {
      this.openSearch();
    }
  }

  /**
   * Mở search overlay
   */
  openSearch() {
    this.searchOverlay.classList.add('active');
    document.body.classList.add('search-active');
    
    // Focus vào search input
    setTimeout(() => {
      if (this.searchInput) {
        this.searchInput.focus();
      }
    }, 100);
  }

  /**
   * Đóng search overlay
   */
  closeSearch() {
    this.searchOverlay.classList.remove('active');
    document.body.classList.remove('search-active');
    this.clearSearch();
  }

  /**
   * Xóa search input và kết quả
   */
  clearSearch() {
    if (this.searchInput) {
      this.searchInput.value = '';
    }
    if (this.searchResults) {
      this.searchResults.innerHTML = '';
    }
  }

  /**
   * Thực hiện tìm kiếm
   */
  performSearch(query) {
    if (!this.searchIndex || !query.trim()) {
      this.searchResults.innerHTML = '';
      return;
    }

    try {
      // Thực hiện search với Lunr
      const results = this.searchIndex.search(query);
      
      // Lấy thông tin chi tiết cho mỗi kết quả
      const detailedResults = results.map(result => {
        const doc = this.searchData.find(item => item.id === result.ref);
        return {
          ...doc,
          score: result.score,
          matches: result.matchData
        };
      });

      this.displayResults(detailedResults, query);
    } catch (error) {
      console.error('❌ Lỗi khi tìm kiếm:', error);
      this.displayError();
    }
  }

  /**
   * Hiển thị kết quả tìm kiếm
   */
  displayResults(results, query) {
    if (results.length === 0) {
      this.searchResults.innerHTML = `
        <div class="search-no-results">
          <p>${this.currentLang === 'vi' ? 'Không tìm thấy kết quả nào' : 'No results found'}</p>
          <small>${this.currentLang === 'vi' ? 'Thử với từ khóa khác' : 'Try different keywords'}</small>
        </div>
      `;
      return;
    }

    const resultsHTML = results.map(result => {
      // Highlight matching text
      const highlightedTitle = this.highlightText(result.title, query);
      const highlightedContent = this.highlightText(
        this.truncateText(result.content, 150), 
        query
      );

      return `
        <div class="search-result-item" data-url="${result.url}">
          <div class="search-result-header">
            <h4 class="search-result-title">${highlightedTitle}</h4>
            <span class="search-result-chapter">Chapter ${result.chapter}</span>
          </div>
          <p class="search-result-content">${highlightedContent}</p>
          <div class="search-result-meta">
            ${result.owner ? `<span class="search-result-owner">${result.owner}</span>` : ''}
            ${result.lesson_type ? `<span class="search-result-type">${result.lesson_type}</span>` : ''}
          </div>
        </div>
      `;
    }).join('');

    this.searchResults.innerHTML = `
      <div class="search-results-header">
        <span class="search-results-count">
          ${results.length} ${this.currentLang === 'vi' ? 'kết quả' : 'results'}
        </span>
      </div>
      <div class="search-results-list">
        ${resultsHTML}
      </div>
    `;

    // Bind click events cho kết quả
    this.bindResultEvents();
  }

  /**
   * Hiển thị lỗi
   */
  displayError() {
    this.searchResults.innerHTML = `
      <div class="search-error">
        <p>${this.currentLang === 'vi' ? 'Có lỗi xảy ra khi tìm kiếm' : 'An error occurred while searching'}</p>
      </div>
    `;
  }

  /**
   * Bind events cho search results
   */
  bindResultEvents() {
    const resultItems = this.searchResults.querySelectorAll('.search-result-item');
    
    resultItems.forEach(item => {
      item.addEventListener('click', () => {
        const url = item.dataset.url;
        if (url) {
          // Thêm baseurl nếu chưa có
          const finalUrl = this.addBaseUrl(url);
          window.location.href = finalUrl;
        }
      });

      // Keyboard navigation
      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          item.click();
        }
      });
    });
  }

  /**
   * Thêm baseurl vào URL nếu cần
   */
  addBaseUrl(url) {
    // Detect baseurl từ current URL
    const currentPath = window.location.pathname;
    let baseUrl = '';
    
    // Nếu current URL có baseurl pattern
    if (currentPath.includes('/optimization-for-data-science-iuh-2025')) {
      baseUrl = '/optimization-for-data-science-iuh-2025';
    }
    
    // Kiểm tra xem URL đã có baseurl chưa
    if (baseUrl && url.startsWith(baseUrl)) {
      return url; // Đã có baseurl
    }
    
    // Thêm baseurl nếu cần
    return baseUrl + url;
  }

  /**
   * Chọn kết quả đầu tiên
   */
  selectFirstResult() {
    const firstResult = this.searchResults.querySelector('.search-result-item');
    if (firstResult) {
      firstResult.click();
    }
  }

  /**
   * Highlight text matching query
   */
  highlightText(text, query) {
    if (!query || !text) return text;
    
    const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  }

  /**
   * Truncate text to specified length
   */
  truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    
    return text.substring(0, maxLength).trim() + '...';
  }

  /**
   * Escape regex special characters
   */
  escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}

// Global variable để access search instance
window.optimizationSearch = null;

// Khởi tạo search khi DOM ready và Lunr.js đã load
document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 DOM loaded, initializing search...');
  
  // Kiểm tra xem Lunr.js đã được load chưa
  if (typeof lunr !== 'undefined') {
    console.log('✅ Lunr.js detected');
    try {
      window.optimizationSearch = new OptimizationSearch();
      console.log('✅ OptimizationSearch instance created');
    } catch (error) {
      console.error('❌ Lỗi tạo OptimizationSearch instance:', error);
    }
  } else {
    console.error('❌ Lunr.js chưa được load. Vui lòng thêm Lunr.js vào trang.');
    
    // Thử load lại sau 1 giây
    setTimeout(() => {
      if (typeof lunr !== 'undefined') {
        console.log('✅ Lunr.js loaded after retry');
        window.optimizationSearch = new OptimizationSearch();
      } else {
        console.error('❌ Lunr.js vẫn chưa được load sau retry');
      }
    }, 1000);
  }
});

// Debug function để test search
window.testSearch = () => {
  if (window.optimizationSearch) {
    window.optimizationSearch.openSearch();
    console.log('🔍 Search opened via testSearch()');
  } else {
    console.error('❌ Search instance not found');
  }
};
