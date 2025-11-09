/**
 * Multi-language support JavaScript
 * Handles smooth language switching and URL management
 */

(function() {
    'use strict';

    // Language switching functionality
    function initLanguageSwitch() {
        const languageLinks = document.querySelectorAll('.language-switch');
        
        languageLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Add loading state
                const contentWrapper = document.querySelector('.content');
                if (contentWrapper) {
                    contentWrapper.classList.add('switching');
                }
                
                // Show notification for language switch
                showLanguageSwitchNotification();
                
                // Smooth transition
                setTimeout(() => {
                    window.location.href = this.href;
                }, 300);
            });
        });
    }
    
    // Show notification when switching languages
    function showLanguageSwitchNotification() {
        const currentLang = getCurrentLanguage();
        const targetLang = currentLang === 'en' ? 'vi' : 'en';
        const message = currentLang === 'en' ? 
            'Đang chuyển sang tiếng Việt...' : 
            'Switching to English...';
            
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'language-switch-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #268bd2;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 1000;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Remove notification after animation
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 2000);
    }

    // Detect current language from URL or page attributes
    function getCurrentLanguage() {
        const htmlLang = document.documentElement.lang;
        if (htmlLang && htmlLang !== 'en-us') {
            return htmlLang;
        }
        
        const path = window.location.pathname;
        if (path.includes('/contents/vi/')) {
            return 'vi';
        } else if (path.includes('/contents/en/')) {
            return 'en';
        }
        
        return 'en'; // default
    }

    // Update page language attributes
    function updateLanguageAttributes() {
        const currentLang = getCurrentLanguage();
        document.documentElement.lang = currentLang;
        
        // Update meta tags if needed
        const metaLang = document.querySelector('meta[http-equiv="content-language"]');
        if (metaLang) {
            metaLang.setAttribute('content', currentLang);
        } else {
            const meta = document.createElement('meta');
            meta.setAttribute('http-equiv', 'content-language');
            meta.setAttribute('content', currentLang);
            document.head.appendChild(meta);
        }
    }

    // Add language indicator to posts
    function addLanguageIndicators() {
        const posts = document.querySelectorAll('.post');
        const currentLang = getCurrentLanguage();
        
        posts.forEach(post => {
            const existingIndicator = post.querySelector('.post-language');
            if (!existingIndicator) {
                const indicator = document.createElement('span');
                indicator.className = `post-language lang-${currentLang}`;
                indicator.textContent = currentLang === 'vi' ? 'Tiếng Việt' : 'English';
                
                const postHeader = post.querySelector('.post-title') || post.querySelector('h1') || post.querySelector('h2');
                if (postHeader) {
                    postHeader.parentNode.insertBefore(indicator, postHeader);
                }
            }
        });
    }

    // Keyboard navigation for language switching
    function initKeyboardNavigation() {
        document.addEventListener('keydown', function(e) {
            // Alt + L to switch language
            if (e.altKey && e.key === 'l') {
                e.preventDefault();
                const languageLink = document.querySelector('.language-switch');
                if (languageLink) {
                    languageLink.click();
                }
            }
        });
    }

    // Store user's language preference
    function storeLanguagePreference() {
        const currentLang = getCurrentLanguage();
        localStorage.setItem('preferred-language', currentLang);
    }

    // Load user's language preference
    function loadLanguagePreference() {
        const preferredLang = localStorage.getItem('preferred-language');
        const currentLang = getCurrentLanguage();
        
        // If user has a preference and it's different from current, suggest switching
        if (preferredLang && preferredLang !== currentLang) {
            const currentPath = window.location.pathname;
            let newPath;
            
            if (preferredLang === 'vi' && currentPath.includes('/contents/en/')) {
                newPath = currentPath.replace('/contents/en/', '/contents/vi/');
            } else if (preferredLang === 'en' && currentPath.includes('/contents/vi/')) {
                newPath = currentPath.replace('/contents/vi/', '/contents/en/');
            }
            
            // Optionally auto-redirect or show a notification
            // For now, we'll just store the preference
        }
    }

    // Add accessibility improvements
    function enhanceAccessibility() {
        const languageLinks = document.querySelectorAll('.language-switch');
        
        languageLinks.forEach(link => {
            // Add ARIA labels
            const currentLang = getCurrentLanguage();
            const targetLang = currentLang === 'en' ? 'vi' : 'en';
            const langName = targetLang === 'vi' ? 'Vietnamese' : 'English';
            
            link.setAttribute('aria-label', `Switch to ${langName}`);
            link.setAttribute('title', `Switch to ${langName}`);
            
            // Add keyboard support
            link.setAttribute('tabindex', '0');
            link.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.click();
                }
            });
        });
    }

    // Initialize all functionality when DOM is ready
    function init() {
        updateLanguageAttributes();
        initLanguageSwitch();
        addLanguageIndicators();
        initKeyboardNavigation();
        enhanceAccessibility();
        storeLanguagePreference();
        loadLanguagePreference();
        
        // Remove loading state if it exists
        setTimeout(() => {
            const contentWrapper = document.querySelector('.content');
            if (contentWrapper) {
                contentWrapper.classList.remove('switching');
            }
        }, 100);
    }

    // Run when DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Expose some functions globally for debugging
    window.MultiLang = {
        getCurrentLanguage: getCurrentLanguage,
        init: init
    };

})();
