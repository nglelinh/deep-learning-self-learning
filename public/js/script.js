(function(document) {
  var toggle = document.querySelector('.sidebar-toggle');
  var sidebar = document.querySelector('#sidebar');
  var checkbox = document.querySelector('#sidebar-checkbox');

  if (checkbox && window.innerWidth > 1480) {
    checkbox.checked = true;
  }

  // Close the Lanyon drawer when clicking outside (unchanged behavior)
  document.addEventListener('click', function(e) {
    if (!checkbox || !sidebar) return;

    var target = e.target;

    if (!checkbox.checked ||
        sidebar.contains(target) ||
        (target === checkbox || target === toggle)) return;

    checkbox.checked = false;
  }, false);

  // --- Collapsible chapter + section lesson submenus ---
  var CHAPTER_KEY = 'sidebar-open-chapters';
  var SECTION_KEY = 'sidebar-open-sections';

  function readJsonArray(key) {
    try {
      var raw = sessionStorage.getItem(key);
      if (!raw) return [];
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch (err) {
      return [];
    }
  }

  function writeJsonArray(key, values) {
    try {
      sessionStorage.setItem(key, JSON.stringify(values));
    } catch (err) {
      // ignore quota / private mode
    }
  }

  function setOpen(el, open, btn, list) {
    if (!el || !list) return;
    el.classList.toggle('is-open', open);
    if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
      list.removeAttribute('hidden');
    } else {
      list.setAttribute('hidden', '');
    }
  }

  function setChapterOpen(chapterEl, open) {
    if (!chapterEl) return;
    var btn = chapterEl.querySelector(':scope > .sidebar-chapter-header > .sidebar-chapter-btn');
    var list = chapterEl.querySelector(':scope > .sidebar-lesson-list');
    setOpen(chapterEl, open, btn, list);
  }

  function setSectionOpen(sectionEl, open) {
    if (!sectionEl) return;
    var btn = sectionEl.querySelector(':scope > .sidebar-section-header > .sidebar-section-btn');
    var list = sectionEl.querySelector(':scope > .sidebar-section-list');
    setOpen(sectionEl, open, btn, list);
  }

  function persistChapterState() {
    var open = [];
    document.querySelectorAll('.sidebar-chapter.is-open').forEach(function(el) {
      var ch = el.getAttribute('data-chapter');
      if (ch != null) open.push(String(ch));
    });
    writeJsonArray(CHAPTER_KEY, open);
  }

  function persistSectionState() {
    var open = [];
    document.querySelectorAll('.sidebar-section.is-open').forEach(function(el) {
      var code = el.getAttribute('data-section');
      if (code) open.push(String(code));
    });
    writeJsonArray(SECTION_KEY, open);
  }

  function hasActive(entry) {
    if (!entry) return false;
    if (entry.el && entry.el.classList.contains('active')) return true;
    for (var i = 0; i < entry.children.length; i++) {
      if (hasActive(entry.children[i])) return true;
    }
    return false;
  }

  function buildTree(nodes) {
    var root = [];
    var stack = [{ depth: -1, children: root }];

    nodes.forEach(function(node) {
      var depth = parseInt(node.getAttribute('data-depth') || '0', 10);
      if (isNaN(depth)) depth = 0;

      while (stack.length > 1 && stack[stack.length - 1].depth >= depth) {
        stack.pop();
      }

      var entry = {
        el: node,
        depth: depth,
        code: node.getAttribute('data-code') || '',
        children: []
      };
      stack[stack.length - 1].children.push(entry);
      stack.push(entry);
    });

    return root;
  }

  function renderTree(entries, container, storedSections) {
    entries.forEach(function(entry) {
      if (!entry.children.length) {
        container.appendChild(entry.el);
        return;
      }

      // Depth-0 items (e.g. "04 Introduction") should not wrap the whole
      // chapter into one section — emit the link, then children as peers.
      if (entry.depth < 1) {
        container.appendChild(entry.el);
        renderTree(entry.children, container, storedSections);
        return;
      }

      var section = document.createElement('div');
      section.className = 'sidebar-section';
      section.setAttribute('data-section', entry.code);
      section.setAttribute('data-depth', String(entry.depth));
      section.style.setProperty('--depth', String(entry.depth));

      var header = document.createElement('div');
      header.className = 'sidebar-section-header';

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'sidebar-collapse-btn sidebar-section-btn';
      btn.setAttribute('aria-controls', 'sidebar-section-' + entry.code.replace(/[^a-zA-Z0-9_-]/g, '-'));
      btn.title = 'Expand/collapse section';

      var icon = document.createElement('span');
      icon.className = 'sidebar-collapse-icon';
      icon.setAttribute('aria-hidden', 'true');
      btn.appendChild(icon);

      // Keep the original lesson link as the section title
      entry.el.classList.add('sidebar-section-link');
      header.appendChild(btn);
      header.appendChild(entry.el);

      var childList = document.createElement('div');
      childList.className = 'sidebar-section-list';
      childList.id = btn.getAttribute('aria-controls');

      section.appendChild(header);
      section.appendChild(childList);
      container.appendChild(section);

      renderTree(entry.children, childList, storedSections);

      var openByActive = hasActive(entry);
      var openByStore = storedSections.indexOf(String(entry.code)) !== -1;
      var shouldOpen = openByActive || openByStore;
      setSectionOpen(section, shouldOpen);

      btn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        var open = !section.classList.contains('is-open');
        setSectionOpen(section, open);
        persistSectionState();
      });
    });
  }

  function nestLessonSections() {
    var lists = document.querySelectorAll('.sidebar-lesson-list[data-nest-sections="true"]');
    var storedSections = readJsonArray(SECTION_KEY);

    lists.forEach(function(list) {
      if (list.getAttribute('data-nested') === '1') return;

      var lessons = Array.prototype.slice.call(
        list.querySelectorAll(':scope > a.sidebar-lesson')
      );
      if (!lessons.length) return;

      var tree = buildTree(lessons);
      // Clear flat list and re-render nested structure
      while (list.firstChild) list.removeChild(list.firstChild);
      renderTree(tree, list, storedSections);
      list.setAttribute('data-nested', '1');
    });
  }

  function initChapterMenus() {
    nestLessonSections();

    var chapters = document.querySelectorAll('.sidebar-chapter');
    if (!chapters.length) return;

    var storedChapters = readJsonArray(CHAPTER_KEY);
    var hasStored = storedChapters.length > 0;

    chapters.forEach(function(chapterEl) {
      var ch = String(chapterEl.getAttribute('data-chapter') || '');
      var list = chapterEl.querySelector(':scope > .sidebar-lesson-list');
      var btn = chapterEl.querySelector(':scope > .sidebar-chapter-header > .sidebar-chapter-btn');
      if (!list || !btn) return;

      var shouldOpen = chapterEl.classList.contains('is-open');
      if (hasStored) {
        shouldOpen = storedChapters.indexOf(ch) !== -1 || chapterEl.classList.contains('is-open');
      }
      setChapterOpen(chapterEl, shouldOpen);

      btn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        var open = !chapterEl.classList.contains('is-open');
        setChapterOpen(chapterEl, open);
        persistChapterState();
      });
    });

    // Ensure ancestors of the active lesson are open
    var activeLesson = document.querySelector('.sidebar-lesson.active');
    if (activeLesson) {
      var chapter = activeLesson.closest('.sidebar-chapter');
      if (chapter) {
        setChapterOpen(chapter, true);
      }
      var section = activeLesson.closest('.sidebar-section');
      while (section) {
        setSectionOpen(section, true);
        section = section.parentElement ? section.parentElement.closest('.sidebar-section') : null;
      }
      persistChapterState();
      persistSectionState();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initChapterMenus);
  } else {
    initChapterMenus();
  }
})(document);
