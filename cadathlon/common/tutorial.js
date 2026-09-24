/* The tutorial's theme/menu behavior, with offline and keyboard support. */
(() => {
  const root = document.documentElement;
  const themeButton = document.getElementById('theme-toggle');
  const menuButton = document.getElementById('mobile-menu-btn');
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.querySelector('.menu-backdrop');
  const mobile = window.matchMedia('(max-width: 768px)');
  const storageKey = 'cross_cadathlon_theme';
  function applyTheme(theme) {
    root.dataset.theme = theme;
    themeButton.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
  }
  let saved;
  try { saved = localStorage.getItem(storageKey); } catch (_) { /* file:// may block storage */ }
  applyTheme(['dark', 'light'].includes(saved) ? saved :
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  themeButton.addEventListener('click', () => {
    const theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    applyTheme(theme);
    try { localStorage.setItem(storageKey, theme); } catch (_) { /* preference is optional */ }
  });
  function setMenu(open) {
    if (mobile.matches) {
      document.body.classList.remove('sidebar-hidden');
      sidebar.classList.toggle('open', open);
      backdrop.hidden = !open;
    } else {
      sidebar.classList.remove('open');
      document.body.classList.toggle('sidebar-hidden', !open);
      backdrop.hidden = true;
    }
    sidebar.inert = !open;
    menuButton.setAttribute('aria-expanded', String(open));
  }
  menuButton.addEventListener('click', () => setMenu(menuButton.getAttribute('aria-expanded') !== 'true'));
  document.getElementById('mobile-close-btn').addEventListener('click', () => { setMenu(false); menuButton.focus(); });
  backdrop.addEventListener('click', () => setMenu(false));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && mobile.matches) { setMenu(false); menuButton.focus(); }
  });
  sidebar.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
    if (mobile.matches) setMenu(false);
  }));
  mobile.addEventListener('change', () => setMenu(!mobile.matches));
  setMenu(!mobile.matches);
  const anchors = [...document.querySelectorAll('.doc-content section[id], .doc-content h3[id]')];
  const links = [...document.querySelectorAll('.on-this-page a, .sidebar a[href^="#"]')];
  let scheduled = false;
  function markSection() {
    scheduled = false;
    let current = anchors[0];
    for (const anchor of anchors) {
      if (anchor.getBoundingClientRect().top <= 125) current = anchor;
    }
    links.forEach(link => {
      const section = current.closest('section');
      const active = link.hash === `#${current.id}` ||
        (link.closest('.sidebar') && link.hash === `#${section.id}`);
      link.classList.toggle('active', Boolean(active));
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  window.addEventListener('scroll', () => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(markSection); }
  }, { passive: true });
  markSection();
})();
