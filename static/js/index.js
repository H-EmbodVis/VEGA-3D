const scrollTopButton = document.getElementById('scrollTop');

const toggleScrollTop = () => {
  if (window.scrollY > 320) {
    scrollTopButton.classList.add('visible');
  } else {
    scrollTopButton.classList.remove('visible');
  }
};

scrollTopButton.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
      }
    });
  },
  { threshold: 0.15 }
);

document.querySelectorAll('.reveal').forEach((node) => observer.observe(node));
window.addEventListener('scroll', toggleScrollTop);
toggleScrollTop();


const copyCitationButton = document.getElementById('copyCitation');
const citationCode = document.getElementById('citationCode');
const copyStatus = document.getElementById('copyStatus');

if (copyCitationButton && citationCode && copyStatus) {
  const showCopySuccess = () => {
    copyCitationButton.textContent = 'Copied';
    copyCitationButton.classList.add('copied');
    copyStatus.textContent = 'Citation copied to clipboard.';
    window.setTimeout(() => {
      copyCitationButton.textContent = 'Copy';
      copyCitationButton.classList.remove('copied');
      copyStatus.textContent = '';
    }, 2200);
  };

  const fallbackCopy = () => {
    const tempTextarea = document.createElement('textarea');
    tempTextarea.value = citationCode.textContent;
    tempTextarea.setAttribute('readonly', '');
    tempTextarea.style.position = 'absolute';
    tempTextarea.style.left = '-9999px';
    document.body.appendChild(tempTextarea);
    tempTextarea.select();
    const copied = document.execCommand('copy');
    document.body.removeChild(tempTextarea);
    return copied;
  };

  copyCitationButton.addEventListener('click', async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(citationCode.textContent);
        showCopySuccess();
        return;
      }

      if (fallbackCopy()) {
        showCopySuccess();
        return;
      }

      copyStatus.textContent = 'Clipboard copy failed. Please copy manually.';
    } catch (error) {
      copyStatus.textContent = fallbackCopy()
        ? 'Citation copied to clipboard.'
        : 'Clipboard copy failed. Please copy manually.';
      if (copyStatus.textContent === 'Citation copied to clipboard.') {
        showCopySuccess();
      }
    }
  });
}
