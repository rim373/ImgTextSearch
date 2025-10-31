const API = "http://localhost:8000";
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const els = {
  modeBtns: $$('.mode-btn'),
  imageSection: $('#imageSection'),
  textSection: $('#textSection'),
  dropZone: $('#dropZone'),
  fileInput: $('#fileInput'),
  preview: $('#preview'),
  textInput: $('#textInput'),
  topKInput: $('#topKInput'),
  searchBtn: $('#searchBtn'),
  spinner: $('#spinner'),
  resultsContainer: $('#resultsContainer'),
  results: $('#results'),
  resultsCount: $('#resultsCount'),
  modal: $('#modal'),
  modalImg: $('#modalImg'),
  modalClose: $('#modalClose')
};

let selectedFile = null;
let currentMode = 'image';
let abortController = null;

// Mode Toggle
els.modeBtns.forEach(btn => {
  btn.onclick = () => {
    els.modeBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    
    if (currentMode === 'image') {
      els.imageSection.classList.add('active');
      els.textSection.classList.remove('active');
    } else {
      els.textSection.classList.add('active');
      els.imageSection.classList.remove('active');
    }
  };
});

// Drop Zone
els.dropZone.onclick = () => els.fileInput.click();
els.dropZone.ondragover = (e) => {
  e.preventDefault();
  els.dropZone.classList.add('drag-over');
};
els.dropZone.ondragleave = () => els.dropZone.classList.remove('drag-over');
els.dropZone.ondrop = (e) => {
  e.preventDefault();
  els.dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file?.type.startsWith('image/')) handleFile(file);
};

els.fileInput.onchange = (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
};

const handleFile = (file) => {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    els.preview.innerHTML = `<img src="${reader.result}" alt="preview"><br><button class="clear-btn" onclick="clearImage()">✕ Clear</button>`;
  };
  reader.readAsDataURL(file);
};

window.clearImage = () => {
  selectedFile = null;
  els.preview.innerHTML = '';
  els.fileInput.value = '';
};

// Search
els.searchBtn.onclick = async () => {
  if (abortController) abortController.abort();
  abortController = new AbortController();

  els.results.innerHTML = '';
  els.resultsContainer.style.display = 'none';
  els.spinner.classList.add('active');
  els.searchBtn.disabled = true;

  try {
    const topK = parseInt(els.topKInput?.value) || 10;
    const formData = new FormData();
    formData.append("top_k", topK);

    let endpoint;
    if (currentMode === 'image') {
      if (!selectedFile) { alert('⚠️ Please select an image!'); return; }
      formData.append("file", selectedFile);
      endpoint = '/search/image';
    } else {
      const text = els.textInput.value.trim();
      if (!text) { alert('⚠️ Please enter a query!'); return; }
      formData.append("query_text", text);
      endpoint = '/search/text';
    }

    const res = await fetch(API + endpoint, { method: "POST", body: formData, signal: abortController.signal });
    const data = await res.json();

    if (!data.results?.length) {
      els.resultsContainer.style.display = 'block';
      els.results.innerHTML = '<div class="no-results">No results found!</div>';
      return;
    }

    els.resultsContainer.style.display = 'block';
    els.resultsCount.textContent = `Found ${data.results.length} images`;

    const fragment = document.createDocumentFragment();
    data.results.forEach((r, i) => {
      const card = document.createElement('div');
      card.className = 'result-card';
      const imgSrc = API + "/images/" + r.path;
      card.innerHTML = `<img src="${imgSrc}" alt="r${i}" loading="lazy"><div class="result-info"><div class="result-score">#${i + 1}</div></div>`;
      card.onclick = () => { els.modalImg.src = imgSrc; els.modal.classList.add('active'); };
      fragment.appendChild(card);
    });
    els.results.appendChild(fragment);

  } catch (err) {
    if (err.name !== 'AbortError') alert('❌ Error: ' + err.message);
  } finally {
    els.spinner.classList.remove('active');
    els.searchBtn.disabled = false;
  }
};

// Modal
els.modalClose.onclick = () => els.modal.classList.remove('active');
els.modal.onclick = (e) => { if (e.target === els.modal) els.modal.classList.remove('active'); };

// Keyboard shortcuts
els.textInput.onkeydown = (e) => { if (e.ctrlKey && e.key === 'Enter') els.searchBtn.click(); };
