// DOM Elements
const apiUrlInput = document.getElementById('apiUrlInput');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const badgeFileCount = document.getElementById('badgeFileCount');
const anglesList = document.getElementById('anglesList');
const clearFilesLink = document.getElementById('clearFilesLink');
const btnPredict = document.getElementById('btnPredict');
const btnReset = document.getElementById('btnReset');
const btnSpinner = document.getElementById('btnSpinner');
const btnText = document.getElementById('btnText');

// Hero Card DOM
const sortingActionBadge = document.getElementById('sortingActionBadge');
const gradeReasonText = document.getElementById('gradeReasonText');
const gradeStandardRef = document.getElementById('gradeStandardRef');
const gradeBadgeHuge = document.getElementById('gradeBadgeHuge');

// Module 1: YOLO 360 DOM
const valTotalAngles = document.getElementById('valTotalAngles');
const valYolo360Class = document.getElementById('valYolo360Class');
const valYolo360Conf = document.getElementById('valYolo360Conf');
const valYolo360Notes = document.getElementById('valYolo360Notes');

// Module 2: Geometry DOM
const valDimensions = document.getElementById('valDimensions');
const valEsi = document.getElementById('valEsi');
const valWeight = document.getElementById('valWeight');
const valSizeCat = document.getElementById('valSizeCat');

// Raw JSON
const rawJsonResponse = document.getElementById('rawJsonResponse');
const toast = document.getElementById('toast');

let selectedFiles = [];
let loadedImageBitmaps = [];

function showToast(message, type = 'info') {
  toast.textContent = message;
  toast.className = `toast ${type} show`;
  setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
}

// Health Check
if (window.location.protocol.startsWith('http')) {
  apiUrlInput.value = window.location.origin;
}

async function checkApiHealth() {
  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  try {
    const res = await fetch(`${baseUrl}/api/v1/status`, { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      statusDot.className = "status-dot online";
      statusText.textContent = "API Terhubung (" + data.device + ")";
    } else {
      throw new Error("HTTP Status " + res.status);
    }
  } catch (err) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "API Tidak Aktif (Jalankan app_api.py)";
  }
}

checkApiHealth();
setInterval(checkApiHealth, 5000);
apiUrlInput.addEventListener('change', checkApiHealth);

// File Upload Handlers
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFilesSelected(Array.from(e.dataTransfer.files));
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFilesSelected(Array.from(e.target.files));
  }
});

clearFilesLink.addEventListener('click', (e) => {
  e.stopPropagation();
  resetFiles();
});

function handleFilesSelected(newFiles) {
  const validImageFiles = newFiles.filter(f => f.type.startsWith('image/'));
  if (validImageFiles.length === 0) {
    showToast('Harap pilih file citra yang valid (JPG/PNG)!', 'error');
    return;
  }

  selectedFiles = validImageFiles;
  loadedImageBitmaps = [];
  anglesList.innerHTML = '';

  badgeFileCount.textContent = `${selectedFiles.length} Sudut Dipilih`;
  clearFilesLink.style.display = 'inline';
  btnPredict.disabled = false;

  selectedFiles.forEach((file, index) => {
    const angleCard = document.createElement('div');
    angleCard.className = 'angle-card';
    angleCard.id = `angleCard_${index}`;

    const canvasWrap = document.createElement('div');
    canvasWrap.className = 'angle-canvas-wrap';

    const canvas = document.createElement('canvas');
    canvas.id = `canvas_${index}`;
    canvasWrap.appendChild(canvas);

    const infoDiv = document.createElement('div');
    infoDiv.className = 'angle-info';
    infoDiv.innerHTML = `
      <div class="angle-label" title="${file.name}">Sudut ${index + 1}: ${file.name}</div>
      <span class="angle-status" id="statusBadge_${index}" style="background: #e5e7eb; color: #4b5563;">Menunggu Prediksi</span>
    `;

    angleCard.appendChild(canvasWrap);
    angleCard.appendChild(infoDiv);
    anglesList.appendChild(angleCard);

    const reader = new FileReader();
    reader.onload = function(evt) {
      const img = new Image();
      img.onload = function() {
        loadedImageBitmaps[index] = img;
        drawInitialCanvas(canvas, img);
      };
      img.src = evt.target.result;
    };
    reader.readAsDataURL(file);
  });

  showToast(`Berhasil memuat ${selectedFiles.length} sudut citra telur!`, 'success');
}

function drawInitialCanvas(canvas, img) {
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
}

// Draw Bounding Boxes Overlay on a specific canvas
function drawDetectionsOnCanvas(canvas, img, detections) {
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  if (!detections || detections.length === 0) return;

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox_xyxy;
    const w = x2 - x1;
    const h = y2 - y1;
    const name = det.class_name;
    const conf = (det.confidence * 100).toFixed(1);

    let boxColor = '#10b981'; // Green for good
    if (name === 'cracked') boxColor = '#ef4444'; // Red
    if (name === 'dirt') boxColor = '#f59e0b'; // Amber

    ctx.lineWidth = Math.max(3, Math.round(canvas.width * 0.006));
    ctx.strokeStyle = boxColor;
    ctx.strokeRect(x1, y1, w, h);

    const label = `${name.toUpperCase()} ${conf}%`;
    const fontSize = Math.max(14, Math.round(canvas.width * 0.035));
    ctx.font = `bold ${fontSize}px sans-serif`;
    const textWidth = ctx.measureText(label).width;
    const padding = 6;

    ctx.fillStyle = boxColor;
    ctx.fillRect(x1, Math.max(0, y1 - fontSize - padding * 2), textWidth + padding * 2, fontSize + padding * 2);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x1 + padding, Math.max(fontSize, y1 - padding));
  });
}

// Predict Handler (Sends all angles in 1 request)
btnPredict.addEventListener('click', async () => {
  if (selectedFiles.length === 0) {
    showToast('Pilih minimal 1 citra sudut telur!', 'error');
    return;
  }

  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  btnPredict.disabled = true;
  btnSpinner.style.display = 'inline-block';
  btnText.textContent = 'Menganalisis 360°...';

  const formData = new FormData();
  selectedFiles.forEach(file => {
    formData.append('files', file);
  });

  try {
    const response = await fetch(`${baseUrl}/api/v1/predict`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errJson = await response.json();
      throw new Error(errJson.detail || 'Gagal memproses prediksi dari server.');
    }

    const data = await response.json();
    displayResults(data);
    showToast('Inspeksi 360° & Penentuan Mutu Selesai!', 'success');

  } catch (error) {
    console.error(error);
    showToast('Error: ' + error.message, 'error');
  } finally {
    btnPredict.disabled = false;
    btnSpinner.style.display = 'none';
    btnText.textContent = '⚡ Analisis 360° & Tentukan Grade';
  }
});

// Display Prediction Results
function displayResults(data) {
  rawJsonResponse.textContent = JSON.stringify(data, null, 2);

  // Update each angle card
  const angles = data.angles || [];
  angles.forEach((angleData, idx) => {
    const card = document.getElementById(`angleCard_${idx}`);
    const badge = document.getElementById(`statusBadge_${idx}`);
    const canvas = document.getElementById(`canvas_${idx}`);
    const img = loadedImageBitmaps[idx];

    if (canvas && img) {
      drawDetectionsOnCanvas(canvas, img, angleData.detections || []);
    }

    if (card && badge) {
      card.classList.remove('cracked-border', 'dirt-border', 'good-border');
      const cName = angleData.detected_class;
      const confPct = (angleData.confidence * 100).toFixed(1);

      if (cName === 'cracked') {
        card.classList.add('cracked-border');
        badge.className = 'angle-status pill-cracked';
        badge.textContent = `CRACKED (${confPct}%)`;
      } else if (cName === 'dirt') {
        card.classList.add('dirt-border');
        badge.className = 'angle-status pill-dirt';
        badge.textContent = `DIRT (${confPct}%)`;
      } else {
        card.classList.add('good-border');
        badge.className = 'angle-status pill-good';
        badge.textContent = `GOOD (${confPct}%)`;
      }
    }
  });

  // Final Grade Hero Card
  const final = data.final_grading_decision || {};
  const grade = final.grade || 'Tidak Diketahui';
  const gradeCode = final.grade_code || '';
  const reason = final.reason || '-';
  const action = final.hardware_sorting_action || '-';
  const standard = final.standard_reference || 'SNI 3926:2023';

  sortingActionBadge.textContent = '🎯 ' + action;
  gradeReasonText.textContent = reason;
  gradeStandardRef.textContent = 'Acuan: ' + standard;

  gradeBadgeHuge.style.display = 'inline-flex';
  gradeBadgeHuge.textContent = grade;
  gradeBadgeHuge.className = 'grade-badge-huge';

  if (gradeCode === 'MUTU_I') {
    gradeBadgeHuge.classList.add('grade-a');
  } else if (gradeCode === 'MUTU_II') {
    gradeBadgeHuge.classList.add('grade-b');
  } else {
    gradeBadgeHuge.classList.add('grade-reject');
  }

  // Module 1: YOLO 360 Summary
  const sum360 = data.summary_360 || {};
  valTotalAngles.textContent = `${data.total_angles || selectedFiles.length} Sudut`;
  const dominantDefect = sum360.dominant_defect || 'good';
  valYolo360Class.innerHTML = `<span class="pill pill-${dominantDefect}">${dominantDefect.toUpperCase()}</span>`;
  valYolo360Conf.textContent = sum360.confidence ? (sum360.confidence * 100).toFixed(1) + '%' : '-';
  valYolo360Notes.textContent = sum360.notes || '-';

  // Module 2: OpenCV Geometry & Size
  const geo = data.opencv_geometry_inspection || {};
  valDimensions.textContent = `${geo.length_mm ?? 0} x ${geo.width_mm ?? 0} mm`;
  valEsi.textContent = `${geo.esi_percentage ?? 0}% (${geo.shape_status ?? 'Normal'})`;
  valWeight.textContent = `~${geo.estimated_weight_gram ?? 0} gram`;
  valSizeCat.textContent = geo.size_category ?? '-';

  // Module 3: LLM Gemini Interpretation
  lastInspectionData = data;
  const llmBox = document.getElementById('llmNarrativeBox');
  if (llmBox) {
    if (data.llm_interpretation) {
      llmBox.innerHTML = `<div style="white-space: pre-wrap; font-weight: 500;">${data.llm_interpretation}</div>`;
    } else {
      llmBox.innerHTML = `<span style="color: #9ca3af; font-style: italic;">Tidak ada narasi auditor yang dihasilkan.</span>`;
    }
  }
}

// LLM Interactive Chatbot Logic
let lastInspectionData = null;
let chatHistoryList = [];
const btnToggleChat = document.getElementById('btnToggleChat');
const llmChatSection = document.getElementById('llmChatSection');
const chatHistory = document.getElementById('chatHistory');
const chatInput = document.getElementById('chatInput');
const btnSendChat = document.getElementById('btnSendChat');

if (btnToggleChat) {
  btnToggleChat.addEventListener('click', () => {
    if (llmChatSection.style.display === 'none') {
      llmChatSection.style.display = 'block';
      btnToggleChat.textContent = '❌ Tutup Chatbot';
    } else {
      llmChatSection.style.display = 'none';
      btnToggleChat.textContent = '💬 Buka Chatbot Auditor';
    }
  });
}

async function sendChatMessage(query) {
  const msg = (query || (chatInput ? chatInput.value : '')).trim();
  if (!msg) return;

  if (chatInput) chatInput.value = '';

  // Append user bubble
  chatHistory.innerHTML += `
    <div style="background: #059669; padding: 8px 12px; border-radius: 8px; font-size: 12px; color: #fff; align-self: flex-end; max-width: 85%;">
      <strong>Operator:</strong> ${msg}
    </div>
  `;
  chatHistory.scrollTop = chatHistory.scrollHeight;

  chatHistoryList.push({ role: 'user', content: msg });

  // Loading indicator
  const loadingId = 'loading_' + Date.now();
  chatHistory.innerHTML += `
    <div id="${loadingId}" style="background: #1e293b; padding: 8px 12px; border-radius: 8px; font-size: 12px; color: #94a3b8; align-self: flex-start;">
      <em>Auditor Gemini sedang mengetik analisis...</em>
    </div>
  `;
  chatHistory.scrollTop = chatHistory.scrollHeight;

  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  try {
    const res = await fetch(`${baseUrl}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        history: chatHistoryList.slice(-6),
        inspection_context: lastInspectionData
      })
    });

    const replyEl = document.getElementById(loadingId);
    if (replyEl) replyEl.remove();

    if (res.ok) {
      const data = await res.json();
      const reply = data.reply || 'Tidak ada balasan dari model auditor.';
      chatHistoryList.push({ role: 'assistant', content: reply });
      chatHistory.innerHTML += `
        <div style="background: #1e293b; padding: 8px 12px; border-radius: 8px; font-size: 12px; color: #cbd5e1; align-self: flex-start; max-width: 90%;">
          <strong>Auditor Gemini SNI:</strong><br>${reply}
        </div>
      `;
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    const replyEl = document.getElementById(loadingId);
    if (replyEl) replyEl.remove();
    chatHistory.innerHTML += `
      <div style="background: #7f1d1d; padding: 8px 12px; border-radius: 8px; font-size: 12px; color: #fecaca; align-self: flex-start;">
        ⚠️ Gagal menghubungi Auditor: ${err.message}
      </div>
    `;
  }
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

if (btnSendChat) {
  btnSendChat.addEventListener('click', () => sendChatMessage());
}

if (chatInput) {
  chatInput.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') sendChatMessage();
  });
}

window.quickPrompt = function(text) {
  sendChatMessage(text);
};

function resetFiles() {
  selectedFiles = [];
  loadedImageBitmaps = [];
  fileInput.value = '';
  btnPredict.disabled = true;
  badgeFileCount.textContent = '0 Sudut Dipilih';
  clearFilesLink.style.display = 'none';
  lastInspectionData = null;
  chatHistoryList = [];

  const llmBox = document.getElementById('llmNarrativeBox');
  if (llmBox) {
    llmBox.innerHTML = `
      <span style="color: #9ca3af; font-style: italic;">
        Narasi audit mutu resmi SNI 3926:2023 dari Google Gemini akan tampil otomatis setelah analisis telur selesai.
      </span>
    `;
  }

  anglesList.innerHTML = `
    <div style="grid-column: 1 / -1; text-align: center; color: #9ca3af; font-size: 12px; padding: 20px;">
      Belum ada foto telur diunggah.<br>Pilih 1 atau beberapa foto untuk memulai.
    </div>
  `;

  sortingActionBadge.textContent = '🎯 Menunggu Analisis Telur';
  gradeReasonText.textContent = 'Silakan masukkan foto sudut cangkang telur pada panel di samping lalu tekan tombol analisis.';
  gradeStandardRef.textContent = 'Acuan: SNI 3926:2023 (Tabel 1: Kerabang & Bentuk)';
  gradeBadgeHuge.style.display = 'none';

  valTotalAngles.textContent = '-';
  valYolo360Class.textContent = '-';
  valYolo360Conf.textContent = '-';
  valYolo360Notes.textContent = '-';

  valDimensions.textContent = '-';
  valEsi.textContent = '-';
  valWeight.textContent = '-';
  valSizeCat.textContent = '-';

  rawJsonResponse.textContent = '// Respons dari endpoint POST /api/v1/predict akan tampil di sini secara lengkap.';
  showToast('Dashboard telah di-reset.', 'info');
}

btnReset.addEventListener('click', resetFiles);

