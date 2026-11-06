// ==============================================================================
// CLIENT LOGIC SCRIPT - EGGSORT AI DASHBOARD
// Multi-Angle YOLOv8 + OpenCV Geometry + Google Gemini (gemini-3.8-flash) Popup Chatbot
// SNI 3926:2023 Compliant with Strict Manual Prompt Boundary Warning
// ==============================================================================

// DOM Elements: General
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

// DOM Elements: Hero Card
const sortingActionBadge = document.getElementById('sortingActionBadge');
const gradeReasonText = document.getElementById('gradeReasonText');
const gradeStandardRef = document.getElementById('gradeStandardRef');
const gradeBadgeHuge = document.getElementById('gradeBadgeHuge');

// DOM Elements: Module 1 (YOLO 360)
const valTotalAngles = document.getElementById('valTotalAngles');
const valYolo360Class = document.getElementById('valYolo360Class');
const valYolo360Conf = document.getElementById('valYolo360Conf');
const valYolo360Notes = document.getElementById('valYolo360Notes');

// DOM Elements: Module 2 (OpenCV Geometry)
const valDimensions = document.getElementById('valDimensions');
const valEsi = document.getElementById('valEsi');
const valWeight = document.getElementById('valWeight');
const valSizeCat = document.getElementById('valSizeCat');

// DOM Elements: Floating Popup Chatbot Widget
const chatFloatingBtn = document.getElementById('chatFloatingBtn');
const chatFloatingIcon = document.getElementById('chatFloatingIcon');
const chatBadgeDot = document.getElementById('chatBadgeDot');
const chatPopupWidget = document.getElementById('chatPopupWidget');
const btnCloseChat = document.getElementById('btnCloseChat');
const btnClearChat = document.getElementById('btnClearChat');
const btnSetApiKey = document.getElementById('btnSetApiKey');
const chatMessagesContainer = document.getElementById('chatMessagesContainer');
const quickChipsContainer = document.getElementById('quickChipsContainer');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const btnSendChat = document.getElementById('btnSendChat');
const chatSendText = document.getElementById('chatSendText');
const chatSendSpinner = document.getElementById('chatSendSpinner');
const llmBadge = document.getElementById('llmBadge');

// Raw JSON & Toast
const rawJsonResponse = document.getElementById('rawJsonResponse');
const toast = document.getElementById('toast');

// Client State
let selectedFiles = [];
let loadedImageBitmaps = [];
let currentInspectionData = null;
let chatHistory = [];

// Toast Helper
function showToast(message, type = 'info') {
  toast.textContent = message;
  toast.className = `toast ${type} show`;
  setTimeout(() => {
    toast.classList.remove('show');
  }, 4000);
}

// Markdown Formatter Helper
function formatMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .split(/\n\s*\n/)
    .map(p => {
      const lines = p.split('\n');
      const isList = lines.every(l => l.trim().startsWith('- ') || l.trim().startsWith('* ') || l.trim().startsWith('• '));
      if (isList) {
        return '<ul>' + lines.map(l => `<li>${l.replace(/^[-*•]\s*/, '')}</li>`).join('') + '</ul>';
      }
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    })
    .join('');
}

// ==============================================================================
// POPUP CHATBOT OPEN / CLOSE CONTROLLER
// ==============================================================================
function openChatPopup() {
  chatPopupWidget.classList.add('open');
  chatFloatingIcon.textContent = '✕';
  chatBadgeDot.classList.remove('active');
  chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  setTimeout(() => {
    chatInput.focus();
  }, 150);
}

function closeChatPopup() {
  chatPopupWidget.classList.remove('open');
  chatFloatingIcon.textContent = '💬';
}

function toggleChatPopup() {
  if (chatPopupWidget.classList.contains('open')) {
    closeChatPopup();
  } else {
    openChatPopup();
  }
}

chatFloatingBtn.addEventListener('click', toggleChatPopup);
btnCloseChat.addEventListener('click', closeChatPopup);

// Tutup popup dengan tombol Escape pada keyboard
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && chatPopupWidget.classList.contains('open')) {
    closeChatPopup();
  }
});

// API Key Storage Helper
function getStoredApiKey() {
  return localStorage.getItem('gemini_api_key') || '';
}

// ==============================================================================
// API HEALTH CHECK
// ==============================================================================
if (window.location.protocol.startsWith('http')) {
  apiUrlInput.value = window.location.origin;
}

async function checkApiHealth() {
  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  const storedKey = getStoredApiKey();
  const queryParam = storedKey ? `?api_key=${encodeURIComponent(storedKey)}` : '';
  try {
    const res = await fetch(`${baseUrl}/api/v1/status${queryParam}`, {
      method: 'GET',
      headers: storedKey ? { 'X-Gemini-Key': storedKey } : {}
    });
    if (res.ok) {
      const data = await res.json();
      statusDot.className = "status-dot online";
      const llmInfo = data.llm_active
        ? ` • Gemini: Aktif (${data.llm_model || 'gemini-3.8-flash'})`
        : ' • Gemini: Offline Fallback';
      statusText.textContent = `API Terhubung (${data.device || 'Flask'}${llmInfo})`;

      if (llmBadge) {
        if (data.llm_active) {
          llmBadge.textContent = data.llm_model || 'Gemini 3.8 Flash';
          llmBadge.className = 'llm-badge gemini';
        } else {
          llmBadge.textContent = 'Offline Fallback';
          llmBadge.className = 'llm-badge';
        }
      }
    } else {
      throw new Error("HTTP Status " + res.status);
    }
  } catch (err) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "API Tidak Aktif (Jalankan app_flask.py)";
  }
}

checkApiHealth();
setInterval(checkApiHealth, 5000);
apiUrlInput.addEventListener('change', checkApiHealth);

// ==============================================================================
// DRAG & DROP FILE UPLOAD HANDLERS
// ==============================================================================
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

function handleFilesSelected(files) {
  const validFiles = files.filter(f => f.type.startsWith('image/'));
  if (validFiles.length === 0) {
    showToast('Harap pilih file citra yang valid (JPG/PNG)!', 'error');
    return;
  }

  selectedFiles = validFiles;
  badgeFileCount.textContent = `${selectedFiles.length} Sudut Dipilih`;
  btnPredict.disabled = false;
  clearFilesLink.style.display = 'inline';

  renderAnglePreviews();
}

function renderAnglePreviews() {
  anglesList.innerHTML = '';
  loadedImageBitmaps = new Array(selectedFiles.length);

  selectedFiles.forEach((file, idx) => {
    const card = document.createElement('div');
    card.className = 'angle-card';
    card.id = `angleCard_${idx}`;

    const angleNumber = idx + 1;
    card.innerHTML = `
      <div class="angle-header">
        <span class="angle-title">Sudut ${angleNumber}: ${file.name}</span>
        <span class="angle-status" id="statusBadge_${idx}">Menunggu...</span>
      </div>
      <div class="angle-canvas-wrapper">
        <canvas id="canvas_${idx}"></canvas>
      </div>
    `;
    anglesList.appendChild(card);

    const canvas = card.querySelector(`#canvas_${idx}`);
    const reader = new FileReader();

    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        loadedImageBitmaps[idx] = img;
        drawInitialCanvas(canvas, img);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

function drawInitialCanvas(canvas, img) {
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
}

function drawDetectionsOnCanvas(canvas, img, detections) {
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  if (!detections || detections.length === 0) return;

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox_xyxy;
    const w = x2 - x1;
    const h = y2 - y1;
    const name = det.class_name;
    const conf = (det.confidence * 100).toFixed(0);

    let boxColor = '#10b981';
    if (name === 'cracked') boxColor = '#ef4444';
    else if (name === 'dirt') boxColor = '#f59e0b';

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

// ==============================================================================
// PREDICT HANDLER (POST /api/v1/predict)
// ==============================================================================
btnPredict.addEventListener('click', async () => {
  if (selectedFiles.length === 0) {
    showToast('Pilih minimal 1 citra sudut telur!', 'error');
    return;
  }

  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  btnPredict.disabled = true;
  btnSpinner.style.display = 'inline-block';
  btnText.textContent = 'Menganalisis 360° & Audit Gemini...';

  showChatTyping("Sedang mengaudit citra telur dengan YOLOv8, OpenCV, dan Google Gemini...");

  const formData = new FormData();
  selectedFiles.forEach(file => {
    formData.append('files', file);
  });

  const storedKey = getStoredApiKey();
  if (storedKey) {
    formData.append('api_key', storedKey);
  }

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
    hideChatTyping();
    displayResults(data);
    showToast('Inspeksi 360° Selesai! Klik icon 💬 di pojok kanan bawah untuk laporan & chat.', 'success');

  } catch (error) {
    console.error(error);
    hideChatTyping();
    showToast('Error: ' + error.message, 'error');
  } finally {
    btnPredict.disabled = false;
    btnSpinner.style.display = 'none';
    btnText.textContent = '⚡ Analisis 360° & Tentukan Grade';
  }
});

// ==============================================================================
// DISPLAY PREDICTION RESULTS
// ==============================================================================
function displayResults(data) {
  currentInspectionData = data;
  rawJsonResponse.textContent = JSON.stringify(data, null, 2);

  // 1. Update each angle card
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

  // 2. Final Grade Hero Card
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

  // 3. Module 1: YOLO 360 Summary
  const sum360 = data.summary_360 || {};
  valTotalAngles.textContent = `${data.total_angles || selectedFiles.length} Sudut`;
  const dominantDefect = sum360.dominant_defect || 'good';
  valYolo360Class.innerHTML = `<span class="pill pill-${dominantDefect}">${dominantDefect.toUpperCase()}</span>`;
  valYolo360Conf.textContent = sum360.confidence ? (sum360.confidence * 100).toFixed(1) + '%' : '-';
  valYolo360Notes.textContent = sum360.notes || '-';

  // 4. Module 2: OpenCV Geometry & Size
  const geo = data.opencv_geometry_inspection || {};
  valDimensions.textContent = `${geo.length_mm ?? 0} x ${geo.width_mm ?? 0} mm`;
  valEsi.textContent = `${geo.esi_percentage ?? 0}% (${geo.shape_status ?? 'Normal'})`;
  valWeight.textContent = `~${geo.estimated_weight_gram ?? 0} gram`;
  valSizeCat.textContent = geo.size_category ?? '-';

  // 5. Masukkan Laporan Audit Naratif Langsung ke Chatbot Popup
  const rawLlmText = data.llm_interpretation || '';
  if (rawLlmText.trim()) {
    const reportHtml = `
      <div style="margin-bottom: 8px; font-weight: 700; color: #1e293b; border-bottom: 1px dashed #cbd5e1; padding-bottom: 4px;">
        📋 Laporan Hasil Audit Mutu Telur:
      </div>
      ${formatMarkdown(rawLlmText)}
    `;
    appendChatMessage('assistant', reportHtml, true, 'Auditor Mutu Telur (SNI 3926:2023)');

    chatHistory.push({
      role: 'assistant',
      content: rawLlmText
    });

    // Nyalakan indikator titik badge pada floating button jika popup sedang tertutup
    if (!chatPopupWidget.classList.contains('open')) {
      chatBadgeDot.classList.add('active');
    }
  }

  // Update badge LLM di popup header
  if (llmBadge && data.llm_metadata) {
    const modelName = data.llm_metadata.model || 'Gemini 3.8 Flash';
    llmBadge.textContent = modelName.includes('gemini-3.8-flash') ? 'Gemini 3.8 Flash' : modelName;
    llmBadge.className = 'llm-badge gemini';
  }
}

// ==============================================================================
// CHATBOT INTERACTIVE SYSTEM (GEMINI 3.8 FLASH POPUP)
// ==============================================================================
function appendChatMessage(role, content, isHtml = false, customSender = null, isWarning = false) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-msg msg-${role}${isWarning ? ' msg-warning' : ''}`;

  let avatar = '🤖';
  if (role === 'user') avatar = '👤';
  else if (isWarning) avatar = '⚠️';

  const defaultSender = role === 'user' ? 'Anda (Operator)' : 'Auditor Mutu Telur (SNI 3926:2023)';
  const senderName = customSender || defaultSender;

  const formattedContent = isHtml ? content : formatMarkdown(content);

  msgDiv.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-sender">${senderName}</div>
      <div class="msg-text">${formattedContent}</div>
    </div>
  `;

  chatMessagesContainer.appendChild(msgDiv);
  chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  return msgDiv;
}

let activeTypingElement = null;

function showChatTyping(customText = "Auditor sedang menganalisis...") {
  hideChatTyping();
  const typingDiv = document.createElement('div');
  typingDiv.className = 'chat-msg msg-assistant';
  typingDiv.id = 'chatTypingIndicator';

  typingDiv.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body">
      <div class="msg-sender">Auditor Mutu Telur (SNI 3926:2023)</div>
      <div class="msg-text">
        <div class="chat-typing-indicator">
          <span style="font-size: 12px; color: #64748b; margin-right: 6px;">${customText}</span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    </div>
  `;

  chatMessagesContainer.appendChild(typingDiv);
  chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  activeTypingElement = typingDiv;
}

function hideChatTyping() {
  if (activeTypingElement && activeTypingElement.parentNode) {
    activeTypingElement.parentNode.removeChild(activeTypingElement);
  }
  activeTypingElement = null;
}

async function handleChatSubmit(userQuery) {
  const query = (userQuery || '').trim();
  if (!query) return;

  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");

  // Render User Message
  appendChatMessage('user', query);
  chatInput.value = '';

  // Update State & UI
  chatHistory.push({ role: 'user', content: query });
  btnSendChat.disabled = true;
  chatInput.disabled = true;
  chatSendText.style.display = 'none';
  chatSendSpinner.style.display = 'inline-block';

  showChatTyping("Auditor sedang memvalidasi prompt...");

  const storedKey = getStoredApiKey();

  try {
    const response = await fetch(`${baseUrl}/api/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(storedKey ? { 'X-Gemini-Key': storedKey } : {})
      },
      body: JSON.stringify({
        message: query,
        history: chatHistory.slice(-6), // Kirim 6 giliran terakhir untuk efisiensi token
        inspection_context: currentInspectionData,
        api_key: storedKey || undefined
      })
    });

    hideChatTyping();

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || 'Gagal menghubungi layanan chat Gemini.');
    }

    const data = await response.json();
    const replyText = data.reply || 'Tidak ada balasan dari Auditor.';
    const isWarning = data.is_warning || data.status === 'warning' || replyText.includes('PERINGATAN: TOPIK DI LUAR KONTEKS') || replyText.startsWith('⚠️');

    if (isWarning) {
      appendChatMessage('assistant', replyText, false, '⚠️ Auditor Mutu (Peringatan Batasan)', true);
      showToast('⚠️ Peringatan: Pertanyaan di luar batasan sistem sortir telur!', 'error');
    } else {
      appendChatMessage('assistant', replyText, false);
    }

    chatHistory.push({ role: 'assistant', content: replyText });

  } catch (err) {
    hideChatTyping();
    appendChatMessage(
      'assistant',
      `⚠️ *Terjadi kesalahan saat memproses respon*: ${err.message}. Pastikan server Flask aktif.`,
      false,
      'System Error',
      true
    );
  } finally {
    btnSendChat.disabled = false;
    chatInput.disabled = false;
    chatSendText.style.display = 'inline';
    chatSendSpinner.style.display = 'none';
    chatInput.focus();
  }
}

// Chat Form Listener
chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  handleChatSubmit(chatInput.value);
});

// Quick Chips Click Listeners
if (quickChipsContainer) {
  quickChipsContainer.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip-btn');
    if (chip) {
      const query = chip.getAttribute('data-query');
      if (query) {
        handleChatSubmit(query);
      }
    }
  });
}

// Clear Chat Function
function resetChat() {
  chatHistory = [];
  chatMessagesContainer.innerHTML = `
    <div class="chat-msg msg-assistant">
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-sender">Auditor Mutu Telur (SNI 3926:2023)</div>
        <div class="msg-text">
          Riwayat percakapan telah dibersihkan. Silakan ajukan prompt manual atau pertanyaan seputar inspeksi telur!
        </div>
      </div>
    </div>
  `;
  showToast('Percakapan telah dibersihkan.', 'info');
}

btnClearChat.addEventListener('click', resetChat);

// Set Gemini API Key directly from UI
if (btnSetApiKey) {
  btnSetApiKey.addEventListener('click', () => {
    const curKey = getStoredApiKey();
    const newKey = prompt(
      '🔑 SETEL GOOGLE GEMINI API KEY\n\n' +
      'Masukkan API Key Google Gemini Anda (cth: AIzaSy...):\n' +
      'Kunci ini disimpan di browser Anda untuk mengaktifkan Gemini 3.8 Flash secara live.\n\n' +
      'Kosongkan dan klik OK jika ingin kembali ke mode offline:',
      curKey
    );
    if (newKey !== null) {
      const cleanKey = newKey.trim();
      if (cleanKey) {
        localStorage.setItem('gemini_api_key', cleanKey);
        showToast('✅ Gemini API Key tersimpan! Mengaktifkan Gemini 3.8 Flash...', 'success');
      } else {
        localStorage.removeItem('gemini_api_key');
        showToast('Gemini API Key dihapus (Mode Offline Fallback).', 'info');
      }
      checkApiHealth();
    }
  });
}

// ==============================================================================
// RESET ALL FILES & RESULTS
// ==============================================================================
function resetFiles() {
  selectedFiles = [];
  loadedImageBitmaps = [];
  currentInspectionData = null;
  fileInput.value = '';
  btnPredict.disabled = true;
  badgeFileCount.textContent = '0 Sudut Dipilih';
  clearFilesLink.style.display = 'none';

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
