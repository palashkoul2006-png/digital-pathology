/**
 * script.js — PathAI Frontend Logic
 * Handles: tab switching, drag-and-drop upload, camera capture,
 *          image preview, API call to /predict, results rendering.
 * Updated to display 7-class ViT skin cancer predictions.
 */

'use strict';

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const tabBtns        = document.querySelectorAll('.tab-btn');
const tabContents    = { upload: id('tab-upload'), camera: id('tab-camera') };
const dropZone       = id('drop-zone');
const fileInput      = id('file-input');
const previewWrap    = id('preview-wrap');
const previewImg     = id('preview-img');
const btnAnalyze     = id('btn-analyze');
const btnText        = btnAnalyze.querySelector('.btn-text');
const btnLoader      = btnAnalyze.querySelector('.btn-loader');
// Patient Tracking
const patientIdInput  = id('patient-id-input');
const btnViewHistory  = id('btn-view-history');
const historyCard     = id('history-card');
const historyTimeline = id('history-timeline');
// Chat UI & Memory
const chatInput = id('chat-input');
const btnSendChat = id('btn-send-chat');
const chatWindow = id('chat-window');
let currentDiagnosisContext = "";
let chatHistoryString = "";

// Camera
const btnStartCamera = id('btn-start-camera');
const btnCapture     = id('btn-capture');
const cameraFeed     = id('camera-feed');
const captureCanvas  = id('capture-canvas');

// Results
const idleState      = id('idle-state');
const loadingState   = id('loading-state');
const resultsContent = id('results-content');

// Generative AI
const btnSimulate         = id('btn-simulate');
const btnSimText          = id('btn-sim-text');
const btnSimLoader        = id('btn-sim-loader');
const simulatedResultWrap = id('simulated-result-wrap');
const simulatedImg        = id('simulated-img');

function id(s) { return document.getElementById(s); }

/* ── State ─────────────────────────────────────────────────────────────────── */
let currentFile   = null;   // File object for upload
let capturedB64   = null;   // Base64 string for camera capture
let cameraStream  = null;

/* ════════════════════════════ TAB SWITCHING ══════════════════════════════════ */
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    tabBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    Object.entries(tabContents).forEach(([key, el]) => {
      el.classList.toggle('hidden', key !== target);
    });
    // Stop camera when switching away
    if (target !== 'camera') stopCamera();
  });
});

/* ════════════════════════════ DRAG & DROP ════════════════════════════════════ */
//dropZone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('dragover'); })
);

dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFileSelected(fileInput.files[0]);
});

function handleFileSelected(file) {
  const allowed = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/webp'];
  if (!allowed.includes(file.type)) {
    alert('Unsupported file type. Please upload PNG, JPG, BMP, TIFF, or WEBP.');
    return;
  }
  currentFile  = file;
  capturedB64  = null;
  const url = URL.createObjectURL(file);
  showPreview(url);
}

/* ════════════════════════════ CAMERA ════════════════════════════════════════ */
btnStartCamera.addEventListener('click', async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    cameraFeed.srcObject = cameraStream;
    btnCapture.disabled = false;
    btnStartCamera.textContent = 'Camera On ✓';
    btnStartCamera.disabled = true;
  } catch (err) {
    alert('Camera access denied or unavailable: ' + err.message);
  }
});

btnCapture.addEventListener('click', () => {
  captureCanvas.width  = cameraFeed.videoWidth  || 640;
  captureCanvas.height = cameraFeed.videoHeight || 480;
  captureCanvas.getContext('2d').drawImage(cameraFeed, 0, 0);
  capturedB64 = captureCanvas.toDataURL('image/jpeg', 0.92);
  currentFile = null;
  showPreview(capturedB64);
  stopCamera();
});

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream  = null;
    cameraFeed.srcObject = null;
  }
  btnCapture.disabled      = true;
  btnStartCamera.disabled  = false;
  btnStartCamera.textContent = 'Start Camera';
}

/* ════════════════════════════ PREVIEW ═══════════════════════════════════════ */
function showPreview(src) {
  previewImg.src = src;
  previewWrap.classList.remove('hidden');
  resetResults();
  // Smoothly scroll to show the button
  previewWrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ════════════════════════════ ANALYSE ═══════════════════════════════════════ */
btnAnalyze.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!currentFile && !capturedB64) {
    alert('No image selected.');
    return;
  }
  setLoading(true);
  showLoadingResults();

  try {
    let response;
    // Grab the ID, default to 'Anonymous' if left blank
    const pid = patientIdInput.value.trim() || 'Anonymous';

    if (currentFile) {
      const formData = new FormData();
      formData.append('file', currentFile);
      formData.append('patient_id', pid); // <--- ADD THIS LINE
      response = await fetch('/predict', { method: 'POST', body: formData });
    } else {
      response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_b64: capturedB64, patient_id: pid }), // <--- UPDATE THIS LINE
      });
    }

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || `Server error ${response.status}`);
    }

    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

/* ════════════════════════════ RESULTS RENDERING ════════════════════════════ */

// Category → colour mapping
const CATEGORY_COLORS = {
  'Malignant':     '#ff4f6a',
  'Pre-Malignant': '#ffb347',
  'Benign':        '#34d97b',
};

// Display name formatter
function formatClassName(raw) {
  return raw.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function renderResults(data) {
  // Store the diagnosis context for the chatbot
  currentDiagnosisContext = `Prediction: ${data.display_name}, Confidence: ${data.confidence}%, Risk: ${data.risk_level}.`;
  chatHistoryString = ""; // Reset memory for a new patient
  chatWindow.innerHTML = `<div style="background: rgba(45,125,210,0.2); padding: 8px 12px; border-radius: 8px; align-self: flex-start; max-width: 85%; color: #c5e8fb;">Hello! Do you have any questions about this ${data.display_name} diagnosis?</div>`;
  // Switch to results view
  idleState.classList.add('hidden');
  loadingState.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  const categoryColor = CATEGORY_COLORS[data.category] || '#a0aec0';

  // ── Prediction ─────────────────────────────────────────────────────
  const predEl = id('pred-text');
  // Show human-readable display name
  predEl.textContent = data.display_name || formatClassName(data.prediction);
  predEl.className   = 'result-value pred-value';
  predEl.style.color = categoryColor;

  // Category tag under prediction (Malignant / Pre-Malignant / Benign)
  const catTag = id('category-tag');
  if (catTag) {
    catTag.textContent   = data.category || '';
    catTag.style.background = categoryColor + '22';
    catTag.style.color      = categoryColor;
    catTag.style.border     = `1px solid ${categoryColor}44`;
  }

  // ── Confidence bar ──────────────────────────────────────────────────
  const bar  = id('confidence-bar');
  const conf = data.confidence;                       // 0–100
  bar.style.background = categoryColor;
  requestAnimationFrame(() => { bar.style.width = conf + '%'; });
  id('conf-text').textContent = conf.toFixed(1) + '%';

  // ── Low-confidence warning banner ───────────────────────────────────
  let existingBanner = id('low-conf-banner');
  if (existingBanner) existingBanner.remove();
  if (conf < 60) {
    const banner = document.createElement('div');
    banner.id = 'low-conf-banner';
    banner.innerHTML = `
      <span style="font-size:1.1rem;">⚠️</span>
      <span>
        <strong>Low Confidence (${conf.toFixed(1)}%)</strong> — The model is uncertain.
        Results should not be relied upon without professional evaluation.
      </span>
    `;
    banner.style.cssText = [
      'display:flex', 'align-items:flex-start', 'gap:0.6rem',
      'background:rgba(255,179,71,0.12)', 'border:1px solid rgba(255,179,71,0.45)',
      'border-left:3px solid #ffb347', 'border-radius:10px',
      'padding:0.75rem 1rem', 'font-size:0.82rem', 'color:#ffb347',
      'line-height:1.5', 'animation:fadeInUp 0.3s ease both',
    ].join(';');
    // Insert after the prediction card (first child of results)
    resultsContent.insertBefore(banner, resultsContent.children[1]);
  }

  // ── Risk badge ──────────────────────────────────────────────────────
  const riskEl = id('risk-badge');
  riskEl.textContent = data.risk_level;
  riskEl.className   = 'risk-badge ' + data.risk_level.toLowerCase();

  // ── Model tag ───────────────────────────────────────────────────────
  id('model-tag').textContent = data.model_used || 'ViT (fine-tuned)';

  // ── Class probability bars ──────────────────────────────────────────
  const probContainer = id('prob-bars');
  if (probContainer && data.all_probs) {
    probContainer.innerHTML = '';
    const sorted = Object.entries(data.all_probs).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([cls, pct]) => {
      const isTop = cls === data.prediction;
      const row = document.createElement('div');
      row.className = 'prob-row' + (isTop ? ' prob-top' : '');
      row.innerHTML = `
        <span class="prob-label">${formatClassName(cls)}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width:0%;background:${isTop ? categoryColor : '#4a5568'}"></div>
        </div>
        <span class="prob-pct">${pct.toFixed(1)}%</span>
      `;
      probContainer.appendChild(row);
      // Animate after paint
      requestAnimationFrame(() => {
        row.querySelector('.prob-fill').style.width = pct + '%';
      });
    });
  }

  // ── Images ──────────────────────────────────────────────────────────
  id('original-img').src = data.original_image;
  id('heatmap-img').src  = data.heatmap;

  // ── Recommendation ──────────────────────────────────────────────────
  const rec = data.recommendation;
  id('rec-summary').textContent = rec.summary;

  const stepsList = id('rec-steps');
  stepsList.innerHTML = '';
  (rec.steps || []).forEach(step => {
    const li = document.createElement('li');
    li.textContent = step;
    stepsList.appendChild(li);
  });

  const mapsLink = id('maps-link');
  mapsLink.href = rec.maps_link || '#';

  id('rec-disclaimer').textContent = rec.disclaimer;

  // ── New Analysis button ──────────────────────────────────────────────
  let existingReset = id('reset-btn-results');
  if (existingReset) existingReset.remove();
  const resetBtn = document.createElement('button');
  resetBtn.id = 'reset-btn-results';
  resetBtn.className = 'btn btn-ghost btn-full';
  resetBtn.textContent = '🔄 New Analysis';
  resetBtn.style.marginTop = '0.5rem';
  resetBtn.addEventListener('click', () => {
    currentFile  = null;
    capturedB64  = null;
    previewImg.src = '';
    previewWrap.classList.add('hidden');
    fileInput.value = '';
    resetResults();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  resultsContent.appendChild(resetBtn);

  // Scroll results into view
  resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showLoadingResults() {
  idleState.classList.add('hidden');
  loadingState.classList.remove('hidden');
  resultsContent.classList.add('hidden');
}

function showError(msg) {
  idleState.classList.add('hidden');
  loadingState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  idleState.classList.remove('hidden');
  idleState.innerHTML = `
    <div class="idle-icon">⚠️</div>
    <p style="color:#ff4f6a;font-weight:600;">Analysis Error</p>
    <p style="font-size:0.82rem;">${escapeHtml(msg)}</p>
    <button class="btn btn-ghost" onclick="resetResults()" style="margin-top:0.5rem;">Try Again</button>
  `;
}

function resetResults() {
  idleState.classList.remove('hidden');
  loadingState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  idleState.innerHTML = `
    <div class="idle-icon">🧬</div>
    <p>Upload or capture an image to begin analysis</p>
  `;
  // Add these two lines to hide the generative AI card on reset
  if(simulatedResultWrap) simulatedResultWrap.classList.add('hidden');
  if(simulatedImg) simulatedImg.src = '';
}

/* ════════════════════════════ HELPERS ═══════════════════════════════════════ */
function setLoading(active) {
  btnAnalyze.disabled = active;
  btnText.classList.toggle('hidden', active);
  btnLoader.classList.toggle('hidden', !active);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ════════════════════════════ DISCLAIMER MODAL ══════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  const modal = id('disclaimer-modal');
  const acceptBtn = id('btn-accept-disclaimer');

  // We use localStorage so it doesn't annoy you every time you refresh during dev
  if (!localStorage.getItem('pathai_disclaimer_accepted')) {
    modal.classList.remove('hidden');
  } else {
    modal.classList.add('hidden');
  }

  acceptBtn.addEventListener('click', () => {
    localStorage.setItem('pathai_disclaimer_accepted', 'true');
    modal.style.opacity = '0';
    setTimeout(() => modal.classList.add('hidden'), 400);
  });
});

/* ════════════════════════════ GENERATIVE AI SIMULATION ═════════════════════ */
btnSimulate.addEventListener('click', async () => {
  if (!currentFile && !capturedB64) {
    alert('No image available to simulate.');
    return;
  }

  // Set loading state (Generative AI takes 10-20 seconds on 6GB VRAM)
  btnSimulate.disabled = true;
  btnSimText.classList.add('hidden');
  btnSimLoader.classList.remove('hidden');

  try {
    const formData = new FormData();
    
    // If uploaded via file
    if (currentFile) {
      formData.append('file', currentFile);
    } 
    // If captured via camera, convert base64 to a Blob for the backend
    else if (capturedB64) {
      const response = await fetch(capturedB64);
      const blob = await response.blob();
      formData.append('file', blob, 'camera_capture.jpg');
    }

    // Hit the Stable Diffusion backend
    const response = await fetch('/api/simulate_progression', { 
      method: 'POST', 
      body: formData 
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || `Server error ${response.status}`);
    }

    // Display the synthetic image!
    simulatedImg.src = data.synthetic_image;
    simulatedResultWrap.classList.remove('hidden');

  } catch (err) {
    alert('Simulation failed: ' + err.message);
  } finally {
    // Reset button state
    btnSimulate.disabled = false;
    btnSimText.classList.remove('hidden');
    btnSimLoader.classList.add('hidden');
  }
}); // <--- ADD THESE CHARACTERS HERE TO CLOSE THE SIMULATE BUTTON

/* ════════════════════════════ AGENTIC CHAT LOGIC ═══════════════════════════ */
btnSendChat.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendChatMessage(); });

async function sendChatMessage() {
  const message = chatInput.value.trim();
  if (!message) return;

  appendMessage('user', message);
  chatInput.value = '';
  const loadingId = 'loading-' + Date.now();
  appendMessage('ai', 'Thinking...', loadingId);

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, context: currentDiagnosisContext, history: chatHistoryString })
    });
    const data = await response.json();
    document.getElementById(loadingId).remove();

    if (data.status === 'success') {
      appendMessage('ai', data.reply);
      chatHistoryString += `\nPatient: ${message}\nAI: ${data.reply}\n`;
    } else {
      appendMessage('ai', 'Error: ' + data.error);
    }
  } catch (err) {
    document.getElementById(loadingId).remove();
    appendMessage('ai', 'Connection failed.');
  }
}

function appendMessage(sender, text, id = null) {
  const msgDiv = document.createElement('div');
  if (id) msgDiv.id = id;
  msgDiv.style.padding = '8px 12px';
  msgDiv.style.borderRadius = '8px';
  msgDiv.style.maxWidth = '85%';
  msgDiv.style.lineHeight = '1.4';
  msgDiv.textContent = text;

  if (sender === 'user') {
    msgDiv.style.background = 'rgba(0,212,170,0.2)';
    msgDiv.style.color = '#00d4aa';
    msgDiv.style.alignSelf = 'flex-end';
    msgDiv.style.border = '1px solid rgba(0,212,170,0.3)';
  } else {
    msgDiv.style.background = 'rgba(45,125,210,0.2)';
    msgDiv.style.color = '#c5e8fb';
    msgDiv.style.alignSelf = 'flex-start';
  }
  chatWindow.appendChild(msgDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

/* ════════════════════════════ PATIENT TIMELINE LOGIC ═══════════════════════ */
btnViewHistory.addEventListener('click', async () => {
  const pid = patientIdInput.value.trim();
  if (!pid) return;

  btnViewHistory.textContent = 'Loading records...';
  
  try {
    const res = await fetch(`/api/history/${pid}`);
    const data = await res.json();
    
    if (data.status === 'success') {
      historyTimeline.innerHTML = '';
      
      if (data.history.length === 0) {
        historyTimeline.innerHTML = '<p class="panel-sub">No previous records found for this ID.</p>';
      } else {
        // Build the visual timeline blocks
        data.history.forEach(item => {
          let riskColor = '#34d97b'; // Low (Green)
          if (item.risk_level === 'Medium') riskColor = '#ffb347'; // Warning (Orange)
          if (item.risk_level === 'High') riskColor = '#ff4f6a'; // Danger (Red)

          const entry = document.createElement('div');
          entry.style.cssText = `padding: 12px; border-left: 3px solid ${riskColor}; background: rgba(255,255,255,0.03); border-radius: 0 8px 8px 0;`;
          entry.innerHTML = `
            <div style="font-size: 0.75rem; color: rgba(196,224,255,0.5); margin-bottom: 4px;">${item.date}</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #fff; margin-bottom: 4px;">${item.prediction}</div>
            <div style="font-size: 0.8rem; color: rgba(196,224,255,0.7);">
              Confidence: ${item.confidence}% | Risk: <span style="color: ${riskColor}; font-weight: bold;">${item.risk_level}</span>
            </div>
          `;
          historyTimeline.appendChild(entry);
        });
      }
      
      historyCard.classList.remove('hidden');
      btnViewHistory.classList.add('hidden'); // Hide button once loaded
    }
  } catch (err) {
    alert('Failed to load patient history.');
  } finally {
    btnViewHistory.textContent = '🕒 Load Patient History';
  }
});

// Show the timeline button inside renderResults() IF an ID was provided
const originalRenderResults = renderResults;
renderResults = function(data) {
  originalRenderResults(data);
  if (patientIdInput.value.trim() && patientIdInput.value.trim() !== 'Anonymous') {
    btnViewHistory.classList.remove('hidden');
  }
};

// Hide the timeline elements when resetResults() is called
const originalResetResults = resetResults;
resetResults = function() {
  originalResetResults();
  if(historyCard) historyCard.classList.add('hidden');
  if(btnViewHistory) btnViewHistory.classList.add('hidden');
};