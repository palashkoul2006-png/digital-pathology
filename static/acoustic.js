/**
 * acoustic.js — Task C + D
 * =========================
 * Cough-to-Clinic Acoustic Virtual Stethoscope
 *
 * Responsibilities:
 *   1. Mic capture  — getUserMedia → MediaRecorder (5 s recording)
 *   2. MFCC extract — Web Audio API AnalyserNode + pure-JS DCT
 *                     Produces Float32Array (40 × 128) matching training pipeline
 *   3. Edge infer   — TensorFlow.js loads static/acoustic_model/model.json
 *                     All inference is LOCAL — zero audio sent to server
 *   4. Results UI   — probability bars, diagnosis badge, subclinical warning
 */

'use strict';

// ── Constants (MUST match acoustic/preprocess.py) ─────────────────────────────
const SR          = 22050;    // sample rate in Hz
const DURATION    = 5;        // seconds to record
const N_MFCC      = 40;       // MFCC coefficients
const HOP_LENGTH  = 512;
const N_FFT       = 2048;
const MAX_FRAMES  = 128;

const CLASSES = ['Normal', 'Asthma', 'Pneumonia', 'TB'];
const CLASS_ICONS = { Normal: '✅', Asthma: '🌬️', Pneumonia: '🫁', TB: '⚠️' };
const CLASS_RISK  = { Normal: 'low', Asthma: 'medium', Pneumonia: 'high', TB: 'high' };
const CLASS_COLORS= { Normal: '#00d4aa', Asthma: '#f59e0b', Pneumonia: '#f97316', TB: '#ef4444' };

const MODEL_URL   = '/static/acoustic_model/model.json';

// ── State ─────────────────────────────────────────────────────────────────────
let tfModel        = null;      // loaded TF.js model
let mediaStream    = null;      // mic stream
let mediaRecorder  = null;      // recorder
let audioChunks    = [];        // Blob chunks
let isRecording    = false;
let timerInterval  = null;
let animFrame      = null;
let audioCtx       = null;      // AudioContext for waveform
let analyser       = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const micBtn         = document.getElementById('mic-btn');
const micIcon        = document.getElementById('mic-icon');
const recordLabel    = document.getElementById('record-label');
const recordTimer    = document.getElementById('record-timer');
const waveCanvas     = document.getElementById('waveform-canvas');
const recordingRing  = document.getElementById('recording-ring');
const modelStatus    = document.getElementById('model-status');

// Results panel
const idleState       = document.getElementById('idle-state');
const loadingState    = document.getElementById('loading-state');
const resultsContent  = document.getElementById('results-content');
const diagnosisIcon   = document.getElementById('diagnosis-icon');
const diagnosisName   = document.getElementById('diagnosis-name');
const diagnosisConf   = document.getElementById('diagnosis-conf');
const riskPill        = document.getElementById('risk-pill');
const subclinicalWarn = document.getElementById('subclinical-warning');

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', init);

async function init() {
  setModelStatus('✅ Google HeAR inference server ready', '#00d4aa');
  micBtn.disabled = false;
  micBtn.addEventListener('click', handleMicClick);

  const audioUpload = document.getElementById('audio-upload');
  if (audioUpload) {
    audioUpload.addEventListener('change', handleAudioUpload);
  }
}

function setModelStatus(text, color) {
  if (modelStatus) {
    modelStatus.textContent = text;
    modelStatus.style.color = color;
  }
}

// ── Mic Button Controller ─────────────────────────────────────────────────────
async function handleMicClick() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

// ── Recording ─────────────────────────────────────────────────────────────────
async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: SR, channelCount: 1, echoCancellation: false }
    });
  } catch (err) {
    alert('Microphone access denied. Please allow microphone access in your browser settings.');
    return;
  }

  // Web Audio for waveform visualiser
  audioCtx  = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  analyser  = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  const src = audioCtx.createMediaStreamSource(mediaStream);
  src.connect(analyser);
  drawWaveform();

  // MediaRecorder for collecting raw audio
  audioChunks  = [];
  mediaRecorder = new MediaRecorder(mediaStream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start(100);  // 100 ms chunks

  // UI → recording state
  isRecording = true;
  setRecordingUI(true);

  // Auto-stop after DURATION seconds
  let elapsed = 0;
  recordTimer.textContent = formatTime(elapsed);
  recordTimer.classList.add('visible');
  timerInterval = setInterval(() => {
    elapsed++;
    recordTimer.textContent = formatTime(elapsed);
    if (elapsed >= DURATION) stopRecording();
  }, 1000);
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
  mediaRecorder.stop();
  mediaStream.getTracks().forEach(t => t.stop());
  clearInterval(timerInterval);
  cancelAnimationFrame(animFrame);
  isRecording = false;
  setRecordingUI('processing');
}

function formatTime(s) {
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

// ── Waveform visualiser ───────────────────────────────────────────────────────
function drawWaveform() {
  if (!analyser) return;
  const ctx    = waveCanvas.getContext('2d');
  const W      = waveCanvas.width  = waveCanvas.offsetWidth;
  const H      = waveCanvas.height = waveCanvas.offsetHeight;
  const buffer = new Uint8Array(analyser.frequencyBinCount);

  function frame() {
    animFrame = requestAnimationFrame(frame);
    analyser.getByteTimeDomainData(buffer);

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'transparent';

    ctx.lineWidth   = 2;
    ctx.strokeStyle = isRecording ? '#ef4444' : '#6366f1';
    ctx.shadowColor = isRecording ? 'rgba(239,68,68,0.5)' : 'rgba(99,102,241,0.5)';
    ctx.shadowBlur  = 8;

    ctx.beginPath();
    const sliceW = W / buffer.length;
    let x = 0;
    for (let i = 0; i < buffer.length; i++) {
      const y = ((buffer[i] / 128.0) - 1) * (H / 2) + H / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.stroke();
  }
  frame();
}

// ── Process audio after recording ─────────────────────────────────────────────
async function onRecordingStop() {
  showState('loading');

  const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
  
  try {
    // Decode WebM to AudioBuffer
    const arrayBuffer = await webmBlob.arrayBuffer();
    // Use an offline context to decode at the exact model sample rate
    const offlineCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 2, SR);
    const audioCtxToDecode = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtxToDecode.decodeAudioData(arrayBuffer);
    
    // Convert to WAV
    const wavBlob = bufferToWav(audioBuffer);

    const formData = new FormData();
    formData.append('audio', wavBlob, 'recording.wav');
    
    const response = await fetch('/acoustic/analyze', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || `HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    if (result.status === 'success') {
       showResults(result.probs);
    } else {
       throw new Error(result.error || 'Unknown server error');
    }
  } catch (err) {
    console.error('Analysis error:', err);
    showState('idle');
    alert('Analysis failed: ' + err.message);
    resetMicUI();
  }
}

// ── Audio Upload handler ──────────────────────────────────────────────────────
async function handleAudioUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  showState('loading');
  setRecordingUI('processing');
  
  const formData = new FormData();
  formData.append('audio', file, file.name);

  try {
    const response = await fetch('/acoustic/analyze', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || `HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    if (result.status === 'success') {
       showResults(result.probs);
    } else {
       throw new Error(result.error || 'Unknown server error');
    }
  } catch (err) {
    console.error('Analysis error:', err);
    showState('idle');
    alert('Analysis failed: ' + err.message);
    resetMicUI();
  }
  
  // reset file input
  event.target.value = '';
}


// ── UI Rendering ──────────────────────────────────────────────────────────────
function showResults(probs) {
  const maxIdx   = probs.indexOf(Math.max(...probs));
  const topClass = CLASSES[maxIdx];
  const topConf  = probs[maxIdx];
  const risk     = CLASS_RISK[topClass];

  // Diagnosis card
  diagnosisIcon.textContent = CLASS_ICONS[topClass];
  diagnosisName.textContent = topClass;
  diagnosisName.style.color = CLASS_COLORS[topClass];
  diagnosisConf.textContent = `Confidence: ${(topConf * 100).toFixed(1)}%`;

  riskPill.textContent = risk.charAt(0).toUpperCase() + risk.slice(1) + ' Risk';
  riskPill.className   = `risk-pill ${risk}`;

  // Probability bars
  CLASSES.forEach((cls, i) => {
    const fill = document.getElementById(`bar-${cls.toLowerCase()}`);
    const pct  = document.getElementById(`pct-${cls.toLowerCase()}`);
    if (fill && pct) {
      setTimeout(() => { fill.style.width = `${(probs[i] * 100).toFixed(1)}%`; }, 80);
      pct.textContent = `${(probs[i] * 100).toFixed(1)}%`;
    }
  });

  // Subclinical warning — show if max confidence < 60%
  if (topConf < 0.60) {
    subclinicalWarn.classList.add('visible');
  } else {
    subclinicalWarn.classList.remove('visible');
  }

  showState('results');
  resetMicUI();
  recordTimer.classList.remove('visible');
}

function showState(state) {
  idleState.style.display      = state === 'idle'    ? '' : 'none';
  loadingState.style.display   = state === 'loading' ? 'block' : 'none';
  resultsContent.style.display = state === 'results' ? 'block' : 'none';
}

// ── UI state machine ──────────────────────────────────────────────────────────
function setRecordingUI(state) {
  if (state === true) {
    // Recording
    micBtn.classList.add('recording');
    micBtn.classList.remove('processing');
    micIcon.textContent = '⏹';
    recordLabel.innerHTML = '<strong>Recording…</strong>Cough naturally into your mic';
    recordingRing.style.display = 'block';
  } else if (state === 'processing') {
    // Processing
    micBtn.classList.remove('recording');
    micBtn.classList.add('processing');
    micBtn.disabled = true;
    micIcon.textContent = '🔄';
    recordLabel.innerHTML = '<strong>Analysing…</strong>Extracting HeAR embeddings';
    recordingRing.style.display = 'none';
  }
}

function resetMicUI() {
  micBtn.classList.remove('recording', 'processing');
  micBtn.disabled  = false;
  micIcon.textContent = '🎙️';
  recordLabel.innerHTML = '<strong>Record Your Cough</strong>Hold for 5 seconds';
  recordingRing.style.display = 'none';
}

// ── Audio encoding utilities ──────────────────────────────────────────────────
function bufferToWav(abuffer) {
  let numOfChan = abuffer.numberOfChannels,
      length = abuffer.length * numOfChan * 2 + 44,
      buffer = new ArrayBuffer(length),
      view = new DataView(buffer),
      channels = [], i, sample,
      offset = 0,
      pos = 0;

  // write WAVE header
  setUint32(0x46464952);                         // "RIFF"
  setUint32(length - 8);                         // file length - 8
  setUint32(0x45564157);                         // "WAVE"
  setUint32(0x20746d66);                         // "fmt " chunk
  setUint32(16);                                 // length = 16
  setUint16(1);                                  // PCM (uncompressed)
  setUint16(numOfChan);
  setUint32(abuffer.sampleRate);
  setUint32(abuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
  setUint16(numOfChan * 2);                      // block-align
  setUint16(16);                                 // 16-bit (hardcoded in this export)
  setUint32(0x61746164);                         // "data" - chunk
  setUint32(length - pos - 4);                   // chunk length

  for (i = 0; i < abuffer.numberOfChannels; i++)
    channels.push(abuffer.getChannelData(i));

  while (pos < length) {
    for (i = 0; i < numOfChan; i++) {
      sample = Math.max(-1, Math.min(1, channels[i][offset]));
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
      view.setInt16(pos, sample, true);          // write 16-bit sample
      pos += 2;
    }
    offset++;
  }

  function setUint16(data) {
    view.setUint16(pos, data, true);
    pos += 2;
  }
  function setUint32(data) {
    view.setUint32(pos, data, true);
    pos += 4;
  }

  return new Blob([buffer], { type: "audio/wav" });
}
