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
  setModelStatus('⏳ Loading AI model…', '#f59e0b');
  try {
    await loadModel();
  } catch (err) {
    setModelStatus('❌ Model not found. Run export_model.py first.', '#ef4444');
    console.error('Model load error:', err);
    // Still allow button — will show friendly error
  }
  micBtn.addEventListener('click', handleMicClick);
}

// ── TF.js Model Loading ───────────────────────────────────────────────────────
async function loadModel() {
  if (typeof tf === 'undefined') {
    throw new Error('TensorFlow.js not loaded');
  }
  tfModel = await tf.loadLayersModel(MODEL_URL);
  // Warm-up pass to compile shaders
  const dummy = tf.zeros([1, N_MFCC, MAX_FRAMES, 1]);
  tfModel.predict(dummy).dispose();
  dummy.dispose();
  setModelStatus('✅ Edge AI model ready — runs 100% locally', '#00d4aa');
  micBtn.disabled = false;
  console.log('CoughCNN loaded:', tfModel.inputs[0].shape);
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

  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  try {
    const mfcc   = await extractMFCC(blob);
    const probs  = await runInference(mfcc);
    showResults(probs);
  } catch (err) {
    console.error('Analysis error:', err);
    showState('idle');
    alert('Analysis failed: ' + err.message);
    resetMicUI();
  }
}

// ── Client-side MFCC extraction ───────────────────────────────────────────────
/**
 * extractMFCC(blob) → Float32Array of shape [N_MFCC × MAX_FRAMES]
 *
 * Pipeline:
 *   1. Decode audio blob with AudioContext
 *   2. Resample to SR if needed
 *   3. Compute STFT power spectrum using AnalyserNode (or manual FFT)
 *   4. Apply Mel filterbank
 *   5. Take log
 *   6. Apply DCT → MFCCs
 *   7. Pad / truncate to MAX_FRAMES
 *   8. Normalize
 */
async function extractMFCC(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const offlineCtx  = new OfflineAudioContext(1, SR * DURATION, SR);
  let audioBuffer;
  try {
    audioBuffer = await offlineCtx.decodeAudioData(arrayBuffer);
  } catch {
    // Fallback: use a fresh context
    const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioBuffer  = await tmpCtx.decodeAudioData(arrayBuffer.slice(0));
    tmpCtx.close();
  }

  // Get mono PCM, resampled to SR
  const pcm = resampleMono(audioBuffer, SR);

  // Trim / pad to exactly DURATION seconds
  const targetLen = SR * DURATION;
  let samples = new Float32Array(targetLen);
  samples.set(pcm.slice(0, targetLen));  // zero-pads automatically

  return computeMFCC(samples);
}

function resampleMono(audioBuffer, targetSR) {
  const src = audioBuffer.getChannelData(0);
  if (audioBuffer.sampleRate === targetSR) return src;
  const ratio  = audioBuffer.sampleRate / targetSR;
  const outLen = Math.floor(src.length / ratio);
  const out    = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const lo  = Math.floor(pos);
    const hi  = Math.min(lo + 1, src.length - 1);
    const t   = pos - lo;
    out[i] = src[lo] * (1 - t) + src[hi] * t;
  }
  return out;
}

/**
 * computeMFCC — pure JS, no external library
 * Returns Float32Array, length = N_MFCC * MAX_FRAMES (row-major)
 */
function computeMFCC(samples) {
  const hopLen    = HOP_LENGTH;
  const fftSize   = N_FFT;
  const nMFCC     = N_MFCC;
  const nFrames   = MAX_FRAMES;

  // Pre-compute Hann window
  const window_ = hanningWindow(fftSize);

  // Mel filterbank
  const melFilters = buildMelFilterbank(fftSize, SR, 0, SR / 2, 128);

  // Output: [N_MFCC × MAX_FRAMES]
  const mfccMatrix = new Float32Array(nMFCC * nFrames);

  // Frame-by-frame
  const numFrames = Math.floor((samples.length - fftSize) / hopLen) + 1;

  for (let fi = 0; fi < nFrames; fi++) {
    const start = fi * hopLen;
    const frame = new Float32Array(fftSize);

    // Copy + window
    for (let i = 0; i < fftSize; i++) {
      const si = start + i;
      frame[i] = (si < samples.length ? samples[si] : 0) * window_[i];
    }

    // FFT → power spectrum
    const spectrum = powerSpectrum(frame);

    // Apply Mel filterbank → log
    const logMel = new Float32Array(melFilters.length);
    for (let m = 0; m < melFilters.length; m++) {
      let energy = 0;
      for (let k = 0; k < spectrum.length; k++) energy += melFilters[m][k] * spectrum[k];
      logMel[m] = Math.log(Math.max(energy, 1e-10));
    }

    // DCT-II → MFCCs (first nMFCC coefficients)
    for (let n = 0; n < nMFCC; n++) {
      let sum = 0;
      for (let m = 0; m < logMel.length; m++) {
        sum += logMel[m] * Math.cos(Math.PI * n * (2 * m + 1) / (2 * logMel.length));
      }
      mfccMatrix[n * nFrames + fi] = sum;
    }
  }

  // Normalize each MFCC coefficient across time
  for (let n = 0; n < nMFCC; n++) {
    let mean = 0, std = 0;
    for (let fi = 0; fi < nFrames; fi++) mean += mfccMatrix[n * nFrames + fi];
    mean /= nFrames;
    for (let fi = 0; fi < nFrames; fi++) std += (mfccMatrix[n * nFrames + fi] - mean) ** 2;
    std = Math.sqrt(std / nFrames + 1e-8);
    for (let fi = 0; fi < nFrames; fi++) {
      mfccMatrix[n * nFrames + fi] = (mfccMatrix[n * nFrames + fi] - mean) / std;
    }
  }

  return mfccMatrix;
}

function hanningWindow(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
  return w;
}

/** DFT-based power spectrum (uses Cooley-Tukey FFT if size is power-of-2) */
function powerSpectrum(frame) {
  const N    = frame.length;
  const real = Array.from(frame);
  const imag = new Array(N).fill(0);
  fftCooleyTukey(real, imag, N);
  const half = Math.floor(N / 2) + 1;
  const ps   = new Float32Array(half);
  for (let i = 0; i < half; i++) ps[i] = real[i] ** 2 + imag[i] ** 2;
  return ps;
}

function fftCooleyTukey(re, im, N) {
  if (N <= 1) return;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  // Cooley-Tukey butterfly
  for (let len = 2; len <= N; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j],          uIm = im[i + j];
        const vRe = re[i + j + len/2] * curRe - im[i + j + len/2] * curIm;
        const vIm = re[i + j + len/2] * curIm + im[i + j + len/2] * curRe;
        re[i + j]         = uRe + vRe; im[i + j]         = uIm + vIm;
        re[i + j + len/2] = uRe - vRe; im[i + j + len/2] = uIm - vIm;
        const tmpRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe; curRe = tmpRe;
      }
    }
  }
}

function buildMelFilterbank(nFFT, sr, fMin, fMax, nFilters) {
  const halfFFT = Math.floor(nFFT / 2) + 1;
  const melMin  = hzToMel(fMin);
  const melMax  = hzToMel(fMax);
  const melPts  = Array.from({ length: nFilters + 2 }, (_, i) =>
    melToHz(melMin + i * (melMax - melMin) / (nFilters + 1))
  );
  // Convert Hz to FFT bin
  const bins = melPts.map(f => Math.floor((nFFT + 1) * f / sr));
  const filters = [];
  for (let m = 1; m <= nFilters; m++) {
    const f = new Float32Array(halfFFT);
    for (let k = bins[m - 1]; k < bins[m]; k++)
      f[k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1] + 1e-8);
    for (let k = bins[m]; k < bins[m + 1]; k++)
      f[k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m] + 1e-8);
    filters.push(f);
  }
  return filters;
}

const hzToMel  = hz  => 2595 * Math.log10(1 + hz / 700);
const melToHz  = mel => 700 * (Math.pow(10, mel / 2595) - 1);

// ── TF.js Inference ───────────────────────────────────────────────────────────
async function runInference(mfccFlat) {
  if (!tfModel) throw new Error('Model not loaded. Run export_model.py first.');

  // mfccFlat: Float32Array [N_MFCC × MAX_FRAMES] row-major
  // Reshape to [1, N_MFCC, MAX_FRAMES, 1]
  const input  = tf.tensor4d(mfccFlat, [1, N_MFCC, MAX_FRAMES, 1]);
  const output = tfModel.predict(input);
  const probs  = await output.data();   // Float32Array length 4
  input.dispose(); output.dispose();
  return Array.from(probs);
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
    recordLabel.innerHTML = '<strong>Analysing…</strong>Extracting spectral features';
    recordingRing.style.display = 'none';
  }
}

function resetMicUI() {
  micBtn.classList.remove('recording', 'processing');
  micBtn.disabled  = !tfModel;
  micIcon.textContent = '🎙️';
  recordLabel.innerHTML = '<strong>Record Your Cough</strong>Hold for 5 seconds';
  recordingRing.style.display = 'none';
}
