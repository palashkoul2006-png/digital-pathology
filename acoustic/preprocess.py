"""
acoustic/preprocess.py — Task A
================================
MFCC extraction pipeline for cough audio classification.

Features:
  - Load real audio files (.wav, .mp3, .ogg, .flac) via librosa
  - Extract 40 MFCCs at sr=22050 Hz, hop_length=512
  - Pad / truncate to fixed 128 time-frames → shape (40, 128, 1)
  - generate_dummy_dataset() creates synthetic audio for instant testing
  - Saves processed arrays to acoustic/data/X.npy & y.npy
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants (must match model input & JS pipeline) ──────────────────────────
SAMPLE_RATE   = 22050          # Hz
DURATION      = 5              # seconds of audio to analyse
N_MFCC        = 40             # number of MFCC coefficients
HOP_LENGTH    = 512            # samples between frames
N_FFT         = 2048           # FFT window size
MAX_FRAMES    = 128            # fixed time-axis length (pad/truncate)
INPUT_SHAPE   = (N_MFCC, MAX_FRAMES, 1)   # CNN input

CLASSES       = ['Normal', 'Asthma', 'Pneumonia', 'TB']
NUM_CLASSES   = len(CLASSES)

DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ── Core MFCC extraction ──────────────────────────────────────────────────────

def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract MFCCs from a raw audio array.

    Parameters
    ----------
    audio : np.ndarray  shape (n_samples,)
    sr    : int         sample rate

    Returns
    -------
    np.ndarray  shape (N_MFCC, MAX_FRAMES, 1) — CNN-ready
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    # Ensure mono
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Trim / pad to DURATION seconds
    target_len = SAMPLE_RATE * DURATION
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )   # shape: (N_MFCC, T)

    # Pad or truncate time axis to MAX_FRAMES
    if mfcc.shape[1] < MAX_FRAMES:
        pad = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    # Normalize per feature to [-1, 1]
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (
        mfcc.std(axis=1, keepdims=True) + 1e-8
    )

    return mfcc[..., np.newaxis]   # (40, 128, 1)


def load_audio_file(path: str) -> np.ndarray:
    """Load an audio file and return MFCC tensor."""
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    audio, sr = librosa.load(path, sr=None, mono=True, duration=DURATION + 1)
    logger.info(f"Loaded {path}  sr={sr}  samples={len(audio)}")
    return extract_mfcc(audio, sr)


# ── Dummy / Synthetic data generator ─────────────────────────────────────────

def _make_class_audio(class_idx: int, sr: int = SAMPLE_RATE,
                      duration: float = DURATION) -> np.ndarray:
    """
    Generate synthetic audio that mimics class-specific spectral signatures:

    Normal    — low-amplitude broadband noise (resting breath)
    Asthma    — wheezing: narrow-band high-freq tones + modulation
    Pneumonia — wet crackles: short amplitude bursts on noise
    TB        — dry cough: periodic chirp bursts with strong low-freq energy
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(seed=class_idx * 42)

    if class_idx == 0:           # Normal
        audio = rng.normal(0, 0.02, n)

    elif class_idx == 1:         # Asthma — wheeze: AM-modulated sine
        f0 = rng.uniform(400, 800)
        mod = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))  # 5 Hz amplitude mod
        audio = mod * np.sin(2 * np.pi * f0 * t) * 0.3
        audio += rng.normal(0, 0.01, n)

    elif class_idx == 2:         # Pneumonia — wet crackles
        audio = rng.normal(0, 0.03, n)
        # Random burst positions
        burst_times = rng.integers(0, n - sr // 10, size=20)
        for bt in burst_times:
            burst_len = rng.integers(200, 800)
            end = min(bt + burst_len, n)
            audio[bt:end] += rng.normal(0, 0.2, end - bt)

    else:                        # TB — periodic dry cough chirps
        audio = rng.normal(0, 0.01, n)
        cough_interval = int(sr * 0.8)   # every 0.8 s
        for start in range(0, n - sr // 4, cough_interval):
            chirp_len = int(sr * 0.25)
            end = min(start + chirp_len, n)
            chirp_t = np.linspace(0, 1, end - start)
            f_start, f_end = 80, 300
            freq = f_start + (f_end - f_start) * chirp_t
            chirp = np.sin(2 * np.pi * freq * chirp_t) * 0.5
            env = np.hanning(end - start)
            audio[start:end] += chirp * env

    # Normalise to [-1, 1]
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    return audio.astype(np.float32)


def generate_dummy_dataset(n_samples: int = 200,
                            save: bool = True) -> tuple:
    """
    Generate n_samples synthetic cough recordings (n_samples / 4 per class).

    Parameters
    ----------
    n_samples : int   total samples (divisible by NUM_CLASSES recommended)
    save      : bool  if True, saves X.npy and y.npy to DATA_DIR

    Returns
    -------
    X : np.ndarray  shape (n_samples, N_MFCC, MAX_FRAMES, 1)
    y : np.ndarray  shape (n_samples,)  integer class labels
    """
    logger.info(f"Generating {n_samples} synthetic audio samples …")
    per_class = n_samples // NUM_CLASSES
    X_list, y_list = [], []

    for cls_idx in range(NUM_CLASSES):
        for i in range(per_class):
            # Slight seed variation per sample
            np.random.seed(cls_idx * 1000 + i)
            audio = _make_class_audio(cls_idx)
            # Add slight random perturbation
            audio += np.random.normal(0, 0.005, len(audio))
            mfcc = extract_mfcc(audio)
            X_list.append(mfcc)
            y_list.append(cls_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    if save:
        np.save(os.path.join(DATA_DIR, 'X.npy'), X)
        np.save(os.path.join(DATA_DIR, 'y.npy'), y)
        logger.info(f"Saved X.npy {X.shape} and y.npy {y.shape} to {DATA_DIR}")

    return X, y


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    X, y = generate_dummy_dataset(n_samples=40, save=True)
    print(f"X shape: {X.shape}  dtype: {X.dtype}")
    print(f"y shape: {y.shape}  classes: {np.unique(y)}")
    print(f"MFCC range: [{X.min():.3f}, {X.max():.3f}]")
    print("✅ Preprocessing pipeline OK")
