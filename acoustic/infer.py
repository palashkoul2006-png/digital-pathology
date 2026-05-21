import os
import sys
import json
import logging
import warnings
import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    import librosa
    from hear_pipeline import extract_hear_embedding
except Exception as e:
    print(json.dumps({"error": f"Failed to import dependencies: {str(e)}"}))
    sys.exit(1)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAR_MODEL_PATH = os.path.join(BASE_DIR, 'acoustic', 'saved_model', 'hear_mlp.h5')
MFCC_MODEL_PATH = os.path.join(BASE_DIR, 'acoustic', 'saved_model', 'final_model.keras')

CLASSES = ['Normal', 'Asthma', 'Pneumonia', 'TB']
MAX_LEN = 128
N_MFCC = 40

def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5.0)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
            
        return mfcc
    except Exception as e:
        raise RuntimeError(f"MFCC extraction failed: {str(e)}")

def run_inference(audio_path):
    try:
        if not os.path.exists(HEAR_MODEL_PATH):
            raise FileNotFoundError(f"HeAR model not found at {HEAR_MODEL_PATH}")
        if not os.path.exists(MFCC_MODEL_PATH):
            raise FileNotFoundError(f"MFCC model not found at {MFCC_MODEL_PATH}")
            
        hear_model = tf.keras.models.load_model(HEAR_MODEL_PATH)
        mfcc_model = tf.keras.models.load_model(MFCC_MODEL_PATH)
        
        # 1. HeAR Pipeline
        hear_features = extract_hear_embedding(audio_path)
        hear_features = np.expand_dims(hear_features, axis=0)
        hear_preds = hear_model.predict(hear_features, verbose=0)[0]
        
        # 2. MFCC Pipeline
        mfcc_features = extract_mfcc(audio_path)
        mfcc_features = np.expand_dims(mfcc_features, axis=-1) # Add channel: (40, 128, 1)
        mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch: (1, 40, 128, 1)
        mfcc_preds = mfcc_model.predict(mfcc_features, verbose=0)[0]
        
        # 3. Ensemble (Average)
        ensemble_preds = (hear_preds + mfcc_preds) / 2.0
        
        # Format results
        probs = ensemble_preds.tolist()
        max_idx = int(np.argmax(ensemble_preds))
        
        result = {
            "status": "success",
            "probs": probs,
            "classes": CLASSES,
            "prediction": CLASSES[max_idx],
            "confidence": probs[max_idx],
            "details": {
                "hear_confidence": float(hear_preds[max_idx]),
                "mfcc_confidence": float(mfcc_preds[max_idx])
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No audio path provided"}))
        sys.exit(1)
        
    run_inference(sys.argv[1])
