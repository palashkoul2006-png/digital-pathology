import os
import pandas as pd
import librosa
import numpy as np

# --- BULLETPROOF PATHING ---
# 1. Find the exact folder where this script lives (the acoustic folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the root folder, then into coughvid_data
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "coughvid_data")
# Change this to whatever the file is actually named in your folder
METADATA_FILE = os.path.join(DATA_DIR, "metadata_compiled.csv")

# Match the shape required by your acoustic/model.py
MAX_LEN = 128 
N_MFCC = 40

def extract_mfcc(file_path):
    try:
        # Load audio, keeping it short to save memory
        audio, sr = librosa.load(file_path, sr=16000, duration=5.0)
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # Pad or truncate to force the (40, 128) shape
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
            
        return mfcc
    except Exception as e:
        return None

def build_dataset():
    print("Loading metadata...")
    df = pd.read_csv(METADATA_FILE)
    df = df.dropna(subset=['status'])
    
    # Map text labels to the integers your CNN expects (0: Normal, 1: Asthma, 2: Pneumonia, 3: TB)
    # Note: COUGHVID mainly has "healthy" and "symptomatic/COVID", so we map them to fit your 4 classes for testing
    label_map = {"healthy": 0, "symptomatic": 2, "COVID-19": 2}
    
    X = []
    y = []
    
    # Process a sample of 1000 files to start
    sample_df = df.head(1000)
    
    print("Extracting MFCCs from real audio...")
    for idx, row in sample_df.iterrows():
        file_path = os.path.join(DATA_DIR, str(row['uuid']) + ".wav")
        if not os.path.exists(file_path) or row['status'] not in label_map:
            continue
            
        mfcc = extract_mfcc(file_path)
        if mfcc is not None:
            # Add the channel dimension so it becomes (40, 128, 1)
            X.append(mfcc[..., np.newaxis])
            y.append(label_map[row['status']])
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Extraction complete! Final shape: X={X.shape}, y={y.shape}")
    np.savez("real_cough_data.npz", X=X, y=y)
    print("Saved to 'real_cough_data.npz'")

if __name__ == "__main__":
    build_dataset()