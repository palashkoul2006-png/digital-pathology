import os

# 🚨 CRITICAL FIX: Set the token BEFORE importing the Kaggle library 🚨
os.environ['KAGGLE_API_TOKEN'] = "KGAT_aab30dcbe5c7ecbf13305dd93f3224d5"

# Now Kaggle will see the token immediately when it wakes up
from kaggle.api.kaggle_api_extended import KaggleApi

def fetch_coughvid_data(target_dir="./coughvid_data"):
    dataset_slug = "nasrulhakim86/coughvid-wav"
    
    os.makedirs(target_dir, exist_ok=True)
    print(f"Directory ready: {target_dir}")
    
    try:
        print("Authenticating with Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading '{dataset_slug}'... (This may take a few minutes)")
        api.dataset_download_cli(dataset_slug, path=target_dir, unzip=True)
        
        wav_files = [f for f in os.listdir(target_dir) if f.endswith('.wav')]
        print(f"Success! Extracted {len(wav_files)} .wav files into '{target_dir}'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_coughvid_data()