import os
from datasets import load_dataset
import pandas as pd

HF_DIR = os.path.expanduser("~/Downloads/Agent_Training_Assets/Metadata")
os.makedirs(HF_DIR, exist_ok=True)

print("Downloading MixAssist dataset...")
try:
    ds_mixassist = load_dataset("Qwen/MixAssist", trust_remote_code=True)
    # This might fail if Qwen/MixAssist doesn't exist. Wait, the paper was just published. The actual repo might be "Qwen/MixAssist" or something else. I'll search or just handle errors.
    print(ds_mixassist)
except Exception as e:
    print(f"MixAssist not found or error: {e}. Will attempt another repo or skip.")

print("Downloading SonicMasterDataset...")
try:
    ds_sonic = load_dataset("amaai-lab/SonicMasterDataset", trust_remote_code=True)
    # Export SonicMaster subset
    df = ds_sonic['train'].to_pandas()
    # Save a sample to CSV
    csv_path = os.path.join(HF_DIR, "SonicMasterDataset_sample.csv")
    df.head(1000).to_csv(csv_path, index=False)
    print(f"Saved SonicMaster sample to {csv_path}")
except Exception as e:
    print(f"SonicMaster not found or error: {e}.")

print("Done downloading HuggingFace datasets.")
