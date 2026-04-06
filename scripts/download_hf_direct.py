import os
from huggingface_hub import snapshot_download

HF_DIR = os.path.expanduser("~/Downloads/Agent_Training_Assets/Metadata/SonicMasterDataset")
os.makedirs(HF_DIR, exist_ok=True)

try:
    print("Downloading SonicMasterDataset raw files directly...")
    snapshot_download(
        repo_id="amaai-lab/SonicMasterDataset",
        repo_type="dataset",
        local_dir=HF_DIR,
        allow_patterns=["*.parquet", "*.csv", "*.json"],
        token=os.environ.get("HF_TOKEN")
    )
    print(f"Successfully downloaded to {HF_DIR}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
