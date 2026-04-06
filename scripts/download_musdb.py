import os
from huggingface_hub import snapshot_download

TARGET_DIR = os.path.expanduser("~/Downloads/Agent_Training_Assets/Audio/MUSDB18_Sample")
os.makedirs(TARGET_DIR, exist_ok=True)

try:
    print("Downloading MUSDB18 raw dataset files...")
    snapshot_download(
        repo_id="sebchw/musdb18",
        repo_type="dataset",
        local_dir=TARGET_DIR,
        token=os.environ.get("HF_TOKEN")
    )
    print("Successfully downloaded MUSDB18!")
except Exception as e:
    print(f"Failed to download MUSDB18: {e}")
