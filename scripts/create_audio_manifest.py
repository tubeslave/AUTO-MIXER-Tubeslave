import os
import json
from datetime import datetime

AUDIO_DIR = os.path.expanduser("~/Downloads/Agent_Training_Assets/Audio")
MANIFEST_PATH = os.path.join(AUDIO_DIR, "manifest.json")

def create_manifest():
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR, exist_ok=True)
        
    rows = []
    
    # We pretend the downloaded MUSDB or Cambridge files are here
    for root, dirs, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.endswith('.wav') or f.endswith('.npz'):
                rel_path = os.path.relpath(os.path.join(root, f), AUDIO_DIR)
                rows.append({"rel_path": rel_path, "weight": 1.0})
    
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "Cambridge_MT_and_MUSDB_Subset",
        "dataset_version": "1.0",
        "task": "differentiable_mix_console",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "backend_git_sha": "unknown",
        "config_digest": "none",
        "safety_filter_applied": False,
        "license": "educational_only",
        "rows": rows,
        "feature_spec": {
            "sample_rate": 44100,
            "n_channels_max": 24
        }
    }
    
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Created manifest.json with {len(rows)} files at {MANIFEST_PATH}")

if __name__ == "__main__":
    create_manifest()
