import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_workbench.project import create_manifest, load_manifest, update_context

def _write(path: Path, sr=48000, sec=0.25):
    x = np.zeros((int(sr*sec), 2), dtype="float32")
    sf.write(path, x, sr, subtype="FLOAT")

def test_create_manifest_and_context():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stems = root / "stems"
        stems.mkdir()
        _write(stems / "Kick In.wav")
        _write(stems / "Lead Vocal.wav")
        m = create_manifest(str(root / "workbench"), str(stems), "Song")
        assert m["mode"] == "dawless"
        assert m["track_count"] == 2
        roles = {t["role_guess"] for t in m["tracks"]}
        assert "kick" in roles
        assert "vocal" in roles
        update_context(str(root / "workbench"),
                       sections=[{"name":"verse","start_s":0,"end_s":10}],
                       references=["ref.wav"],
                       notes=["keep vocal forward"])
        m2 = load_manifest(str(root / "workbench"))
        assert m2["sections"][0]["name"] == "verse"
        assert m2["references"] == ["ref.wav"]
