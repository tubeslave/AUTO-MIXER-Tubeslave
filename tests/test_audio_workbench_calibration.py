import tempfile
from pathlib import Path
from audio_workbench.calibration import make_observer_exam

def test_exam_requires_capability_and_status():
    with tempfile.TemporaryDirectory() as td:
        p=make_observer_exam(td,"x.wav",[
          {"type":"gain","params":{"db":6},"capability":"level_direction","expected_detection":"louder"}
        ])
        assert p["plan"]["tests"][0]["status"]=="not_run"
        assert "identical-file catch trials are required" in p["plan"]["rules"]
        assert Path(p["path"]).exists()
