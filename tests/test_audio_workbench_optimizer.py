import tempfile
from audio_workbench.optimizer import create_multiobjective_study, pareto_trials

def test_create_study():
    with tempfile.TemporaryDirectory() as td:
        r=create_multiobjective_study(td,"test",["minimize","maximize"],["damage","target"])
        assert r["metric_names"]==["damage","target"]
