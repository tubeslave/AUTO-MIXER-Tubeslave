import numpy as np
from audio_workbench.mixing import dynamics
from audio_workbench.quality_loop import evaluate_candidate


def test_compression_metadata_blocks_machine_only_baseline_promotion():
    x = (.2 * np.sin(2 * np.pi * 997 * np.arange(48000) / 48000)).astype("float32")
    _, report = dynamics.apply(x, 48000, "vocal")
    plan = {"confidence": {"cause": .9, "intervention": .9}}
    result = evaluate_candidate(plan, {"id": "compression-v2", **report}, True, [], .95)
    assert report["requires_human_review"] is True
    assert "balance_match_passed" in report
    assert result["accepted"] is False
    assert result["acceptance_state"] == "pending_human_review"
