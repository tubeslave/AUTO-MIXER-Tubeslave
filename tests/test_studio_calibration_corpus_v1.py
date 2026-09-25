import json

import pytest

from audio_workbench.mixing.calibration_corpus import (
    append_calibration_record,
    build_calibration_record,
    calibration_record_id,
    load_calibration_corpus,
    replay_calibration_corpus,
    replay_calibration_record,
)


def _source(n=1):
    return {
        "machine_state_path": f".ai/state/machine-{n}.json",
        "machine_state_git_blob_sha": f"{n:040x}",
        "human_feedback_path": f".ai/feedback/human-{n}.json",
        "human_feedback_git_blob_sha": f"{n + 100:040x}",
    }


def _subject(n=1):
    return {"song": "Belye Stai", "candidate": f"candidate-{n}"}


def _critic(*, protected=None, failures=None, decision="rejected"):
    return {
        "target": "vocal_intelligibility",
        "target_before": 0.6800658645341687,
        "target_after": 0.6714029434943347,
        "target_improvement": -0.008662921039834015,
        "machine_decision": decision,
        "failures": list(failures or ["target_not_improved"]),
        "protected_regressions": list(protected or []),
    }


def _review(status="accepted"):
    return {
        "human_review": status,
        "observations": [
            "foreground_stability",
            "phrase_consistency",
            "stable_mix_position",
        ],
    }


def _record(n=1, previous=None, **critic_kwargs):
    return build_calibration_record(
        source_evidence=_source(n),
        subject=_subject(n),
        critic_result=_critic(**critic_kwargs),
        human_review=_review(),
        previous_record_id=previous,
    )


def test_record_construction_is_deterministic_and_evidence_only():
    one = _record()
    two = _record()
    assert one == two
    assert one["record_id"] == calibration_record_id(one)
    assert one["replay_result"]["classification"] == "target_proxy_miss_candidate"
    assert one["replay_result"]["metric_hypotheses"] == ["foreground_stability"]
    assert one["production_threshold_update_allowed"] is False
    assert one["protected_gate_override_allowed"] is False
    assert one["baseline_promotion_allowed"] is False
    assert one["requires_human_listening"] is True


def test_two_record_digest_chain_replays_deterministically():
    first = _record(1)
    second = _record(2, previous=first["record_id"])
    summary = replay_calibration_corpus([first, second])
    assert summary["records"] == 2
    assert summary["last_record_id"] == second["record_id"]
    assert summary["classifications"] == {"target_proxy_miss_candidate": 2}
    assert summary["baseline_promotion_allowed"] is False


def test_record_mutation_is_detected_by_digest():
    record = _record()
    record["subject"]["candidate"] = "silently-mutated"
    with pytest.raises(ValueError, match="digest mismatch"):
        replay_calibration_record(record, expected_previous_record_id=None)


def test_duplicate_append_rejected_without_changing_prior_bytes(tmp_path):
    path = tmp_path / "corpus.jsonl"
    append_calibration_record(
        path,
        source_evidence=_source(1),
        subject=_subject(1),
        critic_result=_critic(),
        human_review=_review(),
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        append_calibration_record(
            path,
            source_evidence=_source(1),
            subject={"song": "Belye Stai", "candidate": "renamed"},
            critic_result=_critic(),
            human_review=_review(),
        )
    assert path.read_bytes() == before


def test_corrupt_existing_prefix_blocks_append_without_rewrite(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"broken":true}\n', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        append_calibration_record(
            path,
            source_evidence=_source(2),
            subject=_subject(2),
            critic_result=_critic(),
            human_review=_review(),
        )
    assert path.read_bytes() == before


def test_replay_rejects_semantic_annotation_drift_even_with_fresh_digest():
    record = _record()
    record["replay_result"]["classification"] = "human_machine_agreement_positive"
    record["record_id"] = calibration_record_id(record)
    with pytest.raises(ValueError, match="classification drift"):
        replay_calibration_record(record, expected_previous_record_id=None)


def test_protected_regression_conflict_is_preserved_by_corpus_replay():
    record = _record(
        protected=["width_regression"],
        failures=["target_not_improved", "width_regression"],
    )
    assert record["replay_result"]["classification"] == "human_preference_safety_conflict"
    assert record["replay_result"]["protected_regressions_preserved"] == ["width_regression"]
    replay_calibration_record(record, expected_previous_record_id=None)


def test_loader_requires_canonical_jsonl(tmp_path):
    record = _record()
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2|canonical"):
        load_calibration_corpus(path)


def test_append_load_and_full_replay_round_trip(tmp_path):
    path = tmp_path / "corpus.jsonl"
    first = append_calibration_record(
        path,
        source_evidence=_source(1),
        subject=_subject(1),
        critic_result=_critic(),
        human_review=_review(),
    )
    second = append_calibration_record(
        path,
        source_evidence=_source(2),
        subject=_subject(2),
        critic_result=_critic(decision="machine_safe", failures=[]),
        human_review=_review(status="rejected"),
    )
    loaded = load_calibration_corpus(path)
    summary = replay_calibration_corpus(loaded)
    assert [r["record_id"] for r in loaded] == [first["record_id"], second["record_id"]]
    assert summary["records"] == 2
    assert summary["classifications"]["target_proxy_miss_candidate"] == 1
    assert summary["classifications"]["machine_false_positive_candidate"] == 1
