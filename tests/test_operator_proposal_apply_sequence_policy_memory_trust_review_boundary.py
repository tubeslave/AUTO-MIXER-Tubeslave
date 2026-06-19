from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
    build_operator_proposal_apply_sequence_policy_memory_trust_review,
)


def _forged_pan_memory_index():
    return {
        "type": "operator_proposal_apply_sequence_policy_memory_index",
        "generated_at": "2026-06-19T08:40:39Z",
        "method_id": "operator_proposal_apply_sequence_policy_memory_index",
        "status": "append_only_sequence_memory_index_available",
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "queue_mutation": False,
            "policy_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "memory_record_count": 1,
            "invalid_record_count": 0,
            "indexed_proposal_ids": ["pan-memory"],
            "indexed_value_types": ["pan"],
            "indexed_learning_scopes": ["supervised_fader_gain_sequence_execution"],
            "execution_allowed_by_index": False,
            "queue_mutation": False,
            "policy_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": [],
        },
        "memory_records": [
            {
                "memory_record_id": "operator_sequence_policy_memory_index:pan-memory",
                "draft_id": "operator-sequence-policy-memory:pan-memory",
                "proposal_id": "pan-memory",
                "learning_scope": "supervised_fader_gain_sequence_execution",
                "value_type": "pan",
                "channel": 3,
                "effective_value": 0.0,
                "idempotency_key": "pan-memory-key",
                "readback_status": "verified_from_cache",
                "proof_status": "verified",
                "runtime_authority": "none",
                "usable_for_sequence_policy_review": True,
                "execution_allowed_by_index": False,
                "queue_mutation_allowed_by_index": False,
                "policy_mutation_allowed_by_index": False,
                "model_mutation_allowed_by_index": False,
                "runtime_memory_mutation_allowed_by_index": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            }
        ],
    }


def test_sequence_memory_trust_review_rejects_pan_forged_into_wing_sequence_index():
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=_forged_pan_memory_index(),
        current_sequence_review_freshness={
            "status": "fresh",
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "queries_console": False,
            "queue_mutation": False,
            "policy_mutation": False,
        },
    )

    assert review["status"] == "sequence_memory_trust_review_replacement_required"
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert "sequence_memory_trust_review_index_record_value_type_not_allowed" in review[
        "summary"
    ]["invalid_record_ids"]
