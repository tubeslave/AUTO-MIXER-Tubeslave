from backend.processing_registry import (
    LIVE_ALLOWED_WRITE_KINDS,
    PROCESSING_DOMAINS,
    PROCESSING_STATUSES,
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
    iter_pipeline_profiles,
    iter_processing_methods,
)


def test_proposal_adapter_module_stays_free_of_live_mixer_imports():
    from pathlib import Path

    source = Path("backend/processing_proposal_adapters.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "mixer_client.set_",
        ".send(",
        "set_eq(",
        "set_compressor(",
    ]
    for token in forbidden:
        assert token not in source


def test_processing_methods_have_valid_statuses_domains_and_unique_ids():
    methods = list(iter_processing_methods())
    ids = [method.id for method in methods]

    assert len(ids) == len(set(ids))
    assert methods
    for method in methods:
        assert method.status in PROCESSING_STATUSES
        assert method.domain in PROCESSING_DOMAINS
        assert method.module


def test_live_capable_methods_require_supervision_and_only_fader_gain():
    live_methods = [method for method in iter_processing_methods() if method.can_write_live]

    assert [method.id for method in live_methods] == ["supervised_manual_fader_gain"]
    for method in live_methods:
        assert method.requires_supervision is True
        assert tuple(method.live_write_kinds) == LIVE_ALLOWED_WRITE_KINDS
        assert method.status == "canonical"


def test_main_live_pipeline_executor_is_only_supervised_fader_gain():
    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert main.status == "canonical"
    assert main.can_write_live is True
    assert main.requires_supervision is True
    assert tuple(main.live_write_kinds) == LIVE_ALLOWED_WRITE_KINDS
    assert tuple(main.executor_method_ids) == ("supervised_manual_fader_gain",)

    blocked_statuses = {"placeholder", "quarantined", "deprecated"}
    for method_id in main.method_ids:
        method = get_processing_method(method_id)
        assert method is not None
        assert method.status not in blocked_statuses

    context_gate = get_processing_method("context_aware_gate_agent")
    assert context_gate is not None
    assert context_gate.can_propose is True
    assert context_gate.can_write_live is False
    assert context_gate.status == "proposal_only"
    assert "context_aware_gate_agent" in main.proposal_method_ids


def test_secondary_pipelines_do_not_declare_live_write_executor():
    for pipeline_id in ("offline_mix", "shadow_quality", "lab_dsp", "quarantine"):
        profile = get_pipeline_profile(pipeline_id)
        assert profile is not None
        assert profile.can_write_live is False
        assert profile.live_write_kinds == ()


def test_quarantined_and_placeholder_methods_are_not_pipeline_executors():
    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    executor_ids = set(main.executor_method_ids)

    for method in iter_processing_methods():
        if method.status in {"placeholder", "quarantined", "deprecated"}:
            assert method.can_write_live is False
            assert method.id not in executor_ids


def test_registry_snapshot_can_hide_full_method_inventory():
    snapshot = build_processing_registry_snapshot(include_methods=False)

    assert snapshot["type"] == "processing_pipeline_snapshot"
    assert snapshot["summary"]["main_pipeline_id"] == "main_live_assist"
    assert snapshot["summary"]["live_allowed_write_kinds"] == list(LIVE_ALLOWED_WRITE_KINDS)
    assert snapshot["summary"]["live_executor_method_ids"] == ["supervised_manual_fader_gain"]
    assert snapshot["methods"] == []
    assert {profile["id"] for profile in snapshot["pipeline_profiles"]} == {
        profile.id for profile in iter_pipeline_profiles()
    }
