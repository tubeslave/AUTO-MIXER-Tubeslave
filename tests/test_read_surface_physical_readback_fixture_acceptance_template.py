import json
from pathlib import Path

from backend.read_surface_physical_readback_artifact_review_queue import (
    INTAKE_SPECS,
    build_read_surface_physical_readback_artifact_review_queue,
)
from backend.raw_mixer_state_physical_readback_artifact_intake import (
    build_raw_mixer_state_physical_readback_artifact_intake,
)
from backend.read_surface_physical_readback_fixture_acceptance_ledger import (
    build_read_surface_physical_readback_fixture_acceptance_ledger,
)
from tests.test_read_surface_physical_readback_artifact_review_queue import (
    _snapshot_for_spec,
    _snapshot_map,
)
from tests.test_raw_mixer_state_physical_readback_artifact_intake import _valid_artifact
from tools import build_read_surface_physical_readback_acceptance_template as template_tool


def _queue_with_raw_mixer_candidate():
    spec = INTAKE_SPECS[0]
    snapshot = _snapshot_for_spec(spec, valid=1)
    snapshot["summary"]["artifact_source_ids"] = ["raw-mixer-physical.json"]
    queue = build_read_surface_physical_readback_artifact_review_queue(
        intake_snapshots=_snapshot_map(**{spec["method_id"]: snapshot})
    )
    return spec, queue


def test_acceptance_template_reports_missing_candidates_without_authority():
    template = template_tool.build_acceptance_template(
        generated_at="2026-06-07T00:00:00Z"
    )

    assert template["type"] == "read_surface_physical_readback_fixture_acceptance_template"
    assert template["status"] == "operator_candidate_missing"
    assert template["source_queue_status"] == "operator_artifact_capture_required"
    assert template["read_only"] is True
    assert template["live_write"] is False
    assert template["mutates_console_state"] is False
    assert template["queries_console"] is False
    assert template["executes_tests"] is False
    assert template["operator_required"] is True
    assert template["execution_allowed_by_template"] is False
    assert template["fixture_closure_allowed_by_template"] is False
    assert template["live_widening_allowed"] is False
    assert template["live_authority_granted"] is False
    assert template["full_autonomous_ready"] is False
    assert template["review_candidate_count"] == 0
    assert template["template_record_count"] == 0
    assert template["template_records"] == []
    assert "operator_must_capture_missing_physical_readback_artifacts" in template[
        "blocked_by"
    ]


def test_acceptance_template_builds_pending_record_for_valid_candidate():
    spec, queue = _queue_with_raw_mixer_candidate()
    template = template_tool.build_acceptance_template(
        review_queue=queue,
        generated_at="2026-06-07T00:00:00Z",
    )

    assert template["status"] == "operator_acceptance_template_ready"
    assert template["source_queue_status"] == "operator_review_candidates_available"
    assert template["review_candidate_count"] == 1
    assert template["template_record_count"] == 1
    record = template["template_records"][0]
    assert record["surface_id"] == spec["surface_id"]
    assert record["artifact_source_id"] == "raw-mixer-physical.json"
    assert record["required_contract_method_id"] == spec["required_contract_method_id"]
    trace = record["candidate_trace"]
    assert trace["source_queue_method_id"] == queue["method_id"]
    assert trace["source_queue_status"] == queue["status"]
    assert trace["candidate_status"] == "operator_review_candidate_available"
    assert trace["artifact_source_ids"] == ["raw-mixer-physical.json"]
    assert trace["valid_candidate_count"] == 1
    assert trace["invalid_candidate_count"] == 0
    assert trace["target_consistency"] is True
    assert trace["candidate_source_id_missing"] is False
    assert trace["accepted_as_physical_fixture_candidate"] is True
    assert trace["fixture_closure_allowed_by_queue"] is False
    assert trace["live_authority_granted"] is False
    assert trace["full_autonomous_ready"] is False
    assert record["decision"] == "<accepted|rejected>"
    assert record["reviewed_by"] == "<operator-name>"
    assert record["reviewed_at"].startswith("<UTC ISO timestamp")
    assert record["live_write"] is False
    assert record["fixture_closure_allowed_by_acceptance"] is False
    assert record["fixture_closure_allowed"] is False
    assert record["live_authority_granted"] is False
    assert record["full_autonomous_ready"] is False


def test_acceptance_template_uses_real_intake_latest_candidate_source_id():
    spec = INTAKE_SPECS[0]
    intake = build_raw_mixer_state_physical_readback_artifact_intake(
        artifact_payloads=(_valid_artifact(),),
        artifact_roots=(),
    )
    queue = build_read_surface_physical_readback_artifact_review_queue(
        intake_snapshots=_snapshot_map(**{spec["method_id"]: intake})
    )
    template = template_tool.build_acceptance_template(
        review_queue=queue,
        generated_at="2026-06-07T00:00:00Z",
    )

    assert template["status"] == "operator_acceptance_template_ready"
    assert template["template_record_count"] == 1
    record = template["template_records"][0]
    assert record["surface_id"] == spec["surface_id"]
    assert record["artifact_source_id"] == "inline_payload_0"
    assert record["candidate_trace"]["artifact_source_ids"] == ["inline_payload_0"]
    assert record["live_authority_granted"] is False
    assert record["fixture_closure_allowed_by_acceptance"] is False


def test_acceptance_template_can_filter_by_surface_id():
    spec, queue = _queue_with_raw_mixer_candidate()
    selected = template_tool.build_acceptance_template(
        review_queue=queue,
        surface_id=spec["surface_id"],
        generated_at="2026-06-07T00:00:00Z",
    )
    missing = template_tool.build_acceptance_template(
        review_queue=queue,
        surface_id="dlive_state",
        generated_at="2026-06-07T00:00:00Z",
    )

    assert selected["status"] == "operator_acceptance_template_ready"
    assert selected["selected_surface_id"] == spec["surface_id"]
    assert selected["template_record_count"] == 1
    assert missing["status"] == "operator_candidate_missing"
    assert missing["selected_surface_id"] == "dlive_state"
    assert missing["template_records"] == []


def test_acceptance_template_is_ledger_compatible_after_operator_edits():
    _, queue = _queue_with_raw_mixer_candidate()
    template = template_tool.build_acceptance_template(
        review_queue=queue,
        generated_at="2026-06-07T00:00:00Z",
    )
    pending_record = template["template_records"][0]

    pending_ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[pending_record],
    )
    assert pending_ledger["status"] == "operator_acceptance_replacement_required"
    assert pending_ledger["summary"]["invalid_record_count"] == 1
    assert "acceptance_record_decision_unknown" in pending_ledger["records"][0][
        "violations"
    ]

    accepted_record = {
        **pending_record,
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-07T12:00:00Z",
        "review_notes": "manual artifact review passed",
    }
    accepted_ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[accepted_record],
    )
    assert accepted_ledger["status"] == "operator_acceptance_records_available"
    assert accepted_ledger["summary"]["accepted_record_count"] == 1
    assert accepted_ledger["summary"]["fixture_closure_allowed_by_acceptance"] is False
    assert accepted_ledger["summary"]["live_authority_granted"] is False
    assert accepted_ledger["summary"]["full_autonomous_ready"] is False


def test_acceptance_template_tool_main_writes_json_and_markdown(tmp_path: Path):
    json_output = tmp_path / "acceptance-template.json"
    markdown_output = tmp_path / "acceptance-template.md"

    assert template_tool.main(["--output", str(json_output), "--quiet"]) == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["type"] == "read_surface_physical_readback_fixture_acceptance_template"
    assert payload["live_authority_granted"] is False

    assert (
        template_tool.main(
            ["--format", "markdown", "--output", str(markdown_output), "--quiet"]
        )
        == 0
    )
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "Read Surface Physical Readback Fixture Acceptance Template" in markdown
    assert "live_authority_granted: `false`" in markdown


def test_acceptance_template_tool_is_transport_free():
    source = Path(
        "tools/build_read_surface_physical_readback_acceptance_template.py"
    ).read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "DLiveClient",
        "EnhancedOSCClient",
        "import socket",
        "websockets",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source
