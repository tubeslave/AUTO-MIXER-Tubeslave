"""Read-only safety proof matrix for live-capable Automixer surfaces.

The matrix is intentionally declarative. It records what evidence exists for
each action family before anything can be promoted toward real WING writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple


COVERAGE_STATUSES = {
    "hardware_proven_narrow",
    "dry_run_ready",
    "proposal_only",
    "quarantined",
    "blocked",
}

PROOF_STATES = {"passed", "partial", "missing", "not_allowed"}


@dataclass(frozen=True)
class SafetyCoverageRow:
    id: str
    label: str
    action_type: str
    status: str
    live_allowed: bool
    canonical_gate: str
    unit_tests: str
    offline_harness: str
    fake_mixer_proof: str
    real_wing_proof: str
    pre_read: str
    apply_readback: str
    rollback_readback: str
    emergency_stop_proof: str
    current_blocker: str
    next_required_evidence: str
    required_evidence_components: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in COVERAGE_STATUSES:
            raise ValueError(f"Unknown coverage status: {self.status}")
        for field_name in (
            "unit_tests",
            "offline_harness",
            "fake_mixer_proof",
            "real_wing_proof",
            "pre_read",
            "apply_readback",
            "rollback_readback",
            "emergency_stop_proof",
        ):
            if getattr(self, field_name) not in PROOF_STATES:
                raise ValueError(f"{self.id}.{field_name} has unknown proof state")
        if self.live_allowed and self.status != "hardware_proven_narrow":
            raise ValueError(f"{self.id} allows live writes without narrow hardware proof")
        if self.live_allowed and "apply_operator_proposal" not in self.canonical_gate:
            raise ValueError(f"{self.id} must use apply_operator_proposal")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


SAFETY_COVERAGE_ROWS: Tuple[SafetyCoverageRow, ...] = (
    SafetyCoverageRow(
        id="fader_supervised",
        label="Supervised channel fader",
        action_type="fader",
        status="hardware_proven_narrow",
        live_allowed=True,
        canonical_gate="apply_operator_proposal -> supervised manual gate",
        unit_tests="passed",
        offline_harness="passed",
        fake_mixer_proof="passed",
        real_wing_proof="partial",
        pre_read="passed",
        apply_readback="passed",
        rollback_readback="passed",
        emergency_stop_proof="partial",
        current_blocker="Limited proof only: one narrow fader path, not broad concert readiness.",
        next_required_evidence="Repeat controlled WING proof across representative channels with connection-loss and emergency-stop reset evidence.",
        notes=("live scope remains supervised only", "no autonomous fader movement"),
    ),
    SafetyCoverageRow(
        id="gain_supervised",
        label="Supervised input gain/trim",
        action_type="gain",
        status="dry_run_ready",
        live_allowed=False,
        canonical_gate="apply_operator_proposal -> supervised manual gate",
        unit_tests="passed",
        offline_harness="passed",
        fake_mixer_proof="partial",
        real_wing_proof="missing",
        pre_read="partial",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker="Physical WING gain proof is not complete.",
        next_required_evidence="One small supervised gain-down proof with physical pre-read, apply readback, rollback readback, and blocked unsafe follow-up.",
        required_evidence_components=(
            "physical_pre_read",
            "bounded_gain_down_apply",
            "apply_readback",
            "rollback_readback",
            "emergency_stop_block",
            "unsafe_gain_up_followup_blocked",
            "gain_up_operator_authored_opt_in",
        ),
        notes=("gain-up remains blocked unless operator-authored",),
    ),
    SafetyCoverageRow(
        id="eq_autoeq",
        label="EQ / AutoEQ / HPF",
        action_type="eq",
        status="proposal_only",
        live_allowed=False,
        canonical_gate="proposal evidence only",
        unit_tests="passed",
        offline_harness="partial",
        fake_mixer_proof="missing",
        real_wing_proof="missing",
        pre_read="missing",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker="Filter audit, feedback safety, physical readback, and rollback proof are missing.",
        next_required_evidence="Keep EQ in offline/proposal mode until filter package audit and fake-mixer proof are complete.",
    ),
    SafetyCoverageRow(
        id="dynamics_compressor",
        label="Compressor / dynamics",
        action_type="compressor",
        status="proposal_only",
        live_allowed=False,
        canonical_gate="proposal evidence only",
        unit_tests="partial",
        offline_harness="partial",
        fake_mixer_proof="missing",
        real_wing_proof="missing",
        pre_read="missing",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker="Direct compressor write paths are quarantined from the live executor.",
        next_required_evidence="Split recommendation code from executor code and add fail-closed tests before any promotion.",
    ),
    SafetyCoverageRow(
        id="gate_expander",
        label="Gate / expander / bleed suppression",
        action_type="gate",
        status="proposal_only",
        live_allowed=False,
        canonical_gate="context-aware recommendation only",
        unit_tests="passed",
        offline_harness="partial",
        fake_mixer_proof="missing",
        real_wing_proof="missing",
        pre_read="missing",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker="ContextAwareGateAgent is dry-run and must be connected to real telemetry before operator trust.",
        next_required_evidence="Feed real telemetry/replay sessions into bleed metrics; keep live gate writes blocked.",
        notes=("threshold-only gate is not the target",),
    ),
    SafetyCoverageRow(
        id="pan_phase_delay",
        label="Pan / phase / delay",
        action_type="pan_phase_delay",
        status="quarantined",
        live_allowed=False,
        canonical_gate="blocked from canonical live runtime",
        unit_tests="partial",
        offline_harness="partial",
        fake_mixer_proof="missing",
        real_wing_proof="missing",
        pre_read="missing",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker="Stage-position and mono-compatibility evidence are not wired into a supervised proof path.",
        next_required_evidence="Define read-only stage-position DTO and proposal schema before fake-mixer proof.",
    ),
    SafetyCoverageRow(
        id="routing_snapshots_scenes",
        label="Routing / snapshots / scenes",
        action_type="routing_snapshot_scene",
        status="blocked",
        live_allowed=False,
        canonical_gate="blocked from automixer live apply",
        unit_tests="partial",
        offline_harness="missing",
        fake_mixer_proof="missing",
        real_wing_proof="not_allowed",
        pre_read="missing",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker="These actions can rewrite broad mixer state and are not MVP automation surfaces.",
        next_required_evidence="Keep manual-only runbooks; no automated promotion without separate ADR.",
    ),
    SafetyCoverageRow(
        id="fx_sends_inserts",
        label="FX / sends / inserts",
        action_type="fx_send_insert",
        status="quarantined",
        live_allowed=False,
        canonical_gate="blocked from canonical live runtime",
        unit_tests="partial",
        offline_harness="partial",
        fake_mixer_proof="missing",
        real_wing_proof="missing",
        pre_read="missing",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker="FX/color logic is useful as an explainable plan, not a live write path.",
        next_required_evidence="Add offline FX plan evaluation before any operator-facing proposal.",
    ),
    SafetyCoverageRow(
        id="bus_group_matrix",
        label="Bus / group / DCA / matrix",
        action_type="bus_group_matrix",
        status="quarantined",
        live_allowed=False,
        canonical_gate="blocked from canonical live runtime",
        unit_tests="partial",
        offline_harness="partial",
        fake_mixer_proof="missing",
        real_wing_proof="missing",
        pre_read="missing",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker=(
            "Bus, group, DCA, matrix, and master-bus writes can reshape the "
            "whole mix and have no supervised physical readback or rollback proof."
        ),
        next_required_evidence=(
            "Add bus/group/matrix safety review with true-peak headroom, "
            "assignment/fader readback, rollback, fake-mixer, real-WING, and "
            "emergency-stop proof before any supervised candidate."
        ),
        notes=("broad_balance_surface", "no_bus_dca_matrix_live_writes"),
    ),
    SafetyCoverageRow(
        id="mute_supervised",
        label="Mute / unmute",
        action_type="mute",
        status="quarantined",
        live_allowed=False,
        canonical_gate="blocked from canonical live runtime",
        unit_tests="partial",
        offline_harness="partial",
        fake_mixer_proof="missing",
        real_wing_proof="missing",
        pre_read="missing",
        apply_readback="missing",
        rollback_readback="missing",
        emergency_stop_proof="missing",
        current_blocker=(
            "Mute and unmute automation can drop sources or reopen unsafe "
            "channels without fresh intent, ownership, readback, and rollback proof."
        ),
        next_required_evidence=(
            "Add fresh mute-state intent, owner, signal-return, physical readback, "
            "rollback, fake-mixer, real-WING, and emergency-stop proof before "
            "any supervised mute candidate."
        ),
        notes=("manual_mute_authority_preserved", "no_mute_live_writes"),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_safety_coverage_rows() -> Iterable[SafetyCoverageRow]:
    return SAFETY_COVERAGE_ROWS


def get_safety_coverage_row(row_id: str) -> SafetyCoverageRow | None:
    return next((row for row in SAFETY_COVERAGE_ROWS if row.id == row_id), None)


def build_safety_coverage_matrix_snapshot() -> Dict[str, Any]:
    status_counts: Dict[str, int] = {status: 0 for status in sorted(COVERAGE_STATUSES)}
    live_allowed_ids = []
    rows = [row.to_dict() for row in SAFETY_COVERAGE_ROWS]
    for row in SAFETY_COVERAGE_ROWS:
        status_counts[row.status] += 1
        if row.live_allowed:
            live_allowed_ids.append(row.id)
    missing_real_wing = [
        row.id
        for row in SAFETY_COVERAGE_ROWS
        if row.real_wing_proof in {"missing", "partial"} and row.status != "blocked"
    ]
    return {
        "type": "safety_coverage_matrix",
        "generated_at": _utc_now(),
        "status": "ok",
        "summary": {
            "row_count": len(SAFETY_COVERAGE_ROWS),
            "status_counts": status_counts,
            "live_allowed_row_ids": live_allowed_ids,
            "live_apply_policy": "supervised_fader_gain_only",
            "missing_or_partial_real_wing_proof_ids": missing_real_wing,
            "next_global_step": "Calibrate thresholds and complete gain proof before widening live claims.",
        },
        "rows": rows,
    }
