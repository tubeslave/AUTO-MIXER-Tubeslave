"""Fail-closed lifecycle registry for supervised soundcheck write families.

The registry describes promotion state and safety contracts only. Presence in
this module never grants live authority; normal soundcheck execution requires
an explicit ``live_allowed`` lifecycle produced from an accepted owner/readiness
promotion decision. Hardware proof alone remains ``hardware_proven``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


LIFECYCLE_PROPOSAL_ONLY = "proposal_only"
LIFECYCLE_PROOF_CANDIDATE = "proof_candidate"
LIFECYCLE_HARDWARE_PROVEN = "hardware_proven"
LIFECYCLE_LIVE_ALLOWED = "live_allowed"

VALID_LIFECYCLES = (
    LIFECYCLE_PROPOSAL_ONLY,
    LIFECYCLE_PROOF_CANDIDATE,
    LIFECYCLE_HARDWARE_PROVEN,
    LIFECYCLE_LIVE_ALLOWED,
)

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_DESTRUCTIVE = "destructive"
VALID_RISK_CLASSES = (RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_DESTRUCTIVE)

AUDIO_RECHECK_REQUIRED = "required"
AUDIO_RECHECK_WHEN_OBSERVABLE = "when_observable"
AUDIO_RECHECK_NOT_APPLICABLE = "not_applicable"
VALID_AUDIO_RECHECK_POLICIES = (
    AUDIO_RECHECK_REQUIRED,
    AUDIO_RECHECK_WHEN_OBSERVABLE,
    AUDIO_RECHECK_NOT_APPLICABLE,
)


@dataclass(frozen=True)
class SupervisedConsoleFamilyContract:
    """Promotion and safety metadata for one console-write family."""

    family_id: str
    label: str
    write_kinds: Tuple[str, ...]
    risk_class: str
    lifecycle: str
    separate_approval: bool
    audio_recheck_policy: str
    rollback_required: bool = True
    physical_readback_required: bool = True
    hardware_proof: str = "missing"
    live_allowed_write_kinds: Tuple[str, ...] = ()
    blockers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("family_id is required")
        if not self.write_kinds:
            raise ValueError(f"{self.family_id} must declare write_kinds")
        if self.lifecycle not in VALID_LIFECYCLES:
            raise ValueError(f"Unknown lifecycle for {self.family_id}: {self.lifecycle}")
        if self.risk_class not in VALID_RISK_CLASSES:
            raise ValueError(f"Unknown risk class for {self.family_id}: {self.risk_class}")
        if self.audio_recheck_policy not in VALID_AUDIO_RECHECK_POLICIES:
            raise ValueError(
                f"Unknown audio recheck policy for {self.family_id}: "
                f"{self.audio_recheck_policy}"
            )
        unknown_live_kinds = set(self.live_allowed_write_kinds) - set(self.write_kinds)
        if unknown_live_kinds:
            raise ValueError(
                f"{self.family_id} has unknown live write kinds: {sorted(unknown_live_kinds)}"
            )
        if self.live_allowed_write_kinds and self.hardware_proof != "accepted":
            raise ValueError(
                f"{self.family_id} cannot allow live write kinds without accepted hardware proof"
            )
        if self.lifecycle == LIFECYCLE_LIVE_ALLOWED and not self.live_allowed_write_kinds:
            raise ValueError(f"{self.family_id} is live_allowed but has no live write kinds")

    def permits_execution(self, *, write_kind: str, proof_run: bool) -> bool:
        """Allow only the exact hardware-proven kind during normal operation."""
        if write_kind not in self.write_kinds:
            return False
        if write_kind in self.live_allowed_write_kinds:
            return True
        return proof_run and self.lifecycle in {
            LIFECYCLE_PROOF_CANDIDATE,
            LIFECYCLE_HARDWARE_PROVEN,
            LIFECYCLE_LIVE_ALLOWED,
        }

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["write_kinds"] = list(self.write_kinds)
        payload["live_allowed_write_kinds"] = list(self.live_allowed_write_kinds)
        payload["blockers"] = list(self.blockers)
        payload["normal_supervised_apply_allowed"] = bool(
            self.live_allowed_write_kinds
        )
        payload["fully_live_allowed"] = set(self.live_allowed_write_kinds) == set(
            self.write_kinds
        )
        payload["proof_run_allowed"] = self.lifecycle in {
            LIFECYCLE_PROOF_CANDIDATE,
            LIFECYCLE_HARDWARE_PROVEN,
            LIFECYCLE_LIVE_ALLOWED,
        }
        return payload


FAMILY_CONTRACTS: Tuple[SupervisedConsoleFamilyContract, ...] = (
    SupervisedConsoleFamilyContract(
        family_id="fader",
        label="Channel fader",
        write_kinds=("fader",),
        risk_class=RISK_LOW,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=False,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=("owner_readiness_promotion_required",),
    ),
    SupervisedConsoleFamilyContract(
        family_id="gain_trim",
        label="Input gain and trim",
        write_kinds=("gain", "trim"),
        risk_class=RISK_MEDIUM,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=False,
        audio_recheck_policy=AUDIO_RECHECK_WHEN_OBSERVABLE,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=(
            "owner_readiness_promotion_required",
            "gain_up_requires_true_peak_and_source_identity",
        ),
    ),
    SupervisedConsoleFamilyContract(
        family_id="mute",
        label="Channel mute",
        write_kinds=("mute",),
        risk_class=RISK_HIGH,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=True,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=(
            "owner_readiness_promotion_required",
            "separate_operator_approval_required",
        ),
    ),
    SupervisedConsoleFamilyContract(
        family_id="pan_phase_delay",
        label="Pan, polarity and delay",
        write_kinds=("pan", "polarity", "delay"),
        risk_class=RISK_MEDIUM,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=False,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=("owner_readiness_promotion_required",),
    ),
    SupervisedConsoleFamilyContract(
        family_id="eq_hpf",
        label="HPF and parametric EQ",
        write_kinds=("hpf", "eq"),
        risk_class=RISK_MEDIUM,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=False,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=(
            "owner_readiness_promotion_required",
            "eq_live_apply_is_cut_only",
        ),
    ),
    SupervisedConsoleFamilyContract(
        family_id="gate_expander",
        label="Gate and expander",
        write_kinds=("gate", "expander"),
        risk_class=RISK_MEDIUM,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=False,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=(
            "owner_readiness_promotion_required",
            "expander_write_kind_not_hardware_proven",
        ),
    ),
    SupervisedConsoleFamilyContract(
        family_id="dynamics_compressor",
        label="Compressor, dynamics and de-esser",
        write_kinds=("compressor", "dynamics", "deesser"),
        risk_class=RISK_MEDIUM,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=False,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=(
            "owner_readiness_promotion_required",
            "dynamics_and_deesser_write_kinds_not_hardware_proven",
        ),
    ),
    SupervisedConsoleFamilyContract(
        family_id="fx_sends_inserts",
        label="FX, inserts, sends and returns",
        write_kinds=("fx", "insert", "send", "return"),
        risk_class=RISK_HIGH,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=True,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=(
            "owner_readiness_promotion_required",
            "fx_insert_return_write_kinds_not_hardware_proven",
            "monitor_safety_required",
        ),
    ),
    SupervisedConsoleFamilyContract(
        family_id="bus_group_dca_matrix",
        label="Bus, group, DCA and matrix",
        write_kinds=("bus", "group", "dca", "matrix"),
        risk_class=RISK_HIGH,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=True,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=(
            "owner_readiness_promotion_required",
            "group_and_matrix_write_kinds_not_hardware_proven",
        ),
    ),
    SupervisedConsoleFamilyContract(
        family_id="master_output",
        label="Master and output processing",
        write_kinds=("master", "output"),
        risk_class=RISK_HIGH,
        lifecycle=LIFECYCLE_HARDWARE_PROVEN,
        separate_approval=True,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="proof_passed_owner_promotion_pending",
        blockers=(
            "owner_readiness_promotion_required",
            "output_processing_write_kind_not_hardware_proven",
            "true_peak_headroom_required",
        ),
    ),
    SupervisedConsoleFamilyContract(
        family_id="routing",
        label="Routing",
        write_kinds=("routing",),
        risk_class=RISK_DESTRUCTIVE,
        lifecycle=LIFECYCLE_PROPOSAL_ONLY,
        separate_approval=True,
        audio_recheck_policy=AUDIO_RECHECK_WHEN_OBSERVABLE,
        blockers=("routing_diff_hardware_proof_missing",),
    ),
    SupervisedConsoleFamilyContract(
        family_id="scene_preset",
        label="Scene, snapshot and preset",
        write_kinds=("scene", "snapshot", "preset"),
        risk_class=RISK_DESTRUCTIVE,
        lifecycle=LIFECYCLE_PROPOSAL_ONLY,
        separate_approval=True,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        blockers=("scoped_scene_diff_hardware_proof_missing",),
    ),
)

_CONTRACTS_BY_ID = {contract.family_id: contract for contract in FAMILY_CONTRACTS}
_FAMILY_BY_WRITE_KIND = {
    write_kind: contract.family_id
    for contract in FAMILY_CONTRACTS
    for write_kind in contract.write_kinds
}


def get_supervised_console_family(
    family_id: str,
) -> SupervisedConsoleFamilyContract | None:
    return _CONTRACTS_BY_ID.get(str(family_id or "").strip().lower())


def family_id_for_write_kind(write_kind: str) -> str | None:
    return _FAMILY_BY_WRITE_KIND.get(str(write_kind or "").strip().lower())


def supervised_console_family_registry() -> Dict[str, Dict[str, object]]:
    return {contract.family_id: contract.to_dict() for contract in FAMILY_CONTRACTS}


def validate_supervised_console_family_registry() -> Tuple[str, ...]:
    failures = []
    family_ids = [contract.family_id for contract in FAMILY_CONTRACTS]
    if len(family_ids) != len(set(family_ids)):
        failures.append("duplicate_family_id")
    all_kinds = [kind for contract in FAMILY_CONTRACTS for kind in contract.write_kinds]
    if len(all_kinds) != len(set(all_kinds)):
        failures.append("duplicate_write_kind")
    for contract in FAMILY_CONTRACTS:
        if contract.live_allowed_write_kinds and contract.hardware_proof != "accepted":
            failures.append(f"live_write_kind_without_accepted_proof:{contract.family_id}")
    return tuple(failures)
