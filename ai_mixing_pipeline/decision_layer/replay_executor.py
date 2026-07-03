"""Deterministic replay executor simulation for proposal-driven control decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .replay_proposal_ranking import ReplayDecisionProposal, ReplayGraphCheckpoint


REPLAY_EXECUTOR_SCHEMA_VERSION = "replay_executor/v1"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stable_dict(payload: dict[str, Any] | Any) -> dict[str, Any] | list[Any] | Any:
    if isinstance(payload, dict):
        return {str(k): _stable_dict(v) for k, v in sorted(payload.items(), key=lambda item: str(item[0]))}
    if isinstance(payload, list):
        return [_stable_dict(item) for item in payload]
    return payload


def _state_signature(state_payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_stable_dict(state_payload), sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _as_state_payload(state: "ReplayExecutionState") -> dict[str, Any]:
    return {
        "gains": {str(k): v for k, v in sorted(state.gains.items(), key=lambda item: str(item[0]))},
        "faders": {str(k): v for k, v in sorted(state.faders.items(), key=lambda item: str(item[0]))},
        "eq_bands": {str(k): [dict(item) for item in state.eq_bands[k]] for k in sorted(state.eq_bands)},
        "compressors": {str(k): dict(state.compressors[k]) for k in sorted(state.compressors)},
        "performer_interaction": {str(k): dict(v) for k, v in sorted(state.performer_interaction.items(), key=lambda item: str(item[0]))},
        "speech_priority": {str(k): dict(v) for k, v in sorted(state.speech_priority.items(), key=lambda item: str(item[0]))},
        "room_compensation": {str(k): dict(v) for k, v in sorted(state.room_compensation.items(), key=lambda item: str(item[0]))},
        "feedback_mitigation": [dict(item) for item in state.feedback_mitigation],
        "incident_log": [dict(item) for item in state.incident_log],
    }


def _restore_from_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "gains": {},
            "faders": {},
            "eq_bands": {},
            "compressors": {},
            "performer_interaction": {},
            "speech_priority": {},
            "room_compensation": {},
            "feedback_mitigation": [],
            "incident_log": [],
        }
    return {
        "gains": {str(k): float(v) for k, v in (payload.get("gains") or {}).items()},
        "faders": {str(k): float(v) for k, v in (payload.get("faders") or {}).items()},
        "eq_bands": {
            str(k): [
                dict(item) for item in (payload.get("eq_bands") or {}).get(k, []) if isinstance(item, dict)
            ]
            for k in sorted((payload.get("eq_bands") or {}).keys())
        },
        "compressors": {str(k): dict(v) for k, v in (payload.get("compressors") or {}).items()},
        "performer_interaction": {
            str(k): dict(v)
            for k, v in (
                (payload.get("performer_interaction"))
                or (payload.get("performer_stage_interaction"))
                or {}
            ).items()
        },
        "speech_priority": {str(k): dict(v) for k, v in (payload.get("speech_priority") or {}).items()},
        "room_compensation": {str(k): dict(v) for k, v in (payload.get("room_compensation") or {}).items()},
        "feedback_mitigation": [dict(item) for item in payload.get("feedback_mitigation") or [] if isinstance(item, dict)],
        "incident_log": [dict(item) for item in payload.get("incident_log") or [] if isinstance(item, dict)],
    }


@dataclass
class ReplayExecutionState:
    gains: dict[str, float] = field(default_factory=dict)
    faders: dict[str, float] = field(default_factory=dict)
    eq_bands: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    compressors: dict[str, dict[str, Any]] = field(default_factory=dict)
    performer_interaction: dict[str, dict[str, Any]] = field(default_factory=dict)
    speech_priority: dict[str, dict[str, Any]] = field(default_factory=dict)
    room_compensation: dict[str, dict[str, Any]] = field(default_factory=dict)
    feedback_mitigation: list[dict[str, Any]] = field(default_factory=list)
    incident_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ReplayExecutionState":
        restored = _restore_from_payload(payload)
        return cls(
            gains=restored["gains"],
            faders=restored["faders"],
            eq_bands=restored["eq_bands"],
            compressors=restored["compressors"],
            performer_interaction=restored["performer_interaction"],
            speech_priority=restored["speech_priority"],
            room_compensation=restored["room_compensation"],
            feedback_mitigation=restored["feedback_mitigation"],
            incident_log=restored["incident_log"],
        )

    def to_dict(self) -> dict[str, Any]:
        return _as_state_payload(self)

    def clone(self) -> "ReplayExecutionState":
        return ReplayExecutionState.from_dict(self.to_dict())


@dataclass
class ReplayExecutionEvent:
    seq: int
    proposal_id: str
    family: str
    replay_correlation_id: str
    action: str
    status: str
    dry_run: bool
    applied: bool
    pre_signature: str
    post_signature: str
    blocked_reason: str = ""
    pre_state: dict[str, Any] = field(default_factory=dict)
    post_state: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": int(self.seq),
            "proposal_id": str(self.proposal_id),
            "family": str(self.family),
            "replay_correlation_id": str(self.replay_correlation_id),
            "action": str(self.action),
            "status": str(self.status),
            "dry_run": bool(self.dry_run),
            "applied": bool(self.applied),
            "pre_signature": str(self.pre_signature),
            "post_signature": str(self.post_signature),
            "blocked_reason": str(self.blocked_reason),
            "pre_state": _stable_dict(self.pre_state),
            "post_state": _stable_dict(self.post_state),
            "details": _stable_dict(self.details),
        }


@dataclass
class ReplayExecutorResult:
    dry_run_only: bool
    proposals_executed: list[str]
    selected_proposal_id: str | None
    events: list[ReplayExecutionEvent]
    rollback_state: dict[str, Any]
    restored_state: dict[str, Any] | None
    state_snapshots: list[dict[str, Any]]
    trace_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_EXECUTOR_SCHEMA_VERSION,
            "dry_run_only": bool(self.dry_run_only),
            "proposals_executed": list(self.proposals_executed),
            "selected_proposal_id": self.selected_proposal_id,
            "events": [event.to_dict() for event in self.events],
            "rollback_state": _stable_dict(self.rollback_state),
            "restored_state": None if self.restored_state is None else _stable_dict(self.restored_state),
            "state_snapshots": [_stable_dict(item) for item in self.state_snapshots],
            "trace_signature": str(self.trace_signature),
        }


def _coerce_proposals(
    proposals: list[ReplayDecisionProposal] | ReplayGraphCheckpoint | dict[str, Any],
) -> list[ReplayDecisionProposal]:
    if isinstance(proposals, ReplayGraphCheckpoint):
        return list(proposals.proposals)
    if isinstance(proposals, dict):
        return [
            ReplayDecisionProposal.from_dict(dict(item))
            for item in (dict(proposals).get("proposals") or ())
        ]
    if isinstance(proposals, list):
        return [proposal if isinstance(proposal, ReplayDecisionProposal) else ReplayDecisionProposal.from_dict(dict(proposal)) for proposal in proposals]
    raise TypeError("proposals must be list, ReplayGraphCheckpoint, or manifest dict payload")


def _coerce_channel(value: Any) -> str:
    return str(_coerce_int(value))


def _emit_block(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
    blocked_reason: str,
    details: dict[str, Any],
) -> ReplayExecutionEvent:
    pre_state = state.to_dict()
    post_signature = _state_signature(pre_state)
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action=proposal.action_type,
        status="blocked",
        dry_run=bool(proposal.dry_run_only),
        applied=False,
        pre_signature=pre_signature,
        post_signature=post_signature,
        blocked_reason=blocked_reason,
        pre_state=pre_state,
        post_state=pre_state,
        details=details,
    )


def _simulate_gain(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    target = dict(proposal.target)
    channel = _coerce_channel(target.get("channel", 0))
    requested = _coerce_float(proposal.requested_state.get("gain_db"), state.gains.get(channel, 0.0))
    safety_step = _coerce_float(proposal.safety.get("max_step_db"), _coerce_float(proposal.safety.get("max_delta_db"), 0.0))
    current = _coerce_float(state.gains.get(channel), _coerce_float(proposal.current_state.get("gain_db")))
    delta = requested - current
    if safety_step and abs(delta) > safety_step:
        return _emit_block(state, proposal, pre_signature, f"gain_step_exceeds_limit:{abs(delta):.3f}>{safety_step:.3f}", {
            "action": "gain",
            "channel": channel,
            "requested": requested,
            "safety_step": safety_step,
        })
    state.gains[channel] = round(requested, 3)
    pre_state = state.to_dict()
    state.incident_log.append(
        {
            "family": proposal.family,
            "channel": channel,
            "action": "gain",
            "requested_gain_db": requested,
            "previous_gain_db": current,
        }
    )
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action="set_gain",
        status="applied",
        dry_run=bool(proposal.dry_run_only),
        applied=True,
        pre_signature=pre_signature,
        post_signature=_state_signature(post_state),
        pre_state=pre_state,
        post_state=post_state,
        details={"channel": channel, "requested": requested, "previous": current},
    )


def _simulate_fader(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    target = dict(proposal.target)
    channel = _coerce_channel(target.get("channel", 0))
    requested = _coerce_float(proposal.requested_state.get("fader_db"), state.faders.get(channel, 0.0))
    safety_step = _coerce_float(proposal.safety.get("max_step_db"), _coerce_float(proposal.safety.get("max_delta_db"), 0.0))
    current = _coerce_float(state.faders.get(channel), _coerce_float(proposal.current_state.get("fader_db")))
    delta = requested - current
    if safety_step and abs(delta) > safety_step:
        return _emit_block(state, proposal, pre_signature, f"fader_step_exceeds_limit:{abs(delta):.3f}>{safety_step:.3f}", {
            "action": "fader",
            "channel": channel,
            "requested": requested,
            "safety_step": safety_step,
        })
    state.faders[channel] = round(requested, 3)
    pre_state = state.to_dict()
    state.incident_log.append(
        {
            "family": proposal.family,
            "channel": channel,
            "action": "fader",
            "requested_fader_db": requested,
            "previous_fader_db": current,
        }
    )
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action="set_fader",
        status="applied",
        dry_run=bool(proposal.dry_run_only),
        applied=True,
        pre_signature=pre_signature,
        post_signature=_state_signature(post_state),
        pre_state=pre_state,
        post_state=post_state,
        details={"channel": channel, "requested": requested, "previous": current},
    )


def _simulate_eq(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    target = dict(proposal.target)
    channel = _coerce_channel(target.get("channel", 0))
    band_id = str(target.get("band_id", f"band_{len(state.eq_bands.get(channel, []))+1}"))
    change = {
        "band_id": band_id,
        "freq_hz": _coerce_float(proposal.requested_state.get("freq_hz", target.get("freq_hz")), 1000.0),
        "gain_db": _coerce_float(proposal.requested_state.get("gain_db"), 0.0),
        "q": _coerce_float(proposal.requested_state.get("q", target.get("q")), 1.0),
        "filter_type": str(proposal.requested_state.get("filter_type", target.get("filter_type", "peaking"))),
    }
    max_gain = _coerce_float(proposal.safety.get("max_gain_db", 6.0))
    if abs(change["gain_db"]) > max_gain:
        return _emit_block(state, proposal, pre_signature, f"eq_gain_exceeds_limit:{abs(change['gain_db']):.3f}>{max_gain:.3f}", {
            "action": "eq",
            "channel": channel,
            "requested": change["gain_db"],
            "max_gain": max_gain,
        })
    state.eq_bands.setdefault(channel, []).append(change)
    pre_state = state.to_dict()
    state.incident_log.append({"family": proposal.family, "channel": channel, "action": "eq", **change})
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action="set_eq_band",
        status="applied",
        dry_run=bool(proposal.dry_run_only),
        applied=True,
        pre_signature=pre_signature,
        post_signature=_state_signature(post_state),
        pre_state=pre_state,
        post_state=post_state,
        details=change,
    )


def _simulate_compression(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    target = dict(proposal.target)
    channel = _coerce_channel(target.get("channel", target.get("target", "mix")))
    ratio = _coerce_float(proposal.requested_state.get("ratio"), 1.0)
    max_ratio = _coerce_float(proposal.safety.get("max_ratio", 3.0))
    if ratio > max_ratio:
        return _emit_block(state, proposal, pre_signature, f"compression_ratio_exceeds_limit:{ratio:.3f}>{max_ratio:.3f}", {
            "action": "compressor",
            "channel": channel,
            "ratio": ratio,
        })
    payload = {
        "threshold_db": _coerce_float(proposal.requested_state.get("threshold_db")),
        "ratio": ratio,
        "attack_ms": _coerce_float(proposal.requested_state.get("attack_ms"), 15.0),
        "release_ms": _coerce_float(proposal.requested_state.get("release_ms"), 140.0),
        "makeup_gain_db": _coerce_float(proposal.requested_state.get("makeup_gain_db")),
        "action_type": proposal.action_type,
    }
    state.compressors[channel] = payload
    pre_state = state.to_dict()
    state.incident_log.append({"family": proposal.family, "channel": channel, "action": "compressor", **payload})
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action="set_compressor",
        status="applied",
        dry_run=bool(proposal.dry_run_only),
        applied=True,
        pre_signature=pre_signature,
        post_signature=_state_signature(post_state),
        pre_state=pre_state,
        post_state=post_state,
        details=payload,
    )


def _simulate_room_compensation(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    target = dict(proposal.target)
    room_target = str(target.get("target", "master"))
    requested = dict(proposal.requested_state)
    state.room_compensation[room_target] = requested
    pre_state = state.to_dict()
    state.incident_log.append(
        {
            "family": proposal.family,
            "room_target": room_target,
            "action": "room_compensation",
            "filters": requested.get("filters", []),
        }
    )
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action="plan_room_eq",
        status="applied",
        dry_run=bool(proposal.dry_run_only),
        applied=True,
        pre_signature=pre_signature,
        post_signature=_state_signature(post_state),
        pre_state=pre_state,
        post_state=post_state,
        details={"target": room_target, "filters": requested.get("filters", [])},
    )


def _simulate_speech_priority(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    target = dict(proposal.target)
    leader = _coerce_channel(target.get("leader_channel", "unknown"))
    follower = _coerce_channel(target.get("suppressed_channel", "unknown"))
    requested = dict(proposal.requested_state)
    boost_limit = _coerce_float(proposal.safety.get("max_boost_db"), 2.0)
    cut_limit = _coerce_float(proposal.safety.get("max_cut_db"), 3.0)
    if abs(_coerce_float(requested.get("priority_boost_db"))) > boost_limit or abs(_coerce_float(requested.get("priority_cut_db"))) > cut_limit:
        return _emit_block(state, proposal, pre_signature, "speech_priority_limits_exceeded", {
            "leader_channel": leader,
            "follower_channel": follower,
            "requested": requested,
        })
    payload = {
        "leader_channel": leader,
        "suppressed_channel": follower,
        "priority_boost_db": _coerce_float(requested.get("priority_boost_db")),
        "priority_cut_db": _coerce_float(requested.get("priority_cut_db")),
    }
    state.speech_priority[f"{leader}->{follower}"] = payload
    pre_state = state.to_dict()
    state.incident_log.append({"family": proposal.family, "action": "speech_priority", **payload})
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action="adjust_speech_priority_balance",
        status="applied",
        dry_run=bool(proposal.dry_run_only),
        applied=True,
        pre_signature=pre_signature,
        post_signature=_state_signature(post_state),
        pre_state=pre_state,
        post_state=post_state,
        details=payload,
    )


def _simulate_feedback(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    target = dict(proposal.target)
    channel = _coerce_channel(target.get("channel", 0))
    payload = {
        "channel": channel,
        "frequency_hz": _coerce_float(target.get("frequency_hz")),
        "gain_db": _coerce_float(proposal.requested_state.get("gain_db")),
        "q": _coerce_float(proposal.requested_state.get("q", 4.0)),
        "action": str(proposal.action_type),
    }
    state.feedback_mitigation.append(payload)
    pre_state = state.to_dict()
    state.incident_log.append({"family": proposal.family, "action": "feedback_mitigation", **payload})
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action=str(proposal.action_type),
        status="applied",
        dry_run=bool(proposal.dry_run_only),
        applied=True,
        pre_signature=pre_signature,
        post_signature=_state_signature(post_state),
        pre_state=pre_state,
        post_state=post_state,
        details=payload,
    )


def _simulate_operator_assistance(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    message = str(proposal.requested_state.get("message", proposal.current_state.get("message", "")))
    state.incident_log.append({"family": proposal.family, "action": "operator_assistance", "message": message})
    pre_state = state.to_dict()
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action="operator_prompt",
        status="observed_only",
        dry_run=True,
        applied=False,
        pre_signature=pre_signature,
        post_signature=pre_signature,
        pre_state=pre_state,
        post_state=post_state,
        details={"message": message},
    )


def _simulate_performer_stage_interaction(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
    pre_signature: str,
) -> ReplayExecutionEvent:
    target = dict(proposal.target)
    performer_id = str(target.get("performer_id", "unknown"))
    channel = _coerce_channel(target.get("channel", 0))
    requested = dict(proposal.requested_state)
    level_delta = _coerce_float(requested.get("level_delta_db"))
    spill_delta = _coerce_float(requested.get("spill_delta_db"))
    max_level_delta = _coerce_float(proposal.safety.get("max_level_delta_db"), 0.0)
    max_spill_delta = _coerce_float(proposal.safety.get("max_spill_delta_db"), 0.0)

    if max_level_delta and abs(level_delta) > max_level_delta:
        return _emit_block(
            state,
            proposal,
            pre_signature,
            f"performer_level_delta_exceeds_limit:{abs(level_delta):.3f}>{max_level_delta:.3f}",
            {"performer_id": performer_id, "requested_level_delta_db": level_delta},
        )
    if max_spill_delta and abs(spill_delta) > max_spill_delta:
        return _emit_block(
            state,
            proposal,
            pre_signature,
            f"performer_spill_delta_exceeds_limit:{abs(spill_delta):.3f}>{max_spill_delta:.3f}",
            {"performer_id": performer_id, "requested_spill_delta_db": spill_delta},
        )

    payload = {
        "performer_id": performer_id,
        "channel": channel,
        "zone_from": target.get("zone_from"),
        "zone_to": target.get("zone_to"),
        "orientation_deg": requested.get("orientation_deg"),
        "distance_m": requested.get("distance_m"),
        "proximity_delta_m": requested.get("proximity_delta_m"),
        "level_delta_db": level_delta,
        "spill_delta_db": spill_delta,
        "risk": requested.get("risk", "moderate"),
    }
    state.performer_interaction[performer_id] = payload
    pre_state = state.to_dict()
    state.incident_log.append(
        {
            "family": proposal.family,
            "action": "simulate_stage_interaction",
            "performer_id": performer_id,
            "channel": channel,
            "level_delta_db": level_delta,
            "spill_delta_db": spill_delta,
        }
    )
    post_state = state.to_dict()
    return ReplayExecutionEvent(
        seq=0,
        proposal_id=proposal.proposal_id,
        family=proposal.family,
        replay_correlation_id=proposal.replay_correlation_id,
        action="simulate_stage_interaction",
        status="applied",
        dry_run=bool(proposal.dry_run_only),
        applied=True,
        pre_signature=pre_signature,
        post_signature=_state_signature(post_state),
        pre_state=pre_state,
        post_state=post_state,
        details=payload,
    )


def _simulate_single_proposal(
    state: ReplayExecutionState,
    proposal: ReplayDecisionProposal,
) -> ReplayExecutionEvent:
    pre_signature = _state_signature(state.to_dict())

    if proposal.auto_apply_blocked and not bool(proposal.dry_run_only):
        return _emit_block(state, proposal, pre_signature, "auto_apply_blocked_without_dry_run", {
            "proposal_id": proposal.proposal_id
        })
    if proposal.family == "gain_adjustment":
        return _simulate_gain(state, proposal, pre_signature)
    if proposal.family == "fader_adjustment":
        return _simulate_fader(state, proposal, pre_signature)
    if proposal.family == "eq_adjustment":
        return _simulate_eq(state, proposal, pre_signature)
    if proposal.family == "compression":
        return _simulate_compression(state, proposal, pre_signature)
    if proposal.family == "speech_arbitration":
        return _simulate_speech_priority(state, proposal, pre_signature)
    if proposal.family == "room_compensation":
        return _simulate_room_compensation(state, proposal, pre_signature)
    if proposal.family == "feedback_mitigation":
        return _simulate_feedback(state, proposal, pre_signature)
    if proposal.family == "operator_assistance":
        return _simulate_operator_assistance(state, proposal, _state_signature(state.to_dict()))
    if proposal.family == "performer_stage_interaction":
        return _simulate_performer_stage_interaction(state, proposal, pre_signature)
    return _emit_block(state, proposal, pre_signature, f"unsupported_family:{proposal.family}", {})


def simulate_replay_decisions(
    proposals: list[ReplayDecisionProposal] | ReplayGraphCheckpoint | dict[str, Any],
    *,
    selected_proposal_id: str | None = None,
    initial_state: dict[str, Any] | ReplayExecutionState | None = None,
    rollback_to: int | None = None,
) -> ReplayExecutorResult:
    resolved = _coerce_proposals(proposals)
    if selected_proposal_id:
        resolved = [proposal for proposal in resolved if proposal.proposal_id == selected_proposal_id]
    state = initial_state.clone() if isinstance(initial_state, ReplayExecutionState) else ReplayExecutionState.from_dict(
        initial_state if isinstance(initial_state, dict) else None
    )
    executed: list[str] = []
    events: list[ReplayExecutionEvent] = []
    snapshots: list[dict[str, Any]] = []
    for index, proposal in enumerate(resolved, start=1):
        if proposal.auto_apply_blocked:
            # Keep dry-run safe by default; still capture deterministic trace.
            pass
        event = _simulate_single_proposal(state, proposal)
        event.seq = index
        events.append(event)
        snapshots.append(state.to_dict())
        if event.applied:
            executed.append(proposal.proposal_id)
        # deterministic, deterministic order by proposal list order.

    rollback_state = state.to_dict()
    restored_state = None
    if rollback_to is not None:
        if rollback_to < 0 or rollback_to > len(snapshots):
            raise ValueError("rollback_to must be within snapshot range")
        if rollback_to == 0:
            restored_state = _restore_from_payload({})
        else:
            restored_state = snapshots[rollback_to - 1]
        rollback_state = dict(restored_state)

    trace_signature = hashlib.sha256(
        json.dumps(
            {
                "dry_run_only": True,
                "schema_version": REPLAY_EXECUTOR_SCHEMA_VERSION,
                "executed": [event.to_dict() for event in events],
                "state": rollback_state,
            },
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    return ReplayExecutorResult(
        dry_run_only=True,
        proposals_executed=executed,
        selected_proposal_id=selected_proposal_id,
        events=events,
        rollback_state=rollback_state,
        restored_state=restored_state,
        state_snapshots=snapshots,
        trace_signature=trace_signature,
    )


def build_replay_executor_from_rewind_context(context: dict[str, Any] | None) -> ReplayExecutionState:
    if not isinstance(context, dict):
        return ReplayExecutionState()
    state = ReplayExecutionState()
    try:
        trim_snapshot = dict(context.get("trim_snapshot") or {})
        for item in (trim_snapshot.get("last_state") or {}).get("applied", []) if isinstance(trim_snapshot.get("last_state"), dict) else ():
            channel = _coerce_channel(item.get("channel"))
            if item.get("current_trim_db") is not None:
                state.gains[channel] = _coerce_float(item.get("current_trim_db"))
            if item.get("target_trim_db") is not None:
                state.gains[channel] = _coerce_float(item.get("target_trim_db"), state.gains.get(channel, 0.0))
    except Exception:
        pass
    return state
