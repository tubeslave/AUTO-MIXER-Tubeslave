"""Deterministic replay executor simulation for proposal-driven control decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .replay_proposal_ranking import ReplayDecisionProposal, ReplayGraphCheckpoint
from .replay_restore_loader import ReplayRewindContext


REPLAY_EXECUTOR_SCHEMA_VERSION = "replay_executor/v1"


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


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


def _stable_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(_stable_json(payload))


def _state_signature(payload: dict[str, Any]) -> str:
    return _stable_hash(payload)


def _as_state_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    state = {
        "channels": {},
        "room_compensation": {},
        "feedback_mitigation": [],
        "speech_priority": {},
        "operator_notes": [],
        "performer_interactions": [],
    }
    if payload:
        for key, value in payload.items():
            if isinstance(value, dict):
                state[key] = deepcopy(value)
            elif isinstance(value, list):
                state[key] = deepcopy(value)
            else:
                state[key] = value
    return state


def _restore_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _as_state_payload(payload)


@dataclass(frozen=True)
class ReplayExecutionState:
    state: dict[str, Any]
    state_signature: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReplayExecutionState":
        stable = _stable_dict(payload)
        return cls(state=stable, state_signature=_state_signature(stable))

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "state_signature": self.state_signature}


@dataclass(frozen=True)
class ReplayExecutionEvent:
    seq: int
    proposal_id: str
    replay_correlation_id: str | None
    family: str
    event_type: str
    status: str
    channel: int | None
    previous_state: dict[str, Any]
    next_state: dict[str, Any]
    rollback_target: dict[str, Any]
    state_signature: str
    blocked_reason: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayExecutorResult:
    schema_version: str
    selected_proposal_id: str
    initial_state: dict[str, Any]
    final_state: dict[str, Any]
    final_state_signature: str
    events: list[ReplayExecutionEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selected_proposal_id": self.selected_proposal_id,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "final_state_signature": self.final_state_signature,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayExecutorResult":
        return cls(
            schema_version=str(payload.get("schema_version", REPLAY_EXECUTOR_SCHEMA_VERSION)),
            selected_proposal_id=str(payload["selected_proposal_id"]),
            initial_state=dict(payload.get("initial_state") or {}),
            final_state=dict(payload.get("final_state") or {}),
            final_state_signature=str(payload.get("final_state_signature") or ""),
            events=[ReplayExecutionEvent(**event) for event in payload.get("events", [])],
        )


def _coerce_proposals(payload: list[ReplayDecisionProposal | dict[str, Any]]) -> list[ReplayDecisionProposal]:
    return [item if isinstance(item, ReplayDecisionProposal) else ReplayDecisionProposal.from_dict(item) for item in payload]


def _coerce_channel(proposal: ReplayDecisionProposal) -> int | None:
    for key in ("mixer_channel", "channel"):
        if key in proposal.target:
            return _coerce_int(proposal.target[key], None)  # type: ignore[arg-type]
    return None


def _emit_block(
    *,
    seq: int,
    proposal: ReplayDecisionProposal,
    state: dict[str, Any],
    reason: str,
) -> ReplayExecutionEvent:
    stable_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="blocked",
        status="blocked",
        channel=_coerce_channel(proposal),
        previous_state=stable_state,
        next_state=stable_state,
        rollback_target=stable_state,
        state_signature=_state_signature(stable_state),
        blocked_reason=reason,
        payload=proposal.to_dict(),
    )


def _channel_state(state: dict[str, Any], proposal: ReplayDecisionProposal) -> dict[str, Any]:
    channel = _coerce_channel(proposal)
    if channel is None:
        return {}
    return state.setdefault("channels", {}).setdefault(str(channel), {})


def _simulate_gain(seq: int, proposal: ReplayDecisionProposal, state: dict[str, Any]) -> ReplayExecutionEvent:
    previous = _stable_dict(state)
    channel_state = _channel_state(state, proposal)
    channel_state["trim_db"] = _coerce_float(proposal.requested_state.get("trim_db"))
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=_coerce_channel(proposal),
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_fader(seq: int, proposal: ReplayDecisionProposal, state: dict[str, Any]) -> ReplayExecutionEvent:
    previous = _stable_dict(state)
    channel_state = _channel_state(state, proposal)
    channel_state["fader_db"] = _coerce_float(proposal.requested_state.get("fader_db"))
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=_coerce_channel(proposal),
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_eq(seq: int, proposal: ReplayDecisionProposal, state: dict[str, Any]) -> ReplayExecutionEvent:
    previous = _stable_dict(state)
    channel_state = _channel_state(state, proposal)
    channel_state["eq"] = deepcopy(proposal.requested_state)
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=_coerce_channel(proposal),
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_compression(seq: int, proposal: ReplayDecisionProposal, state: dict[str, Any]) -> ReplayExecutionEvent:
    previous = _stable_dict(state)
    channel_state = _channel_state(state, proposal)
    channel_state["compression"] = deepcopy(proposal.requested_state)
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=_coerce_channel(proposal),
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_room_compensation(seq: int, proposal: ReplayDecisionProposal, state: dict[str, Any]) -> ReplayExecutionEvent:
    previous = _stable_dict(state)
    state["room_compensation"] = deepcopy(proposal.requested_state)
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=None,
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_speech_priority(seq: int, proposal: ReplayDecisionProposal, state: dict[str, Any]) -> ReplayExecutionEvent:
    previous = _stable_dict(state)
    state["speech_priority"] = deepcopy(proposal.requested_state)
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=None,
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_feedback(seq: int, proposal: ReplayDecisionProposal, state: dict[str, Any]) -> ReplayExecutionEvent:
    previous = _stable_dict(state)
    state.setdefault("feedback_mitigation", []).append(deepcopy(proposal.requested_state))
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=_coerce_channel(proposal),
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_operator_assistance(seq: int, proposal: ReplayDecisionProposal, state: dict[str, Any]) -> ReplayExecutionEvent:
    if proposal.auto_apply_blocked or proposal.safety.get("requires_operator") or proposal.safety.get("requires_confirmation"):
        return _emit_block(seq=seq, proposal=proposal, state=state, reason="operator_confirmation_required")
    previous = _stable_dict(state)
    state.setdefault("operator_notes", []).append(deepcopy(proposal.requested_state))
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=None,
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_performer_stage_interaction(
    seq: int,
    proposal: ReplayDecisionProposal,
    state: dict[str, Any],
) -> ReplayExecutionEvent:
    if proposal.auto_apply_blocked:
        return _emit_block(seq=seq, proposal=proposal, state=state, reason="performer_interaction_requires_review")
    previous = _stable_dict(state)
    state.setdefault("performer_interactions", []).append(deepcopy(proposal.requested_state))
    next_state = _stable_dict(state)
    return ReplayExecutionEvent(
        seq=seq,
        proposal_id=proposal.proposal_id,
        replay_correlation_id=proposal.replay_correlation_id,
        family=proposal.family,
        event_type="simulated",
        status="applied",
        channel=None,
        previous_state=previous,
        next_state=next_state,
        rollback_target=previous,
        state_signature=_state_signature(next_state),
        blocked_reason=None,
        payload=proposal.to_dict(),
    )


def _simulate_single_proposal(
    seq: int,
    proposal: ReplayDecisionProposal,
    state: dict[str, Any],
) -> ReplayExecutionEvent:
    family_handlers = {
        "gain_adjustment": _simulate_gain,
        "fader_adjustment": _simulate_fader,
        "eq_adjustment": _simulate_eq,
        "compression": _simulate_compression,
        "room_compensation": _simulate_room_compensation,
        "speech_arbitration": _simulate_speech_priority,
        "feedback_mitigation": _simulate_feedback,
        "operator_assistance": _simulate_operator_assistance,
        "performer_stage_interaction": _simulate_performer_stage_interaction,
    }
    handler = family_handlers.get(proposal.family)
    if handler is None:
        return _emit_block(seq=seq, proposal=proposal, state=state, reason="unsupported_family")
    return handler(seq, proposal, state)


def simulate_replay_decisions(
    proposals: list[ReplayDecisionProposal | dict[str, Any]],
    *,
    selected_proposal_id: str | None = None,
    initial_state: dict[str, Any] | None = None,
    rollback_to: int | str | None = None,
) -> ReplayExecutorResult:
    resolved = _coerce_proposals(proposals)
    state = _restore_from_payload(initial_state or {})
    starting_state = _stable_dict(state)
    selected = selected_proposal_id or (resolved[0].proposal_id if resolved else "")
    events: list[ReplayExecutionEvent] = []
    step_states: list[dict[str, Any]] = [starting_state]
    for seq, proposal in enumerate(resolved, start=1):
        event = _simulate_single_proposal(seq, proposal, state)
        events.append(event)
        step_states.append(_stable_dict(event.next_state))
    final_state = _stable_dict(state)
    if rollback_to is not None:
        if isinstance(rollback_to, int):
            index = max(0, min(rollback_to, len(step_states) - 1))
            final_state = step_states[index]
        else:
            for event in events:
                if event.proposal_id == rollback_to or event.replay_correlation_id == rollback_to:
                    final_state = _stable_dict(event.rollback_target)
                    break
    return ReplayExecutorResult(
        schema_version=REPLAY_EXECUTOR_SCHEMA_VERSION,
        selected_proposal_id=selected,
        initial_state=starting_state,
        final_state=final_state,
        final_state_signature=_state_signature(final_state),
        events=events,
    )


def build_replay_executor_from_rewind_context(
    rewind_context: ReplayRewindContext | dict[str, Any],
) -> ReplayExecutorResult:
    context = rewind_context if isinstance(rewind_context, ReplayRewindContext) else ReplayRewindContext(**rewind_context)
    graph = context.graph_checkpoint if isinstance(context.graph_checkpoint, ReplayGraphCheckpoint) else ReplayGraphCheckpoint.from_dict(context.graph_checkpoint)
    proposals = [ReplayDecisionProposal.from_dict(item) for item in graph.proposals]
    initial_state = {
        "__trim_snapshot__": context.trim_snapshot.payload,
        "channels": {},
    }
    for record in (context.trim_snapshot.last_state.get("applied") or []):
        channel = record.get("channel")
        if channel is not None:
            initial_state["channels"][str(channel)] = dict(record)
    return simulate_replay_decisions(
        proposals,
        selected_proposal_id=graph.selected_proposal_id,
        initial_state=initial_state,
    )
