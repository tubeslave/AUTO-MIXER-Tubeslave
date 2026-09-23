import pytest

from backend.live_runtime.contracts import ProposedAction, VerifiedAction
from backend.live_runtime.control_plane import RollbackExecution, WriteExecution
from backend.live_runtime.decision_engine import LiveHypothesis
from backend.live_runtime.iteration import IterationCoordinator, IterationPhase


def _hypothesis(name="headroom"):
    action = ProposedAction(
        "main:1",
        "fader_delta_db",
        -0.5,
        "restore main headroom",
        0.95,
        max_step=0.5,
    )
    return LiveHypothesis(
        name,
        "main:1",
        "bounded test hypothesis",
        0.95,
        action,
        "main_peak_dbfs",
        0.25,
    )


class FakeRuntime:
    def __init__(self, *, accepted=True, restored=True, wrote=True):
        self.accepted = accepted
        self.restored = restored
        self.wrote = wrote
        self.execute_calls = 0
        self.rollback_calls = 0

    def execute_action(self, action, *, manual_freeze=False):
        self.execute_calls += 1
        verified = VerifiedAction(
            proposal=action,
            before=-6.0,
            after=-6.5 if self.wrote else -6.0,
            readback=-6.5 if self.accepted and self.wrote else -6.2,
            accepted=self.accepted if self.wrote else False,
            rollback_value=-6.0 if self.wrote else None,
        )
        return WriteExecution(
            verified=verified,
            authorization_reason="allowed" if self.wrote else "observe_only",
            wrote=self.wrote,
        )

    def rollback_action(self, verified, *, manual_freeze=False):
        self.rollback_calls += 1
        return RollbackExecution(
            original=verified,
            before_rollback=verified.readback,
            readback=verified.rollback_value if self.restored else verified.readback,
            restored=self.restored,
            authorization_reason="rollback_allowed",
            wrote=True,
        )


def test_only_one_hypothesis_can_wait_for_verification():
    runtime = FakeRuntime()
    clock = [10.0]
    coordinator = IterationCoordinator(runtime, verification_window_s=1.0, clock=lambda: clock[0])

    result = coordinator.start(_hypothesis(), {"main_peak_dbfs": -1.0})
    assert result.phase is IterationPhase.VERIFY_PENDING
    assert coordinator.has_active_hypothesis

    with pytest.raises(RuntimeError, match="already awaiting verification"):
        coordinator.start(_hypothesis("second"), {"main_peak_dbfs": -1.0})
    assert runtime.execute_calls == 1


def test_verify_window_waits_then_keeps_improved_hypothesis():
    runtime = FakeRuntime()
    clock = [20.0]
    coordinator = IterationCoordinator(runtime, verification_window_s=1.0, clock=lambda: clock[0])
    coordinator.start(_hypothesis(), {"main_peak_dbfs": -1.0, "feedback_risk": 0.1})

    clock[0] = 20.4
    waiting = coordinator.verify({"main_peak_dbfs": -1.6, "feedback_risk": 0.1})
    assert waiting.phase is IterationPhase.VERIFY_WAIT
    assert coordinator.has_active_hypothesis
    assert runtime.rollback_calls == 0

    clock[0] = 21.1
    kept = coordinator.verify({"main_peak_dbfs": -1.6, "feedback_risk": 0.1})
    assert kept.phase is IterationPhase.KEPT
    assert kept.critic["accept"] is True
    assert not coordinator.has_active_hypothesis
    assert runtime.rollback_calls == 0


def test_critic_regression_uses_verified_rollback():
    runtime = FakeRuntime(restored=True)
    coordinator = IterationCoordinator(runtime, verification_window_s=0)
    coordinator.start(_hypothesis(), {"main_peak_dbfs": -1.0, "feedback_risk": 0.1})

    result = coordinator.verify({"main_peak_dbfs": -0.9, "feedback_risk": 0.1})
    assert result.phase is IterationPhase.ROLLED_BACK
    assert result.reason == "critic_rejected"
    assert result.rollback.restored is True
    assert runtime.rollback_calls == 1
    assert not coordinator.has_active_hypothesis
    assert coordinator.hold_reason is None


def test_operator_touch_has_priority_and_does_not_fight_with_rollback():
    runtime = FakeRuntime()
    coordinator = IterationCoordinator(runtime, verification_window_s=0)
    coordinator.start(_hypothesis(), {"main_peak_dbfs": -1.0, "feedback_risk": 0.1})

    result = coordinator.verify(
        {"main_peak_dbfs": -1.6, "feedback_risk": 0.1, "operator_touch": True}
    )
    assert result.phase is IterationPhase.HOLD
    assert result.reason == "operator_took_control"
    assert runtime.rollback_calls == 0
    assert coordinator.hold_reason == "operator_took_control"

    with pytest.raises(RuntimeError, match="on HOLD"):
        coordinator.start(_hypothesis("later"), {"main_peak_dbfs": -1.0})


def test_apply_readback_mismatch_rolls_back_before_critic_window():
    runtime = FakeRuntime(accepted=False, restored=True)
    coordinator = IterationCoordinator(runtime, verification_window_s=1.0)

    result = coordinator.start(_hypothesis(), {"main_peak_dbfs": -1.0})
    assert result.phase is IterationPhase.ROLLED_BACK
    assert result.reason == "apply_readback_mismatch"
    assert runtime.rollback_calls == 1
    assert not coordinator.has_active_hypothesis


def test_failed_rollback_enters_hold_and_blocks_more_autonomy():
    runtime = FakeRuntime(restored=False)
    coordinator = IterationCoordinator(runtime, verification_window_s=0)
    coordinator.start(_hypothesis(), {"main_peak_dbfs": -1.0, "feedback_risk": 0.1})

    result = coordinator.verify({"main_peak_dbfs": -0.9, "feedback_risk": 0.1})
    assert result.phase is IterationPhase.HOLD
    assert result.reason == "critic_rejected_rollback_failed"
    assert coordinator.hold_reason == "critic_rejected_rollback_failed"

    with pytest.raises(RuntimeError, match="on HOLD"):
        coordinator.start(_hypothesis("later"), {"main_peak_dbfs": -1.0})


def test_observe_or_policy_block_never_opens_verify_window():
    runtime = FakeRuntime(wrote=False)
    coordinator = IterationCoordinator(runtime, verification_window_s=0)

    result = coordinator.start(_hypothesis(), {"main_peak_dbfs": -1.0})
    assert result.phase is IterationPhase.APPLY_BLOCKED
    assert result.reason == "observe_only"
    assert not coordinator.has_active_hypothesis
    assert runtime.rollback_calls == 0
