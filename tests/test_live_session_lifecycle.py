from backend.live_runtime.session_lifecycle import LiveSessionLifecycle, LiveSessionPhase


def test_lifecycle_emits_policy_free_start_stop_events_and_delegates_owner_stop():
    events = []
    audits = []
    stop_calls = []

    lifecycle = None

    def stop_owner():
        stop_calls.append(True)
        lifecycle.begin_stop()
        lifecycle.mark_stopped()
        return True

    lifecycle = LiveSessionLifecycle(
        on_state_change=lambda state, message: events.append((state, message)),
        audit_sink=audits.append,
        stop_callback=stop_owner,
    )

    lifecycle.begin_start()
    lifecycle.mark_running()

    assert lifecycle.active is True
    assert lifecycle.state is LiveSessionPhase.RUNNING
    assert lifecycle.get_status()["legacy_engine_attached"] is False
    assert lifecycle.stop() is True
    assert lifecycle.state is LiveSessionPhase.STOPPED
    assert lifecycle.active is False
    assert len(stop_calls) == 1
    assert [state for state, _ in events] == [
        "starting",
        "running",
        "stopping",
        "stopped",
    ]
    assert [event["event"] for event in audits] == ["live_session_lifecycle"] * 4


def test_lifecycle_error_is_inactive_and_preserves_reason():
    lifecycle = LiveSessionLifecycle()
    lifecycle.begin_start()
    lifecycle.mark_error("transport failed")

    assert lifecycle.state is LiveSessionPhase.ERROR
    assert lifecycle.active is False
    assert lifecycle.get_status()["lifecycle_error"] == "transport failed"
