"""Read-only operator cockpit state handlers."""

import time

from operator_product_state import (
    build_agent_communications_snapshot,
    build_bus_main_dca_snapshot,
    build_channel_intelligence,
    build_channel_processing_snapshot,
    build_operator_agent_roster,
    build_processing_pipeline_snapshot,
    build_channel_inventory,
    build_connection_topology,
    build_dashboard_snapshot,
    build_decision_queue,
    build_decision_traces,
    build_groups_buses_snapshot,
    build_matrix_snapshot,
    build_memory_knowledge_snapshot,
    build_replay_runs,
    build_routing_sends_snapshot,
    build_rules_library_snapshot,
    build_safety_monitor,
    build_snapshots_scenes_snapshot,
    build_telemetry_alerts_snapshot,
    build_training_sessions_snapshot,
    build_user_presets_snapshot,
)
from operator_analysis import build_operator_analysis_report
from processing_pipeline_preview import build_main_pipeline_preview


VERIFIED_MANUAL_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}
APPLY_DELTA_LIMITS_DB = {"fader": 1.0, "gain": 0.5}
CUMULATIVE_DELTA_LIMITS_DB = {"fader": 3.0, "gain": 1.5}
CUMULATIVE_DELTA_WINDOW_SEC = 300.0


def manual_write_result_verified(result):
    readback_status = result.get("readback_status")
    proof_status = result.get("proof_status")
    if proof_status is not None and proof_status != "verified":
        return False
    return readback_status in VERIFIED_MANUAL_READBACK_STATUSES


def register_handlers(server):
    def proposal_id_or_index(data):
        return data.get("proposal_id") or data.get("id"), data.get("index")

    def coerce_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def is_operator_authored(proposal, data):
        raw = getattr(proposal, "raw", {}) if proposal is not None else {}
        return bool(
            data.get("operator_authored")
            or raw.get("operator_authored")
            or raw.get("authored_by") == "operator"
            or raw.get("source") == "operator_authored"
        )

    def operator_apply_delta_history():
        history = getattr(server, "_operator_apply_delta_history", None)
        if not isinstance(history, dict):
            history = {}
            setattr(server, "_operator_apply_delta_history", history)
        return history

    def cumulative_delta_blocker(normalized_change):
        value_type = normalized_change.get("value_type")
        limit = CUMULATIVE_DELTA_LIMITS_DB.get(value_type)
        if limit is None:
            return None
        delta = coerce_float(normalized_change.get("delta_db"))
        if delta is None:
            return None
        key = (int(normalized_change["channel"]), value_type)
        now = time.monotonic()
        history = operator_apply_delta_history()
        entries = [
            entry
            for entry in history.get(key, [])
            if now - float(entry.get("ts", 0.0)) <= CUMULATIVE_DELTA_WINDOW_SEC
        ]
        history[key] = entries
        used = round(sum(abs(float(entry.get("delta_db", 0.0))) for entry in entries), 3)
        requested = round(abs(float(delta)), 3)
        if used + requested > limit:
            return {
                "reason": "cumulative_delta_limit_exceeded",
                "channel": normalized_change["channel"],
                "value_type": value_type,
                "delta_db": normalized_change["delta_db"],
                "cumulative_delta_db": used,
                "requested_delta_db": requested,
                "max_cumulative_delta_db": limit,
                "window_sec": CUMULATIVE_DELTA_WINDOW_SEC,
            }
        return None

    def remember_applied_delta(normalized_change, apply_result):
        if apply_result.get("status") != "applied":
            return
        if not manual_write_result_verified(apply_result):
            return
        if not apply_result.get("rollback_available"):
            return
        delta = coerce_float(normalized_change.get("delta_db"))
        if delta is None:
            return
        key = (int(normalized_change["channel"]), normalized_change["value_type"])
        history = operator_apply_delta_history()
        history.setdefault(key, []).append({
            "ts": time.monotonic(),
            "delta_db": delta,
            "proposal_id": apply_result.get("proposal_id"),
            "approval_id": apply_result.get("approval_id"),
        })

    def trace_payload_for_result(
        *,
        proposal=None,
        operator_mode=None,
        requested_change=None,
        normalized_change=None,
        apply_result=None,
        idempotency_key=None,
        status="blocked",
        blocked_reason=None,
        metadata=None,
    ):
        change = normalized_change or requested_change or {}
        result = apply_result or {}
        value_type = str(result.get("value_type") or change.get("value_type") or getattr(proposal, "kind", "") or "").strip().lower()
        target = result.get("channel", change.get("channel", getattr(proposal, "target", None)))
        current_value = result.get("old_value", change.get("old_value", change.get("current_value")))
        proposed_value = result.get("effective_value", result.get("requested_value", change.get("effective_value", change.get("value"))))
        delta = result.get("delta_db", change.get("delta_db"))
        if delta is None:
            current_float = coerce_float(current_value)
            proposed_float = coerce_float(proposed_value)
            if current_float is not None and proposed_float is not None:
                delta = round(proposed_float - current_float, 3)
        mode = (operator_mode or {}).get("mode") if isinstance(operator_mode, dict) else getattr(server, "operator_mode", None)
        readback = None
        if result.get("readback_status") or result.get("readback_value") is not None:
            readback = {
                "status": result.get("readback_status"),
                "source": result.get("readback_source"),
                "value": result.get("readback_value"),
            }
        rollback = None
        if result.get("rollback_available") is not None or result.get("rollback_status") is not None:
            rollback = {
                "available": result.get("rollback_available"),
                "value": result.get("rollback_value"),
                "status": result.get("rollback_status"),
                "readback_status": result.get("rollback_readback_status"),
                "proof_status": result.get("rollback_proof_status"),
            }
        pre_read = None
        pre_read_status = result.get("pre_read_status", change.get("pre_read_status"))
        pre_read_value = result.get("pre_read_value", change.get("pre_read_value", current_value))
        if pre_read_status or pre_read_value is not None:
            pre_read = {
                "status": pre_read_status,
                "source": result.get("pre_read_source", change.get("pre_read_source")),
                "value": pre_read_value,
            }
        return {
            "idempotency_key": idempotency_key,
            "proposal_id": getattr(proposal, "id", None),
            "source": getattr(proposal, "source", "operator_proposal_queue") if proposal else "operator_apply",
            "mode": mode,
            "target": target,
            "action_type": value_type or "proposal",
            "status": status,
            "current_value": current_value,
            "proposed_value": proposed_value,
            "delta": delta,
            "safety_limits": {
                "allowed_live_write_kinds": ["fader", "gain"],
                "max_delta_db": APPLY_DELTA_LIMITS_DB.get(value_type),
                "upward_auto_generated_moves_allowed": False,
            },
            "pre_read": pre_read,
            "readback": readback,
            "rollback": rollback,
            "proof_status": result.get("proof_status", "blocked" if blocked_reason else "unverified"),
            "blocked_reason": blocked_reason,
            "metadata": metadata or {},
        }

    def record_apply_trace(**kwargs):
        recorder = getattr(server, "record_automation_decision_trace", None)
        if callable(recorder):
            recorder(**trace_payload_for_result(**kwargs))

    async def validate_change_before_apply(
        change,
        *,
        proposal,
        data,
        operator_mode,
        linked=False,
    ):
        normalized, error = normalize_linked_change(change)
        if error:
            return None, {
                "reason": error,
                "requested_change": change,
            }
        read_value = None
        read_helper = getattr(server, "_manual_write_read_value", None)
        if callable(read_helper):
            read_value = await read_helper(
                channel=normalized["channel"],
                value_type=normalized["value_type"],
                address=normalized["address"],
                prefer_physical=True,
            )
            old_value = read_value.get("value")
        else:
            state = getattr(getattr(server, "mixer_client", None), "state", {})
            old_value = state.get(normalized["address"]) if isinstance(state, dict) else None
        old_float = coerce_float(old_value)
        if old_float is None:
            return None, {
                "reason": "linked_pre_read_unavailable" if linked else "pre_read_unavailable",
                "channel": normalized["channel"],
                "value_type": normalized["value_type"],
                "address": normalized["address"],
                "pre_read_status": (read_value or {}).get("status"),
                "pre_read_source": (read_value or {}).get("source"),
            }
        normalized["old_value"] = old_value
        normalized["pre_read_value"] = old_value
        normalized["pre_read_status"] = (read_value or {}).get("status")
        normalized["pre_read_source"] = (read_value or {}).get("source")
        normalized["delta_db"] = round(float(normalized["effective_value"]) - old_float, 3)
        max_delta = APPLY_DELTA_LIMITS_DB.get(normalized["value_type"])
        if max_delta is not None and abs(normalized["delta_db"]) > max_delta:
            return normalized, {
                "reason": "delta_limit_exceeded",
                "channel": normalized["channel"],
                "value_type": normalized["value_type"],
                "delta_db": normalized["delta_db"],
                "max_delta_db": max_delta,
            }
        if normalized["delta_db"] > 0 and not is_operator_authored(proposal, data):
            return normalized, {
                "reason": "upward_move_requires_operator_authored_proposal",
                "channel": normalized["channel"],
                "value_type": normalized["value_type"],
                "delta_db": normalized["delta_db"],
            }
        cumulative_error = cumulative_delta_blocker(normalized)
        if cumulative_error:
            return normalized, cumulative_error
        return normalized, None

    async def handle_get_dashboard_snapshot(websocket, data):
        await server.send_to_client(websocket, build_dashboard_snapshot(server))

    async def handle_get_channel_inventory(websocket, data):
        total_channels = data.get("total_channels", 24)
        await server.send_to_client(
            websocket,
            build_channel_inventory(server, total_channels=total_channels),
        )

    async def handle_get_channel_processing_snapshot(websocket, data):
        total_channels = data.get("total_channels", 24)
        await server.send_to_client(
            websocket,
            build_channel_processing_snapshot(server, total_channels=total_channels),
        )

    async def handle_get_decision_queue(websocket, data):
        limit = data.get("limit", 50)
        await server.send_to_client(websocket, build_decision_queue(server, limit=limit))

    async def handle_get_decision_traces(websocket, data):
        limit = data.get("limit", 50)
        await server.send_to_client(websocket, build_decision_traces(server, limit=limit))

    async def handle_get_safety_monitor(websocket, data):
        del data
        await server.send_to_client(websocket, build_safety_monitor(server))

    async def handle_get_replay_runs(websocket, data):
        limit = data.get("limit", 20)
        await server.send_to_client(websocket, build_replay_runs(server, limit=limit))

    async def handle_start_replay_run(websocket, data):
        try:
            from agentic_method_foundation import build_offline_teacher_examples_from_channel_inventory
            from offline_validation_harness import compact_stamp, run_offline_validation, write_offline_validation_artifacts
        except ImportError:  # pragma: no cover - package import compatibility
            from backend.agentic_method_foundation import build_offline_teacher_examples_from_channel_inventory
            from backend.offline_validation_harness import compact_stamp, run_offline_validation, write_offline_validation_artifacts

        total_channels = int(data.get("total_channels", 16) or 16)
        run_id = str(data.get("run_id") or "").strip() or f"{compact_stamp()}_offline_operator_validation"
        result = run_offline_validation(total_channels=total_channels)
        inventory = build_channel_inventory(server, total_channels=total_channels)
        teacher_examples = build_offline_teacher_examples_from_channel_inventory(
            inventory.get("channels", []),
            run_id=run_id,
            reference_metrics={
                "offline_validation_status": result.get("status"),
                "offline_validation_summary": result.get("summary", {}),
            },
        )
        paths = write_offline_validation_artifacts(
            result,
            artifact_root=getattr(server, "_replay_artifacts_dir", "artifacts/replay"),
            run_id=run_id,
            teacher_examples=teacher_examples,
        )
        recorder = getattr(server, "record_automation_decision_trace", None)
        if callable(recorder):
            recorder(
                source="offline_replay_v1",
                mode="offline",
                target=paths.get("run_id"),
                action_type="replay_run",
                status=result.get("status"),
                safety_limits={
                    "offline_only": True,
                    "live_write": False,
                    "mixer_transport_allowed": False,
                    "allowed_live_write_kinds": [],
                },
                proof_status="offline",
                blocked_reason=None if result.get("status") in {"passed", "clear"} else result.get("reason"),
                metadata={
                    "run_id": paths.get("run_id"),
                    "artifact_paths": paths,
                    "summary": result.get("summary", {}),
                    "foundation_teacher_example_count": len(teacher_examples),
                    "connection_policy": result.get("connection_policy"),
                    "write_policy": result.get("write_policy"),
                    "forbidden_messages": result.get("forbidden_messages", []),
                    "offline_only": True,
                    "live_write": False,
                },
            )
        payload = {
            "type": "replay_run_started",
            "status": result.get("status"),
            "success": result.get("status") in {"passed", "clear"},
            "offline_only": True,
            "live_write": False,
            "run_id": (paths.get("run_dir") or "").split("/")[-1],
            "result": result,
            "artifacts": paths,
            "foundation_teacher_example_count": len(teacher_examples),
        }
        await server.send_to_client(websocket, payload)

    async def handle_get_channel_intelligence(websocket, data):
        await server.send_to_client(
            websocket,
            build_channel_intelligence(server, total_channels=data.get("total_channels", 48)),
        )

    async def handle_get_connection_topology(websocket, data):
        await server.send_to_client(websocket, build_connection_topology(server))

    async def handle_get_operator_agent_roster(websocket, data):
        del data
        await server.send_to_client(websocket, build_operator_agent_roster(server))

    async def handle_get_groups_buses_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_groups_buses_snapshot(server))

    async def handle_get_bus_main_dca_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_bus_main_dca_snapshot(
                server,
                bus_count=data.get("bus_count", 16),
                main_count=data.get("main_count", 4),
                dca_count=data.get("dca_count", 16),
            ),
        )

    async def handle_get_matrix_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_matrix_snapshot(
                server,
                matrix_count=data.get("matrix_count", 8),
            ),
        )

    async def handle_get_routing_sends_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_routing_sends_snapshot(
                server,
                total_channels=data.get("total_channels", 24),
                send_count=data.get("send_count", 16),
                main_count=data.get("main_count", 1),
                output_groups=data.get("output_groups", ("MOD", "CRD")),
                output_count=data.get("output_count", 64),
            ),
        )

    async def handle_get_snapshots_scenes_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_snapshots_scenes_snapshot(
                server,
                snapshot_count=data.get("snapshot_count", 200),
            ),
        )

    async def handle_get_user_presets_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_user_presets_snapshot(
                server,
                preset_count=data.get("preset_count", 200),
            ),
        )

    async def handle_get_memory_knowledge_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_memory_knowledge_snapshot(server))

    async def handle_get_training_sessions_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_training_sessions_snapshot(server))

    async def handle_get_telemetry_alerts_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_telemetry_alerts_snapshot(
                server,
                limit=data.get("limit", 50),
                session_limit=data.get("session_limit", 8),
            ),
        )

    async def handle_get_rules_library_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_rules_library_snapshot(server))

    async def handle_get_agent_communications_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_agent_communications_snapshot(server))

    async def handle_get_processing_pipeline_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_processing_pipeline_snapshot(
                server,
                include_methods=bool(data.get("include_methods", True)),
            ),
        )

    async def handle_get_main_pipeline_preview(websocket, data):
        await server.send_to_client(
            websocket,
            build_main_pipeline_preview(
                server,
                total_channels=data.get("total_channels", 24),
                include_methods=bool(data.get("include_methods", False)),
            ),
        )

    async def handle_get_operator_mode(websocket, data):
        del data
        await server.send_to_client(websocket, server.get_operator_mode_status())

    async def handle_set_operator_mode(websocket, data):
        await server.send_to_client(
            websocket,
            server.set_operator_mode(
                data.get("mode"),
                reason=data.get("reason", "websocket_operator_mode_change"),
            ),
        )

    async def handle_run_operator_analysis(websocket, data):
        await server.send_to_client(
            websocket,
            build_operator_analysis_report(
                server,
                total_channels=data.get("total_channels", 24),
                create_proposals=data.get("create_proposals"),
            ),
        )

    async def handle_import_safe_gain_suggestions(websocket, data):
        del data
        await server.send_to_client(
            websocket,
            server.import_safe_gain_suggestions_to_operator_queue(),
        )

    async def handle_import_soundcheck_recommendations(websocket, data):
        await server.send_to_client(
            websocket,
            server.import_soundcheck_recommendations_to_operator_queue(data.get("bundle")),
        )

    async def handle_create_operator_proposal(websocket, data):
        payload = dict(data.get("proposal") or data)
        payload.pop("type", None)
        result = server.operator_proposal_queue.create(payload, server.get_operator_mode_status())
        proposal = result.get("proposal") if isinstance(result, dict) else None
        if isinstance(proposal, dict):
            requested_change = proposal.get("requested_change") or {}
            recorder = getattr(server, "record_automation_decision_trace", None)
            if callable(recorder):
                recorder(
                    proposal_id=proposal.get("id"),
                    source=proposal.get("source") or "operator_proposal_queue",
                    mode=proposal.get("mode_at_creation"),
                    target=proposal.get("target"),
                    action_type=proposal.get("kind") or requested_change.get("value_type") or "proposal",
                    status="proposal_created" if result.get("status") in {"created", "exists"} else str(result.get("status")),
                    proposed_value=requested_change.get("value"),
                    delta=requested_change.get("delta_db"),
                    safety_limits={
                        "allowed_live_write_kinds": ["fader", "gain"],
                        "proposal_only_if_unsupported": True,
                    },
                    proof_status="pending",
                    blocked_reason=result.get("reason") if not result.get("success") else None,
                    metadata={"result_status": result.get("status")},
                )
        await server.send_to_client(websocket, result)

    async def handle_accept_operator_proposal(websocket, data):
        proposal_id, index = proposal_id_or_index(data)
        result = server.operator_proposal_queue.accept(proposal_id=proposal_id, index=index)
        await server.send_to_client(websocket, {"type": "operator_proposal_accepted", **result})

    async def handle_dismiss_operator_proposal(websocket, data):
        proposal_id, index = proposal_id_or_index(data)
        result = server.operator_proposal_queue.dismiss(
            proposal_id=proposal_id,
            index=index,
            reason=data.get("reason", "dismissed_by_operator"),
        )
        await server.send_to_client(websocket, {"type": "operator_proposal_dismissed", **result})

    def linked_change_blocked_result(proposal, reason, **extra):
        return server.operator_proposal_queue.mark_apply_result(
            proposal,
            {
                "status": "blocked",
                "reason": reason,
                **extra,
            },
        )

    def normalize_linked_change(change):
        value_type = server._normalize_manual_write_kind(change.get("value_type"))
        try:
            channel = int(change.get("channel"))
        except (TypeError, ValueError):
            return None, "invalid_channel"
        try:
            value = float(change.get("value"))
        except (TypeError, ValueError):
            return None, "invalid_value"
        address = server._manual_write_address(channel, value_type)
        if not address:
            return None, "unsupported_value_type"
        return {
            "value_type": value_type,
            "channel": channel,
            "value": value,
            "effective_value": server._clamp_manual_write_value(value, value_type),
            "address": address,
        }, None

    async def rollback_linked_results(websocket, proposal, applied_results, approval_id, reason):
        rollback_results = []
        for index, applied in reversed(list(enumerate(applied_results))):
            rollback_result = await server._apply_manual_console_write(
                websocket,
                channel=applied.get("channel"),
                value=applied.get("rollback_value"),
                value_type=applied.get("value_type"),
                source=f"operator_proposal_linked_rollback:{proposal.id}",
                reason=reason,
                approved=True,
                approval_id=f"rollback:{approval_id}:{index + 1}",
                bypass_cooldown=True,
            )
            rollback_result.update({
                "rollback": True,
                "rollback_of": applied.get("approval_id"),
                "rollback_source": applied.get("source"),
            })
            rollback_results.append(rollback_result)
        return rollback_results

    async def apply_linked_operator_proposal(websocket, data, proposal, operator_mode, requested_changes, idempotency_key):
        allowed_kinds = set(operator_mode["capabilities"].get("allowed_live_write_kinds") or [])
        if not operator_mode["capabilities"]["can_apply_to_console"]:
            result = linked_change_blocked_result(
                proposal,
                "operator_mode_blocks_live_write",
                operator_mode=operator_mode,
                requested_changes=requested_changes,
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_changes[0] if requested_changes else {},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="operator_mode_blocks_live_write",
                metadata={"linked": True, "requested_changes": requested_changes},
            )
            await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
            return

        if not bool(data.get("approved", False)):
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_changes[0] if requested_changes else {},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="approval_required",
                metadata={"linked": True},
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                "status": "blocked",
                "success": False,
                "reason": "approval_required",
                "proposal": proposal.to_dict(),
            })
            return

        if not server.mixer_client or not getattr(server.mixer_client, "is_connected", True):
            result = linked_change_blocked_result(proposal, "mixer_not_connected")
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_changes[0] if requested_changes else {},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="mixer_not_connected",
                metadata={"linked": True},
            )
            await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
            return

        apply_manual_console_write = getattr(server, "_apply_manual_console_write", None)
        if apply_manual_console_write is None:
            result = linked_change_blocked_result(proposal, "supervised_apply_path_unavailable")
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_changes[0] if requested_changes else {},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="supervised_apply_path_unavailable",
                metadata={"linked": True},
            )
            await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
            return

        normalized_changes = []
        for raw_change in requested_changes:
            normalized, error = await validate_change_before_apply(
                raw_change,
                proposal=proposal,
                data=data,
                operator_mode=operator_mode,
                linked=True,
            )
            if error:
                result = linked_change_blocked_result(
                    proposal,
                    error["reason"],
                    requested_change=raw_change,
                    requested_changes=requested_changes,
                    **{key: value for key, value in error.items() if key not in {"reason", "requested_change"}},
                )
                record_apply_trace(
                    proposal=proposal,
                    operator_mode=operator_mode,
                    requested_change=raw_change,
                    normalized_change=normalized,
                    idempotency_key=idempotency_key,
                    status="blocked",
                    blocked_reason=error["reason"],
                    metadata={"linked": True, "requested_changes": requested_changes},
                )
                await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
                return
            if normalized["value_type"] not in allowed_kinds:
                result = linked_change_blocked_result(
                    proposal,
                    "unsupported_value_type",
                    value_type=normalized["value_type"],
                    allowed_live_write_kinds=sorted(allowed_kinds),
                )
                record_apply_trace(
                    proposal=proposal,
                    operator_mode=operator_mode,
                    normalized_change=normalized,
                    idempotency_key=idempotency_key,
                    status="blocked",
                    blocked_reason="unsupported_value_type",
                    metadata={"linked": True},
                )
                await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
                return
            normalized_changes.append(normalized)

        approval_id = data.get("approval_id") or proposal.id
        applied_results = []
        for index, change in enumerate(normalized_changes):
            apply_result = await apply_manual_console_write(
                websocket,
                channel=change["channel"],
                value=change["effective_value"],
                value_type=change["value_type"],
                source=f"operator_proposal_linked:{proposal.id}:{index + 1}",
                reason=data.get("reason") or proposal.reason or proposal.title,
                approved=True,
                approval_id=f"{approval_id}:{index + 1}",
            )
            applied_results.append(apply_result)
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                normalized_change=change,
                apply_result=apply_result,
                idempotency_key=idempotency_key,
                status=apply_result.get("status", "unknown"),
                blocked_reason=apply_result.get("reason") if apply_result.get("status") != "applied" else None,
                metadata={"linked": True, "linked_index": index + 1},
            )
            verified = manual_write_result_verified(apply_result)
            rollback_ready = bool(apply_result.get("rollback_available"))
            if apply_result.get("status") != "applied" or not verified or not rollback_ready:
                rollback_candidates = [
                    result for result in applied_results
                    if result.get("status") == "applied" and result.get("rollback_available")
                ]
                rollback_results = await rollback_linked_results(
                    websocket,
                    proposal,
                    rollback_candidates,
                    approval_id,
                    "rollback:linked_operator_proposal_failed",
                )
                result = linked_change_blocked_result(
                    proposal,
                    "linked_apply_failed_rolled_back",
                    failed_result=apply_result,
                    apply_results=applied_results,
                    rollback_results=rollback_results,
                    requested_changes=requested_changes,
                )
                for rollback_result in rollback_results:
                    record_apply_trace(
                        proposal=proposal,
                        operator_mode=operator_mode,
                        apply_result=rollback_result,
                        idempotency_key=idempotency_key,
                        status=rollback_result.get("status", "unknown"),
                        blocked_reason=rollback_result.get("reason") if rollback_result.get("status") != "applied" else None,
                        metadata={"linked": True, "rollback": True},
                    )
                await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
                return

        for change, apply_result in zip(normalized_changes, applied_results):
            remember_applied_delta(change, apply_result)

        result = server.operator_proposal_queue.mark_apply_result(
            proposal,
            {
                "status": "applied",
                "success": True,
                "linked": True,
                "reason": data.get("reason") or proposal.reason or proposal.title,
                "approval_id": approval_id,
                "apply_results": applied_results,
                "readback_status": "verified_from_physical_readback"
                if any(result.get("readback_status") == "verified_from_physical_readback" for result in applied_results)
                else "verified_from_cache",
                "proof_status": "verified",
                "rollback_available": all(result.get("rollback_available") for result in applied_results),
                "requested_changes": requested_changes,
            },
        )
        await server.send_to_client(websocket, {"type": "operator_proposal_apply_result", **result})

    async def handle_apply_operator_proposal(websocket, data):
        proposal_id, index = proposal_id_or_index(data)
        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id, index=index)
        operator_mode = server.get_operator_mode_status()
        if proposal is None:
            recorder = getattr(server, "record_automation_decision_trace", None)
            if callable(recorder):
                recorder(
                    idempotency_key=data.get("idempotency_key"),
                    proposal_id=proposal_id,
                    source="operator_apply",
                    mode=operator_mode.get("mode"),
                    target=None,
                    action_type="proposal_apply",
                    status="blocked",
                    safety_limits={"requires_idempotency_key": True},
                    proof_status="blocked",
                    blocked_reason="proposal_not_found",
                    metadata={"index": index},
                )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                "status": "blocked",
                "success": False,
                "reason": "proposal_not_found",
                "proposal_id": proposal_id,
                "index": index,
            })
            return

        if proposal.is_expired():
            result = server.operator_proposal_queue.mark_apply_result(
                proposal,
                {
                    "status": "blocked",
                    "reason": "proposal_expired",
                    "expires_at": proposal.expires_at,
                    "ttl_seconds": proposal.ttl_seconds,
                },
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=(proposal.requested_change or {}),
                idempotency_key=data.get("idempotency_key"),
                status="blocked",
                blocked_reason="proposal_expired",
                metadata={"expires_at": proposal.expires_at, "ttl_seconds": proposal.ttl_seconds},
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        idempotency_key = str(data.get("idempotency_key") or "").strip()
        reservation = server.reserve_operator_apply_idempotency_key(
            idempotency_key,
            proposal_id=getattr(proposal, "id", None),
            approval_id=data.get("approval_id"),
        )
        if not reservation.get("reserved"):
            reason = reservation.get("reason")
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=(proposal.requested_change or {}),
                idempotency_key=idempotency_key or None,
                status="blocked",
                blocked_reason=reason,
                metadata={"idempotency": reservation},
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                "status": "blocked",
                "success": False,
                "reason": reason,
                "proposal": proposal.to_dict(),
                "idempotency_key": idempotency_key or None,
                "idempotency": reservation,
            })
            return

        requested_changes = getattr(proposal, "requested_changes", []) or []
        if requested_changes:
            await apply_linked_operator_proposal(websocket, data, proposal, operator_mode, requested_changes, idempotency_key)
            return

        requested_change = proposal.requested_change or {}
        value_type = str(requested_change.get("value_type") or "").strip().lower()
        allowed_kinds = set(operator_mode["capabilities"].get("allowed_live_write_kinds") or [])
        if (
            not value_type
            or requested_change.get("channel") is None
            or requested_change.get("value") is None
        ):
            result = server.operator_proposal_queue.mark_apply_result(
                proposal,
                {
                    "status": "blocked",
                    "reason": "proposal_has_no_applyable_requested_change",
                    "requested_change": requested_change or None,
                },
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="proposal_has_no_applyable_requested_change",
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        if not operator_mode["capabilities"]["can_apply_to_console"]:
            result = server.operator_proposal_queue.mark_apply_result(
                proposal,
                {
                    "status": "blocked",
                    "reason": "operator_mode_blocks_live_write",
                    "operator_mode": operator_mode,
                },
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="operator_mode_blocks_live_write",
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        if value_type not in allowed_kinds:
            result = server.operator_proposal_queue.mark_apply_result(
                proposal,
                {
                    "status": "blocked",
                    "reason": "unsupported_value_type",
                    "value_type": value_type,
                    "allowed_live_write_kinds": sorted(allowed_kinds),
                },
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="unsupported_value_type",
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        if not bool(data.get("approved", False)):
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="approval_required",
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                "status": "blocked",
                "success": False,
                "reason": "approval_required",
                "proposal": proposal.to_dict(),
            })
            return

        if not server.mixer_client or not getattr(server.mixer_client, "is_connected", True):
            result = server.operator_proposal_queue.mark_apply_result(
                proposal,
                {"status": "blocked", "reason": "mixer_not_connected"},
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="mixer_not_connected",
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        apply_manual_console_write = getattr(server, "_apply_manual_console_write", None)
        if apply_manual_console_write is None:
            result = server.operator_proposal_queue.mark_apply_result(
                proposal,
                {"status": "blocked", "reason": "supervised_apply_path_unavailable"},
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="supervised_apply_path_unavailable",
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        normalized_change, safety_error = await validate_change_before_apply(
            requested_change,
            proposal=proposal,
            data=data,
            operator_mode=operator_mode,
            linked=False,
        )
        if safety_error:
            result = server.operator_proposal_queue.mark_apply_result(
                proposal,
                {
                    "status": "blocked",
                    "reason": safety_error["reason"],
                    "requested_change": requested_change,
                    **{key: value for key, value in safety_error.items() if key not in {"reason", "requested_change"}},
                },
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                normalized_change=normalized_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason=safety_error["reason"],
            )
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        apply_result = await apply_manual_console_write(
            websocket,
            channel=normalized_change["channel"],
            value=normalized_change["effective_value"],
            value_type=normalized_change["value_type"],
            source=f"operator_proposal:{proposal.id}",
            reason=data.get("reason") or proposal.reason or proposal.title,
            approved=True,
            approval_id=data.get("approval_id") or proposal.id,
        )
        result = server.operator_proposal_queue.mark_apply_result(proposal, apply_result)
        record_apply_trace(
            proposal=proposal,
            operator_mode=operator_mode,
            requested_change=requested_change,
            normalized_change=normalized_change,
            apply_result=apply_result,
            idempotency_key=idempotency_key,
            status=apply_result.get("status", "unknown"),
            blocked_reason=apply_result.get("reason") if apply_result.get("status") != "applied" else None,
        )
        remember_applied_delta(normalized_change, apply_result)
        await server.send_to_client(websocket, {"type": "operator_proposal_apply_result", **result})

    return {
        "get_dashboard_snapshot": handle_get_dashboard_snapshot,
        "get_channel_inventory": handle_get_channel_inventory,
        "get_channel_processing_snapshot": handle_get_channel_processing_snapshot,
        "get_decision_queue": handle_get_decision_queue,
        "get_decision_traces": handle_get_decision_traces,
        "get_safety_monitor": handle_get_safety_monitor,
        "get_replay_runs": handle_get_replay_runs,
        "start_replay_run": handle_start_replay_run,
        "get_channel_intelligence": handle_get_channel_intelligence,
        "get_connection_topology": handle_get_connection_topology,
        "get_operator_agent_roster": handle_get_operator_agent_roster,
        "get_groups_buses_snapshot": handle_get_groups_buses_snapshot,
        "get_bus_main_dca_snapshot": handle_get_bus_main_dca_snapshot,
        "get_matrix_snapshot": handle_get_matrix_snapshot,
        "get_routing_sends_snapshot": handle_get_routing_sends_snapshot,
        "get_snapshots_scenes_snapshot": handle_get_snapshots_scenes_snapshot,
        "get_user_presets_snapshot": handle_get_user_presets_snapshot,
        "get_memory_knowledge_snapshot": handle_get_memory_knowledge_snapshot,
        "get_training_sessions_snapshot": handle_get_training_sessions_snapshot,
        "get_telemetry_alerts_snapshot": handle_get_telemetry_alerts_snapshot,
        "get_rules_library_snapshot": handle_get_rules_library_snapshot,
        "get_agent_communications_snapshot": handle_get_agent_communications_snapshot,
        "get_processing_pipeline_snapshot": handle_get_processing_pipeline_snapshot,
        "get_main_pipeline_preview": handle_get_main_pipeline_preview,
        "get_operator_mode": handle_get_operator_mode,
        "set_operator_mode": handle_set_operator_mode,
        "run_operator_analysis": handle_run_operator_analysis,
        "import_safe_gain_suggestions": handle_import_safe_gain_suggestions,
        "import_soundcheck_recommendations": handle_import_soundcheck_recommendations,
        "create_operator_proposal": handle_create_operator_proposal,
        "accept_operator_proposal": handle_accept_operator_proposal,
        "dismiss_operator_proposal": handle_dismiss_operator_proposal,
        "apply_operator_proposal": handle_apply_operator_proposal,
    }
