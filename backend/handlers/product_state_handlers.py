"""Read-only operator cockpit state handlers."""

from operator_product_state import (
    build_agent_communications_snapshot,
    build_operator_agent_roster,
    build_processing_pipeline_snapshot,
    build_channel_inventory,
    build_connection_topology,
    build_dashboard_snapshot,
    build_decision_queue,
    build_groups_buses_snapshot,
    build_memory_knowledge_snapshot,
    build_rules_library_snapshot,
    build_training_sessions_snapshot,
)
from operator_analysis import build_operator_analysis_report
from processing_pipeline_preview import build_main_pipeline_preview


VERIFIED_MANUAL_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}


def manual_write_result_verified(result):
    readback_status = result.get("readback_status")
    proof_status = result.get("proof_status")
    if proof_status is not None and proof_status != "verified":
        return False
    return readback_status in VERIFIED_MANUAL_READBACK_STATUSES


def register_handlers(server):
    def proposal_id_or_index(data):
        return data.get("proposal_id") or data.get("id"), data.get("index")

    async def handle_get_dashboard_snapshot(websocket, data):
        await server.send_to_client(websocket, build_dashboard_snapshot(server))

    async def handle_get_channel_inventory(websocket, data):
        total_channels = data.get("total_channels", 24)
        await server.send_to_client(
            websocket,
            build_channel_inventory(server, total_channels=total_channels),
        )

    async def handle_get_decision_queue(websocket, data):
        limit = data.get("limit", 50)
        await server.send_to_client(websocket, build_decision_queue(server, limit=limit))

    async def handle_get_connection_topology(websocket, data):
        await server.send_to_client(websocket, build_connection_topology(server))

    async def handle_get_operator_agent_roster(websocket, data):
        del data
        await server.send_to_client(websocket, build_operator_agent_roster(server))

    async def handle_get_groups_buses_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_groups_buses_snapshot(server))

    async def handle_get_memory_knowledge_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_memory_knowledge_snapshot(server))

    async def handle_get_training_sessions_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_training_sessions_snapshot(server))

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

    async def apply_linked_operator_proposal(websocket, data, proposal, operator_mode, requested_changes):
        allowed_kinds = set(operator_mode["capabilities"].get("allowed_live_write_kinds") or [])
        if not operator_mode["capabilities"]["can_apply_to_console"]:
            result = linked_change_blocked_result(
                proposal,
                "operator_mode_blocks_live_write",
                operator_mode=operator_mode,
                requested_changes=requested_changes,
            )
            await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
            return

        if not bool(data.get("approved", False)):
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
            await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
            return

        apply_manual_console_write = getattr(server, "_apply_manual_console_write", None)
        if apply_manual_console_write is None:
            result = linked_change_blocked_result(proposal, "supervised_apply_path_unavailable")
            await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
            return

        normalized_changes = []
        state = getattr(server.mixer_client, "state", {})
        for raw_change in requested_changes:
            normalized, error = normalize_linked_change(raw_change)
            if error:
                result = linked_change_blocked_result(
                    proposal,
                    error,
                    requested_change=raw_change,
                    requested_changes=requested_changes,
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
                await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
                return
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
                old_value = state.get(normalized["address"]) if isinstance(state, dict) else None
            if old_value is None:
                result = linked_change_blocked_result(
                    proposal,
                    "linked_pre_read_unavailable",
                    channel=normalized["channel"],
                    value_type=normalized["value_type"],
                    address=normalized["address"],
                    pre_read_status=(read_value or {}).get("status"),
                    pre_read_source=(read_value or {}).get("source"),
                )
                await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
                return
            normalized["old_value"] = old_value
            normalized_changes.append(normalized)

        approval_id = data.get("approval_id") or proposal.id
        applied_results = []
        for index, change in enumerate(normalized_changes):
            apply_result = await apply_manual_console_write(
                websocket,
                channel=change["channel"],
                value=change["value"],
                value_type=change["value_type"],
                source=f"operator_proposal_linked:{proposal.id}:{index + 1}",
                reason=data.get("reason") or proposal.reason or proposal.title,
                approved=True,
                approval_id=f"{approval_id}:{index + 1}",
            )
            applied_results.append(apply_result)
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
                await server.send_to_client(websocket, {"type": "operator_proposal_apply_blocked", **result})
                return

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
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                "status": "blocked",
                "success": False,
                "reason": "proposal_not_found",
                "proposal_id": proposal_id,
                "index": index,
            })
            return

        requested_changes = getattr(proposal, "requested_changes", []) or []
        if requested_changes:
            await apply_linked_operator_proposal(websocket, data, proposal, operator_mode, requested_changes)
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
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        if not bool(data.get("approved", False)):
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
            await server.send_to_client(websocket, {
                "type": "operator_proposal_apply_blocked",
                **result,
            })
            return

        apply_result = await apply_manual_console_write(
            websocket,
            channel=requested_change.get("channel"),
            value=requested_change.get("value"),
            value_type=value_type,
            source=f"operator_proposal:{proposal.id}",
            reason=data.get("reason") or proposal.reason or proposal.title,
            approved=True,
            approval_id=data.get("approval_id") or proposal.id,
        )
        result = server.operator_proposal_queue.mark_apply_result(proposal, apply_result)
        await server.send_to_client(websocket, {"type": "operator_proposal_apply_result", **result})

    return {
        "get_dashboard_snapshot": handle_get_dashboard_snapshot,
        "get_channel_inventory": handle_get_channel_inventory,
        "get_decision_queue": handle_get_decision_queue,
        "get_connection_topology": handle_get_connection_topology,
        "get_operator_agent_roster": handle_get_operator_agent_roster,
        "get_groups_buses_snapshot": handle_get_groups_buses_snapshot,
        "get_memory_knowledge_snapshot": handle_get_memory_knowledge_snapshot,
        "get_training_sessions_snapshot": handle_get_training_sessions_snapshot,
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
