"""Read-only operator cockpit state handlers."""

from operator_product_state import (
    build_channel_inventory,
    build_connection_topology,
    build_dashboard_snapshot,
    build_decision_queue,
)
from operator_analysis import build_operator_analysis_report


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
        "run_operator_analysis": handle_run_operator_analysis,
        "import_safe_gain_suggestions": handle_import_safe_gain_suggestions,
        "import_soundcheck_recommendations": handle_import_soundcheck_recommendations,
        "create_operator_proposal": handle_create_operator_proposal,
        "accept_operator_proposal": handle_accept_operator_proposal,
        "dismiss_operator_proposal": handle_dismiss_operator_proposal,
        "apply_operator_proposal": handle_apply_operator_proposal,
    }
