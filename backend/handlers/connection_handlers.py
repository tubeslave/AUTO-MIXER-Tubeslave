"""Connection (Wing / dLive / Mixing Station) message handlers."""

import asyncio
import logging
from mixer_discovery import discover_mixers, discover_mixer_auto

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_get_runtime_identity(websocket, data):
        await server.send_to_client(
            websocket,
            server.get_runtime_identity_status(
                request_challenge_nonce=data.get("challenge_nonce")
            ),
        )

    async def handle_connect_wing(websocket, data):
        await server.connect_wing(
            data.get("ip"),
            data.get("send_port"),
            data.get("receive_port"),
            observation_only=data.get("observation_only", data.get("observe_only")),
            record_telemetry=data.get("record_telemetry"),
            telemetry_session_label=data.get("telemetry_session_label"),
            owner_websocket=websocket,
        )

    async def handle_capture_wing_master_route_proof(websocket, data):
        result = await server.capture_wing_master_route_proof(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_arm_m110_attended_control(websocket, data):
        result = await server.arm_m110_attended_control(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_m110_main_output_checkpoint(websocket, data):
        result = await server.verify_m110_main_output_checkpoint(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_run_m110_soundcheck_final_audit(websocket, data):
        result = await server.run_m110_soundcheck_final_audit(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_issue_supervised_capability(websocket, data):
        result = await server.issue_m110_supervised_capability(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_prepare_m110_restore_only_proof(websocket, data):
        result = await server.prepare_m110_restore_only_proof(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_create_m110_maintenance_reset_proposal(websocket, data):
        result = await server.create_m110_maintenance_reset_proposal(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_create_m110_m006_restore_proposal(websocket, data):
        result = await server.create_m110_m006_restore_proposal(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_create_m110_processor_enable_proposal(websocket, data):
        result = await server.create_m110_processor_enable_proposal(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_create_m110_soundcheck_child_proposal(websocket, data):
        result = await server.create_m110_soundcheck_child_proposal(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_advance_m110_soundcheck_replan(websocket, data):
        result = await server.advance_m110_soundcheck_replan(
            websocket=websocket,
            request=data,
        )
        await server.send_to_client(websocket, result)

    async def handle_connect_dlive(websocket, data):
        await server.connect_configured_dlive(owner_websocket=websocket)

    async def handle_connect_mixing_station(websocket, data):
        await server.connect_configured_mixing_station(owner_websocket=websocket)

    async def handle_discover_mixing_station(websocket, data):
        await server.discover_mixing_station()

    async def handle_disconnect(websocket, data):
        await server.disconnect_mixer(requester_websocket=websocket)

    async def handle_scan_mixers(websocket, data):
        """Scan the network for all available mixers (WING + dLive)."""
        if await server.block_packaged_offline_transport(
            "scan_mixers",
            websocket=websocket,
        ):
            return
        full_scan = data.get("full_scan") is True

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: discover_mixers(
                # Full scans derive active local subnets inside the trusted
                # discovery boundary; WebSocket input never selects a target.
                scan_subnet=full_scan, subnet=None, timeout=2.0
            ),
        )

        mixer_list = []
        for m in results:
            mixer_list.append({
                "mixer_type": m.mixer_type,
                "ip": m.ip,
                "port": m.port,
                "name": m.name,
                "model": m.model,
                "tls": m.tls,
                "endpoint": getattr(m, "endpoint", ""),
                "identity_verified": getattr(m, "identity_verified", False),
                "method": m.discovery_method,
                "response_ms": round(m.response_time_ms, 0),
            })

        await server.send_to_client(websocket, {
            "type": "mixer_scan_results",
            "mixers": mixer_list,
            "count": len(mixer_list),
        })
        logger.info(f"Mixer scan complete: found {len(mixer_list)} mixer(s)")

    async def handle_auto_connect(websocket, data):
        """Auto-discover and connect to the best mixer on the network."""
        if await server.block_packaged_offline_transport(
            "auto_connect",
            websocket=websocket,
        ):
            return
        preferred_type = data.get("preferred_type", None)
        # Never let an ingress message choose a network probe destination.
        # A discovered dLive target is independently checked by connect_dlive.
        preferred_ip = None
        full_scan = data.get("full_scan", False)
        midi_base_channel = data.get("midi_base_channel", 11)
        user_profile = data.get("user_profile", 0)

        await server.send_to_client(websocket, {
            "type": "auto_connect_status",
            "status": "scanning",
            "message": "Scanning network for mixers...",
        })

        loop = asyncio.get_event_loop()
        discovered = await loop.run_in_executor(
            None,
            lambda: discover_mixer_auto(
                preferred_type=preferred_type,
                preferred_ip=preferred_ip,
                scan_subnet=full_scan,
                timeout=2.0,
            ),
        )

        if discovered is None:
            await server.send_to_client(websocket, {
                "type": "auto_connect_status",
                "status": "not_found",
                "message": "No mixer found on the network",
            })
            return

        await server.send_to_client(websocket, {
            "type": "auto_connect_status",
            "status": "found",
            "mixer_type": discovered.mixer_type,
            "ip": discovered.ip,
            "port": discovered.port,
            "name": discovered.name,
            "midi_base_channel": midi_base_channel if discovered.mixer_type == "dlive" else None,
            "user_profile": user_profile if discovered.mixer_type == "dlive" else None,
            "message": f"Found {discovered.mixer_type.upper()} @ {discovered.ip}:{discovered.port}",
        })

        if discovered.mixer_type == "dlive":
            await server.connect_dlive(
                ip=discovered.ip,
                port=discovered.port,
                tls=discovered.tls,
                endpoint=getattr(discovered, "endpoint", "surface") or "surface",
                midi_base_channel=midi_base_channel,
                user_profile=user_profile,
                owner_websocket=websocket,
            )
        elif discovered.mixer_type == "wing":
            await server.connect_wing(
                ip=discovered.ip,
                send_port=discovered.port,
                receive_port=discovered.port,
                observation_only=data.get("observation_only", data.get("observe_only")),
                record_telemetry=data.get("record_telemetry"),
                telemetry_session_label=data.get("telemetry_session_label"),
                owner_websocket=websocket,
            )

    return {
        "get_runtime_identity": handle_get_runtime_identity,
        "capture_wing_master_route_proof": (
            handle_capture_wing_master_route_proof
        ),
        "arm_m110_attended_control": handle_arm_m110_attended_control,
        "verify_m110_main_output_checkpoint": (
            handle_verify_m110_main_output_checkpoint
        ),
        "run_m110_soundcheck_final_audit": (
            handle_run_m110_soundcheck_final_audit
        ),
        "issue_supervised_capability": handle_issue_supervised_capability,
        "prepare_m110_restore_only_proof": (
            handle_prepare_m110_restore_only_proof
        ),
        "create_m110_maintenance_reset_proof_proposal": (
            handle_create_m110_maintenance_reset_proposal
        ),
        "create_m110_maintenance_reset_commit_proposal": (
            handle_create_m110_maintenance_reset_proposal
        ),
        "create_m110_m006_restore_proof_proposal": (
            handle_create_m110_m006_restore_proposal
        ),
        "create_m110_m006_restore_commit_proposal": (
            handle_create_m110_m006_restore_proposal
        ),
        "create_m110_processor_enable_proof_proposal": (
            handle_create_m110_processor_enable_proposal
        ),
        "create_m110_processor_enable_commit_proposal": (
            handle_create_m110_processor_enable_proposal
        ),
        "create_m110_soundcheck_child_proof_proposal": (
            handle_create_m110_soundcheck_child_proposal
        ),
        "create_m110_soundcheck_child_commit_proposal": (
            handle_create_m110_soundcheck_child_proposal
        ),
        "advance_m110_soundcheck_replan": (
            handle_advance_m110_soundcheck_replan
        ),
        "connect_wing": handle_connect_wing,
        "connect_dlive": handle_connect_dlive,
        "connect_mixing_station": handle_connect_mixing_station,
        "discover_mixing_station": handle_discover_mixing_station,
        "disconnect_wing": handle_disconnect,
        "disconnect_mixer": handle_disconnect,
        "scan_mixers": handle_scan_mixers,
        "auto_connect": handle_auto_connect,
    }
