"""Auto soundcheck message handlers."""

import asyncio
import copy
import json
import logging

from audio_capture import LIVE_CAPTURE_MAX_CHANNELS
from auto_soundcheck_engine import AutoSoundcheckEngine
from soundcheck_mix_readiness import (
    build_proposal_only_readiness,
    plan_preflight_assessment,
)
from supervised_console_parameter_registry import (
    supervised_console_family_registry,
)

logger = logging.getLogger(__name__)


OPERATOR_PROPOSAL_IMPORT_BROADCAST_RESULT_LIMIT = 200
OPERATOR_PROPOSAL_IMPORT_BROADCAST_MAX_BYTES = 128 * 1024
OPERATOR_PROPOSAL_IMPORT_BROADCAST_TEXT_LIMIT = 2048
OPERATOR_PROPOSAL_IMPORT_BROADCAST_TOP_LEVEL_KEYS = (
    "type",
    "status",
    "success",
    "complete",
    "source",
    "reason",
    "operator_mode",
    "imported_count",
    "created_count",
    "existing_count",
    "blocked_count",
    "generated_at",
    "recommendation_derivation_complete",
    "trigger",
    "trigger_state",
    "soundcheck_readiness",
)


def _bounded_broadcast_value(value, *, depth=0):
    """Return plain JSON data with bounded strings, depth, and fan-out."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= OPERATOR_PROPOSAL_IMPORT_BROADCAST_TEXT_LIMIT:
            return value
        return f"{value[:OPERATOR_PROPOSAL_IMPORT_BROADCAST_TEXT_LIMIT]}...<truncated>"
    if depth >= 4:
        return "<nested_value_omitted>"
    if isinstance(value, dict):
        bounded = {}
        items = list(value.items())
        for raw_key, raw_value in items[:64]:
            key = str(raw_key)[:128]
            bounded[key] = _bounded_broadcast_value(raw_value, depth=depth + 1)
        if len(items) > 64:
            bounded["__omitted_key_count__"] = len(items) - 64
        return bounded
    if isinstance(value, (list, tuple)):
        bounded = [
            _bounded_broadcast_value(item, depth=depth + 1)
            for item in value[:64]
        ]
        if len(value) > 64:
            bounded.append({"__omitted_item_count__": len(value) - 64})
        return bounded
    return _bounded_broadcast_value(str(value), depth=depth)


def _serialized_broadcast_bytes(payload):
    # Match ws_transport.serialize_message: json.dumps defaults to escaped UTF-8.
    return len(json.dumps(payload).encode("utf-8"))


def _counts_only_import_summary(summary, *, result_count):
    payload = {}
    for key in OPERATOR_PROPOSAL_IMPORT_BROADCAST_TOP_LEVEL_KEYS:
        if key == "soundcheck_readiness" or key not in summary:
            continue
        payload[key] = _bounded_broadcast_value(summary.get(key))
    payload.update({
        "results": [],
        "result_count": result_count,
        "results_compacted": True,
        "results_counts_only": True,
        "results_omitted_count": result_count,
        "full_results_endpoint": "get_decision_queue",
        "soundcheck_readiness_endpoint": "get_auto_soundcheck_status",
        "soundcheck_readiness_omitted": isinstance(
            summary.get("soundcheck_readiness"),
            dict,
        ),
    })
    return payload


def compact_operator_proposal_import_summary(summary):
    """Hard-bound automatic import events while full proposals stay in the queue."""

    source = dict(summary or {})
    results = source.get("results")
    if not isinstance(results, list):
        results = []

    payload = {
        key: _bounded_broadcast_value(source.get(key))
        for key in OPERATOR_PROPOSAL_IMPORT_BROADCAST_TOP_LEVEL_KEYS
        if key in source
    }

    compact_results = []
    for result in results[:OPERATOR_PROPOSAL_IMPORT_BROADCAST_RESULT_LIMIT]:
        if not isinstance(result, dict):
            continue
        compact_result = {
            key: _bounded_broadcast_value(result.get(key))
            for key in (
                "type",
                "status",
                "success",
                "reason",
                "proposal_id",
            )
            if result.get(key) is not None
        }
        proposal = result.get("proposal")
        if isinstance(proposal, dict):
            compact_proposal = {
                key: _bounded_broadcast_value(proposal.get(key))
                for key in (
                    "id",
                    "title",
                    "target",
                    "kind",
                    "severity",
                    "confidence",
                    "status",
                    "source",
                    "can_apply",
                    "requires_approval",
                    "apply_blockers",
                )
                if proposal.get(key) is not None
            }
            compact_result["proposal"] = compact_proposal
            if not compact_result.get("proposal_id") and compact_proposal.get("id"):
                compact_result["proposal_id"] = compact_proposal["id"]
        compact_results.append(compact_result)

    payload["results"] = compact_results
    payload["result_count"] = len(results)
    payload["results_compacted"] = True
    payload["results_omitted_count"] = max(0, len(results) - len(compact_results))
    payload["full_results_endpoint"] = "get_decision_queue"
    payload["soundcheck_readiness_endpoint"] = "get_auto_soundcheck_status"
    if _serialized_broadcast_bytes(payload) > OPERATOR_PROPOSAL_IMPORT_BROADCAST_MAX_BYTES:
        payload = _counts_only_import_summary(source, result_count=len(results))
    if _serialized_broadcast_bytes(payload) > OPERATOR_PROPOSAL_IMPORT_BROADCAST_MAX_BYTES:
        # Defensive final form: only fixed-size fields and numeric counts survive.
        payload = {
            "type": "operator_proposals_imported",
            "status": "compacted",
            "success": bool(source.get("success")),
            "results": [],
            "result_count": len(results),
            "results_compacted": True,
            "results_counts_only": True,
            "results_omitted_count": len(results),
            "full_results_endpoint": "get_decision_queue",
            "soundcheck_readiness_endpoint": "get_auto_soundcheck_status",
            "soundcheck_readiness_omitted": isinstance(
                source.get("soundcheck_readiness"),
                dict,
            ),
        }
    return payload


def register_handlers(server):
    def _engine_cleanup_incomplete():
        engine = server.auto_soundcheck_engine
        state = getattr(getattr(engine, "state", None), "value", None)
        return bool(
            engine is not None
            and state in ("error", "stopped")
            and getattr(engine, "_stop_cleanup_complete", False) is not True
        )

    def _engine_is_active():
        engine = server.auto_soundcheck_engine
        state = getattr(getattr(engine, "state", None), "value", None)
        engine_thread = getattr(engine, "_engine_thread", None)
        monitor_thread = getattr(engine, "_monitor_thread", None)
        cleanup_complete = getattr(engine, "_stop_cleanup_complete", False) is True
        thread_alive = any(
            thread is not None
            and callable(getattr(thread, "is_alive", None))
            and thread.is_alive() is True
            for thread in (engine_thread, monitor_thread)
        )
        return bool(
            engine
            and (
                not cleanup_complete
                or thread_alive
                or state not in ("stopped", "error")
            )
        )

    def _channel_selection(data):
        autofoh = server.config.get("autofoh", {}) if isinstance(server.config, dict) else {}
        measurement_taps = autofoh.get("measurement_taps", {}) or {}
        raw_map = measurement_taps.get("raw_channels", {}) or {}
        try:
            source_channel_count = int(raw_map.get("channels", 16))
        except (TypeError, ValueError):
            source_channel_count = 16
        source_channel_count = max(1, min(source_channel_count, 48))

        capture_indices = [48]
        for key in ("raw_channels", "postfader_channels"):
            mapping = measurement_taps.get(key, {}) or {}
            try:
                first = int(mapping.get("first_usb", 1))
                count = int(mapping.get("channels", 0))
            except (TypeError, ValueError):
                continue
            if first > 0 and count > 0:
                capture_indices.append(first + count - 1)
        for mapping in (measurement_taps.get("stem_buses", {}) or {}).values():
            if isinstance(mapping, dict):
                capture_indices.extend(
                    int(value)
                    for value in mapping.get("usb", [])
                    if isinstance(value, int) and not isinstance(value, bool)
                )
        capture_indices.extend(
            int(value)
            for value in measurement_taps.get("main_prefader", []) or []
            if isinstance(value, int) and not isinstance(value, bool)
        )
        master_mix = autofoh.get("master_mix", {}) or {}
        capture_indices.extend(
            int(value)
            for value in master_mix.get("capture_channels", [47, 48]) or []
            if isinstance(value, int) and not isinstance(value, bool)
        )
        num_channels = max(capture_indices)

        raw_channels = data.get("channels", [])
        if isinstance(raw_channels, int):
            if isinstance(raw_channels, bool) or raw_channels < 1:
                raise ValueError("AutoSoundcheck source channel count must be positive")
            selected_channels = list(
                range(1, min(raw_channels, source_channel_count) + 1)
            )
        else:
            selected_channels = []
            for ch in raw_channels:
                try:
                    channel = int(ch)
                except (TypeError, ValueError):
                    continue
                if channel > 0:
                    selected_channels.append(channel)
            selected_channels = sorted(set(selected_channels))
            if not selected_channels:
                selected_channels = list(range(1, source_channel_count + 1))
        invalid_sources = [
            channel
            for channel in selected_channels
            if channel > source_channel_count
        ]
        if invalid_sources:
            raise ValueError(
                "AutoSoundcheck selected channels are outside the configured "
                f"raw source map 1..{source_channel_count}: {invalid_sources}"
            )
        if num_channels > LIVE_CAPTURE_MAX_CHANNELS:
            raise ValueError(
                f"AutoSoundcheck channels must be within 1..{LIVE_CAPTURE_MAX_CHANNELS}"
            )
        return selected_channels, num_channels

    def _resolve_audio_device(data, audio_config):
        """Resolve the operator's device choice to a device name.

        The frontend sends device_id as a PyAudio device index; the engine
        matches devices by name, so translate the index to the actual name.
        """
        explicit = data.get("audio_device")
        if explicit:
            return explicit
        device_id = data.get("device_id")
        if device_id is not None:
            try:
                from audio_devices import get_device_by_id
                device = get_device_by_id(str(device_id))
                if device and device.get("name"):
                    return device["name"]
            except Exception as exc:
                logger.warning("Could not resolve device_id %r to a name: %s", device_id, exc)
            return str(device_id)
        return audio_config.get("device_name", "soundgrid")

    def _channel_mapping(data):
        """Parse the console-channel → capture-channel mapping from the UI."""
        mapping = {}
        raw = data.get("channel_mapping") or {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    console_ch = int(key)
                    capture_ch = int(value)
                except (TypeError, ValueError):
                    continue
                if console_ch > 0 and capture_ch > 0:
                    mapping[console_ch] = capture_ch
        return mapping

    def _schedule(loop, coro, label):
        def _log_failure(task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.warning("%s failed: %s", label, exc)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            task = loop.create_task(coro)
            task.add_done_callback(_log_failure)
            return

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        future.add_done_callback(_log_failure)

    async def _start_engine(websocket, data, *, soundcheck_events: bool):
        """Start the new AutoSoundcheckEngine as the canonical soundcheck path."""
        m110_analysis = bool(
            isinstance(data, dict)
            and data.get("type") == "start_m110_soundcheck_analysis"
        )
        m110_terminal_state = {"sent": False}

        async def send_m110_terminal(payload):
            if not m110_analysis or m110_terminal_state["sent"]:
                return
            m110_terminal_state["sent"] = True
            await server.send_to_client(
                websocket,
                {
                    "type": "m110_soundcheck_analysis_result",
                    "live_authority_granted": False,
                    "readiness_closure_allowed": False,
                    **dict(payload),
                },
            )
        blocked_response_type = (
            "m110_soundcheck_analysis_result"
            if m110_analysis
            else "auto_soundcheck_status"
            if soundcheck_events
            else "auto_engine_status"
        )
        shared_capture = None
        if m110_analysis:
            control, control_reason = server._verified_m110_attended_control(
                websocket
            )
            route_view = (
                await server._verified_master_capture_route_proof_with_fresh_hardware()
            )
            route = server._master_capture_route_authority
            shared_capture = getattr(server, "audio_capture", None)
            route_ready = bool(
                route_view.get("verified") is True
                and route is not None
                and getattr(route, "_owner_ref", None) is websocket
            )
            capture_ready = bool(
                server._audio_capture_preview_owner is websocket
                and shared_capture is not None
                and getattr(shared_capture, "running", False) is True
            )
            reset_ready = bool(
                control is not None
                and route_ready
                and server._m110_full_reset_is_current(
                    websocket=websocket,
                    control=control,
                    route=route,
                )
            )
            unmuted_checkpoint_ready = bool(
                reset_ready
                and server._m110_main_output_checkpoint_is_current(
                    "soundcheck_unmuted",
                    websocket=websocket,
                    control=control,
                    route=route,
                    maximum_age_sec=5.0,
                )
            )
            m110_ready = bool(
                control is not None
                and route_ready
                and capture_ready
                and reset_ready
                and unmuted_checkpoint_ready
            )
            if not m110_ready:
                if control is None:
                    blocked_reason = control_reason
                elif not route_ready:
                    blocked_reason = (
                        route_view.get("reason")
                        or "m110_route_authority_not_current"
                    )
                elif not capture_ready:
                    blocked_reason = "m110_shared_capture_not_current"
                elif not reset_ready:
                    blocked_reason = "m110_committed_reset_required"
                else:
                    blocked_reason = (
                        "m110_soundcheck_unmuted_checkpoint_required"
                    )
                await server.send_to_client(
                    websocket,
                    {
                        "type": blocked_response_type,
                        "status": "blocked",
                        "success": False,
                        "reason": blocked_reason,
                        "live_authority_granted": False,
                        "readiness_closure_allowed": False,
                    },
                )
                return
        if getattr(server, "_packaged_offline_lock", False):
            await server.send_to_client(websocket, {
                "type": blocked_response_type,
                "status": "blocked",
                "is_running": False,
                "running": False,
                "observe_only": True,
                "proposal_only": True,
                "reason": "packaged_offline_lock_active",
                "packaged_offline_lock": True,
            })
            return
        if _engine_cleanup_incomplete():
            await server.send_to_client(websocket, {
                "type": blocked_response_type,
                "status": "blocked",
                "is_running": False,
                "running": False,
                "cleanup_complete": False,
                "error": "Auto soundcheck cleanup is incomplete",
                "reason": "auto_soundcheck_cleanup_incomplete",
            })
            return
        if _engine_is_active():
            await server.send_to_client(websocket, {
                "type": blocked_response_type,
                "is_running": True,
                "running": True,
                "error": "Auto soundcheck engine already running",
            })
            return

        mixer_config = server.config.get("mixer", {})
        audio_config = server.config.get("audio", {})
        try:
            selected_channels, num_channels = _channel_selection(
                {} if m110_analysis else data
            )
            if m110_analysis:
                capture_channels = int(
                    getattr(shared_capture, "num_channels", 0) or 0
                )
                if capture_channels < 48:
                    raise ValueError(
                        "M110 shared capture must expose channels 1..48"
                    )
                selected_channels = list(range(1, 17))
                num_channels = capture_channels
        except (TypeError, ValueError) as exc:
            await server.send_to_client(websocket, {
                "type": blocked_response_type,
                "status": "blocked",
                "is_running": False,
                "running": False,
                "observe_only": True,
                "proposal_only": True,
                "error": str(exc),
                "reason": "invalid_audio_capture_dimensions",
            })
            return

        mixer_type = (
            "wing"
            if m110_analysis
            else data.get("mixer_type", mixer_config.get("type", "dlive"))
        )
        mixer_ip = data.get("mixer_ip", mixer_config.get("ip", "192.168.3.70"))
        mixer_port = data.get(
            "mixer_port",
            mixer_config.get("port", 51328 if mixer_type == "dlive" else 2223)
        )
        audio_device = (
            str(server._master_capture_route_authority.capture_device)
            if m110_analysis
            else _resolve_audio_device(data, audio_config)
        )
        channel_mapping = _channel_mapping(data)
        mode = data.get("mode", "full_band")
        # The canonical soundcheck path measures and proposes. All physical writes
        # must pass through the supervised manual gate with explicit approval.
        observe_only = True
        loop = asyncio.get_running_loop()

        def on_state(state, msg):
            server.auto_soundcheck_running = state not in ("stopped", "error")
            if soundcheck_events:
                _schedule(loop, server.send_to_client(websocket, {
                    "type": "auto_soundcheck_status",
                    "is_running": server.auto_soundcheck_running,
                    "running": server.auto_soundcheck_running,
                    "observe_only": server.auto_soundcheck_observe_only,
                    "current_step": state,
                    "step_progress": 0,
                    "progress": 0,
                    "step_time_remaining": 0,
                    "message": msg or state,
                }), "auto soundcheck status send")

            _schedule(loop, server.broadcast({
                "type": "auto_engine_state",
                "state": state,
                "message": msg,
            }), "auto engine state broadcast")

        def on_channel(ch, ch_data):
            if hasattr(server, "update_mixing_agent_channel"):
                server.update_mixing_agent_channel(ch, ch_data)

            if soundcheck_events:
                _schedule(loop, server.send_to_client(websocket, {
                    "type": "auto_soundcheck_channel_update",
                    "channel": ch,
                    "data": ch_data,
                }), "auto soundcheck channel send")

            _schedule(loop, server.broadcast({
                "type": "auto_engine_channel",
                "channel": ch,
                "data": ch_data,
            }), "auto engine channel broadcast")

        def on_observation(payload):
            if isinstance(payload, dict) and payload.get("master_true_peak_dbtp") is not None:
                server.record_master_true_peak_evidence(payload.get("master_true_peak_dbtp"))
            payload = {
                "type": "auto_soundcheck_observation",
                **payload,
            }
            _schedule(loop, server.send_to_client(websocket, payload), "auto soundcheck observation send")

        recommendation_import = {"done": False}

        async def _import_soundcheck_recommendations(trigger_state):
            if recommendation_import["done"]:
                return
            recommendation_import["done"] = True
            if not hasattr(server, "import_soundcheck_recommendations_to_operator_queue"):
                if m110_analysis:
                    await send_m110_terminal({
                        "status": "blocked",
                        "success": False,
                        "reason": "soundcheck_recommendation_import_unavailable",
                    })
                return
            if getattr(server, "auto_soundcheck_engine", None) is not engine:
                await server.broadcast({
                    "type": "operator_proposals_imported",
                    "status": "blocked",
                    "success": False,
                    "reason": "soundcheck_engine_superseded",
                    "trigger": "auto_soundcheck_state",
                    "trigger_state": trigger_state,
                })
                if m110_analysis:
                    await send_m110_terminal({
                        "status": "blocked",
                        "success": False,
                        "reason": "soundcheck_engine_superseded",
                    })
                return
            bundle = None
            if hasattr(engine, "get_soundcheck_recommendation_bundle"):
                bundle = engine.get_soundcheck_recommendation_bundle()
            manifest = (
                bundle.get("supervised_transaction_plans")
                if isinstance(bundle, dict)
                else None
            )
            preflight_manifest = getattr(
                server,
                "preflight_soundcheck_manifest",
                None,
            )
            if isinstance(manifest, dict) and callable(preflight_manifest):
                assessment = await preflight_manifest(manifest)
                if getattr(server, "auto_soundcheck_engine", None) is not engine:
                    await server.broadcast({
                        "type": "operator_proposals_imported",
                        "status": "blocked",
                        "success": False,
                        "reason": "soundcheck_engine_superseded",
                        "trigger": "auto_soundcheck_state",
                        "trigger_state": trigger_state,
                    })
                    return
                assessments_by_operation = {
                    str(item.get("operation_id") or ""): dict(item)
                    for item in assessment.get("assessments") or []
                    if isinstance(item, dict)
                }
                manifest = copy.deepcopy(manifest)
                manifest["plans"] = [
                    {
                        **dict(plan),
                        "preflight_assessment": plan_preflight_assessment(
                            plan,
                            assessments_by_operation,
                        ),
                    }
                    for plan in manifest.get("plans") or []
                    if isinstance(plan, dict)
                ]
                bundle = {
                    **dict(bundle),
                    "supervised_transaction_plans": manifest,
                    "soundcheck_preflight": assessment,
                }
            summary = server.import_soundcheck_recommendations_to_operator_queue(
                bundle,
                trusted_engine_bundle=True,
            )
            finalized = None
            finalize_without_apply = getattr(
                server,
                "finalize_soundcheck_readiness_if_no_apply",
                None,
            )
            if isinstance(manifest, dict) and callable(finalize_without_apply):
                finalized = await finalize_without_apply(
                    str(manifest.get("run_id") or "")
                )
            if isinstance(finalized, dict):
                summary["soundcheck_readiness"] = dict(finalized)
            summary["recommendation_derivation_complete"] = bool(
                getattr(engine, "_derivation_complete", False)
            )
            summary.update({
                "trigger": "auto_soundcheck_state",
                "trigger_state": trigger_state,
            })
            await server.broadcast(compact_operator_proposal_import_summary(summary))
            if m110_analysis:
                registered = server._m110_current_soundcheck_manifest()
                analysis_complete = bool(
                    isinstance(manifest, dict)
                    and isinstance(registered, dict)
                    and registered.get("run_id") == manifest.get("run_id")
                    and getattr(engine, "_derivation_complete", False) is True
                )
                await send_m110_terminal({
                    "status": "complete" if analysis_complete else "blocked",
                    "success": analysis_complete,
                    "reason": (
                        "m110_soundcheck_manifest_registered"
                        if analysis_complete
                        else "m110_soundcheck_manifest_registration_failed"
                    ),
                    "run_id": (
                        manifest.get("run_id")
                        if isinstance(manifest, dict)
                        else None
                    ),
                    "plan_revision": (
                        manifest.get("plan_revision")
                        if isinstance(manifest, dict)
                        else None
                    ),
                    "imported_count": summary.get("imported_count"),
                    "soundcheck_readiness": summary.get(
                        "soundcheck_readiness"
                    ),
                    "shared_capture_retained": bool(
                        engine.audio_capture is shared_capture
                        and getattr(shared_capture, "running", False)
                    ),
                })

        async def import_soundcheck_recommendations_safely(trigger_state):
            try:
                await _import_soundcheck_recommendations(trigger_state)
            except BaseException as exc:
                logger.exception("M110 soundcheck recommendation import failed")
                await send_m110_terminal({
                    "status": "blocked",
                    "success": False,
                    "reason": "m110_soundcheck_recommendation_import_failed",
                    "exception_type": type(exc).__name__,
                })

        conflict = None
        async with server._audio_capture_lock:
            measurement_task = getattr(server, "_system_measurement_task", None)
            if measurement_task is not None and not measurement_task.done():
                conflict = "audio_capture_in_use_by_system_measurement"
            elif (
                getattr(server, "audio_capture", None) is not None
                and not m110_analysis
            ):
                conflict = "audio_capture_in_use_by_preview"
            elif _engine_is_active():
                conflict = "auto_soundcheck_already_running"
            else:
                # Reserve the capture authority before releasing the lock. The
                # engine owns its private capture; server preview/measurement
                # capture is never borrowed across lifecycle boundaries.
                server.auto_soundcheck_running = True
                server.auto_soundcheck_observe_only = observe_only

        if conflict is not None:
            await server.send_to_client(websocket, {
                "type": blocked_response_type,
                "status": "blocked",
                "is_running": bool(server.auto_soundcheck_running),
                "running": bool(server.auto_soundcheck_running),
                "observe_only": True,
                "proposal_only": True,
                "error": conflict,
                "reason": conflict,
            })
            return

        engine = None
        try:
            engine = AutoSoundcheckEngine(
                mixer_type=mixer_type,
                mixer_ip=mixer_ip,
                mixer_port=mixer_port,
                audio_device_name=audio_device,
                num_channels=num_channels,
                selected_channels=selected_channels,
                channel_mapping=channel_mapping,
                mode=mode,
                observe_only=observe_only,
                auto_apply=False,
                on_state_change=on_state,
                on_channel_update=on_channel,
                on_observation=on_observation,
                mixer_client=(
                    server.mixer_client
                    if getattr(getattr(server, "mixer_client", None), "is_connected", False)
                    else None
                ),
                audio_capture=shared_capture if m110_analysis else None,
            )
            server.auto_soundcheck_engine = engine
        except BaseException:
            async with server._audio_capture_lock:
                if server.auto_soundcheck_engine is engine:
                    server.auto_soundcheck_engine = None
                server.auto_soundcheck_running = False
                server.auto_soundcheck_observe_only = False
            if m110_analysis:
                await send_m110_terminal({
                    "status": "blocked",
                    "success": False,
                    "reason": "m110_soundcheck_engine_construction_failed",
                })
                return
            raise

        original_on_state = on_state

        def on_state_with_recommendation_import(state, msg):
            original_on_state(state, msg)
            if state in {"running", "stopped"}:
                _schedule(
                    loop,
                    import_soundcheck_recommendations_safely(state),
                    "auto soundcheck recommendation import",
                )
            elif state == "error" and m110_analysis:
                _schedule(
                    loop,
                    send_m110_terminal({
                        "status": "blocked",
                        "success": False,
                        "reason": "m110_soundcheck_engine_error",
                    }),
                    "M110 soundcheck error terminal",
                )

        engine.on_state_change = on_state_with_recommendation_import
        try:
            engine.start_async()
        except BaseException:
            async with server._audio_capture_lock:
                if server.auto_soundcheck_engine is engine:
                    server.auto_soundcheck_engine = None
                server.auto_soundcheck_running = False
                server.auto_soundcheck_observe_only = False
            if m110_analysis:
                await send_m110_terminal({
                    "status": "blocked",
                    "success": False,
                    "reason": "m110_soundcheck_engine_start_failed",
                })
                return
            raise

        live_action_config = getattr(engine, "correction_executor_config", None)
        await server.send_to_client(websocket, {
            "type": (
                "m110_soundcheck_analysis_status"
                if m110_analysis
                else "auto_soundcheck_status"
                if soundcheck_events
                else "auto_engine_status"
            ),
            "status": "started",
            "is_running": True,
            "running": True,
            "observe_only": observe_only,
            "proposal_only": True,
            "auto_apply": False,
            "mode": mode,
            "mixer_type": mixer_type,
            "mixer_ip": mixer_ip,
            "selected_channels": selected_channels,
            "live_action_profile": getattr(live_action_config, "profile_id", ""),
            "live_action_executor_enabled": bool(
                getattr(live_action_config, "enabled", False)
            ),
            "message": "AutoSoundcheckEngine started in proposal-only mode",
        })

    async def handle_start_auto_soundcheck(websocket, data):
        await _start_engine(websocket, data, soundcheck_events=True)

    async def handle_start_m110_soundcheck_analysis(websocket, data):
        await _start_engine(websocket, data, soundcheck_events=True)

    async def handle_stop_m110_soundcheck_analysis(websocket, data):
        if getattr(server, "_audio_capture_preview_owner", None) is not websocket:
            await server.send_to_client(websocket, {
                "type": "m110_soundcheck_analysis_stop_result",
                "status": "blocked",
                "success": False,
                "reason": "m110_soundcheck_analysis_owner_required",
            })
            return
        await server._stop_auto_soundcheck_engine_async(observe_only=False)
        await server.send_to_client(websocket, {
            "type": "m110_soundcheck_analysis_stop_result",
            "status": "stopped",
            "success": True,
            "shared_capture_retained": bool(
                getattr(getattr(server, "audio_capture", None), "running", False)
            ),
        })

    async def handle_start_per_channel_soundcheck(websocket, data):
        """Step-through soundcheck: one channel at a time, operator paced."""
        data = dict(data)
        data["mode"] = "per_channel"
        await _start_engine(websocket, data, soundcheck_events=True)

    async def _handle_channel_command(websocket, command):
        engine = server.auto_soundcheck_engine
        accepted = bool(engine) and engine.per_channel_command(command)
        payload = {
            "type": "soundcheck_channel_command_result",
            "command": command,
            "accepted": accepted,
        }
        if engine:
            status = engine.get_status()
            payload["current_channel"] = status.get("current_channel")
            payload["channel_index"] = status.get("channel_index")
            payload["channel_total"] = status.get("channel_total")
        else:
            payload["error"] = "Auto soundcheck engine is not running"
        await server.send_to_client(websocket, payload)

    async def handle_soundcheck_next_channel(websocket, data):
        await _handle_channel_command(websocket, "next")

    async def handle_soundcheck_skip_channel(websocket, data):
        await _handle_channel_command(websocket, "skip")

    async def handle_soundcheck_repeat_channel(websocket, data):
        await _handle_channel_command(websocket, "repeat")

    async def handle_stop_auto_soundcheck(websocket, data):
        await server._stop_auto_soundcheck_engine_async(observe_only=False)
        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status",
            "is_running": False,
            "running": False,
            "observe_only": False,
            "message": "Auto soundcheck stopped"
        })

    async def handle_get_auto_soundcheck_status(websocket, data):
        refresh_readiness = getattr(
            server,
            "refresh_soundcheck_readiness_truth",
            None,
        )
        if callable(refresh_readiness):
            refresh_readiness()
        engine = server.auto_soundcheck_engine
        engine_status = (
            await asyncio.to_thread(engine.get_status)
            if engine is not None
            else {}
        )
        if isinstance(engine_status, dict) and isinstance(
            engine_status.get("soundcheck_readiness"),
            dict,
        ):
            engine_readiness = dict(engine_status["soundcheck_readiness"])
            retained_readiness = getattr(server, "last_soundcheck_readiness", {})
            retained_is_aggregate = bool(
                isinstance(retained_readiness, dict)
                and retained_readiness.get("run_id") == engine_readiness.get("run_id")
                and retained_readiness.get("plan_revision")
                == engine_readiness.get("plan_revision")
                and retained_readiness.get("plan_fingerprint")
                == engine_readiness.get("plan_fingerprint")
                and retained_readiness.get("proposal_only") is False
            )
            if retained_is_aggregate:
                engine_status = dict(engine_status)
                engine_status["soundcheck_readiness"] = dict(retained_readiness)
            else:
                server.last_soundcheck_readiness = engine_readiness
        # The full AutoFOH session report can contain analyzer evidence measured
        # in hundreds of megabytes. Status polling is a bounded control-plane
        # surface; the report remains available through its dedicated artifact.
        if isinstance(engine_status, dict) and engine_status.get("autofoh_session_report"):
            engine_status = dict(engine_status)
            engine_status["autofoh_session_report_available"] = True
            engine_status.pop("autofoh_session_report", None)
        if not engine_status:
            family_registry = supervised_console_family_registry()
            retained_readiness = getattr(server, "last_soundcheck_readiness", None)
            if not isinstance(retained_readiness, dict) or not retained_readiness:
                retained_readiness = build_proposal_only_readiness(
                    {
                        "run_id": "idle",
                        "plan_revision": 1,
                        "plan_fingerprint": "",
                        "plans": [],
                        "unsupported_actions": [],
                    }
                )
            engine_status = {
                "state": "idle",
                "mixer_connected": bool(
                    getattr(server, "mixer_client", None)
                    and getattr(server.mixer_client, "is_connected", False)
                ),
                "audio_running": bool(
                    getattr(server, "audio_capture", None)
                    and getattr(server.audio_capture, "running", False)
                ),
                "supervised_transaction_plan_count": 0,
                "supervised_transaction_operation_count": 0,
                "soundcheck_readiness": retained_readiness,
                "supervised_console_family_registry": family_registry,
                "supervised_console_live_allowed_families": sorted(
                    family_id
                    for family_id, contract in family_registry.items()
                    if contract.get("normal_supervised_apply_allowed") is True
                ),
            }
        readiness_for_replan = engine_status.get("soundcheck_readiness") or {}
        replan_run_id = str(readiness_for_replan.get("run_id") or "")
        public_replan = getattr(server, "_public_soundcheck_replan", None)
        if callable(public_replan):
            engine_status = dict(engine_status)
            engine_status["soundcheck_replan"] = public_replan(
                getattr(server, "_soundcheck_replan_drafts", {}).get(replan_run_id)
            )
        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status",
            "is_running": server.auto_soundcheck_running,
            "running": server.auto_soundcheck_running,
            "observe_only": server.auto_soundcheck_observe_only,
            "current_step": engine_status.get("state"),
            "step_progress": 0,
            "progress": 0,
            "step_time_remaining": 0,
            "message": "Idle" if not server.auto_soundcheck_running else "Running",
            **engine_status,
        })

    async def handle_accept_soundcheck_replan(websocket, data):
        accept_replan = getattr(server, "accept_soundcheck_replan_draft", None)
        if not callable(accept_replan):
            result = {
                "type": "soundcheck_replan_result",
                "status": "blocked",
                "success": False,
                "reason": "soundcheck_replan_path_unavailable",
            }
        else:
            result = await accept_replan(
                run_id=data.get("run_id"),
                draft_id=data.get("draft_id"),
                approved=data.get("approved") is True,
                approval_id=data.get("approval_id"),
            )
        await server.send_to_client(websocket, result)

    async def handle_reanalyze_soundcheck_stage(websocket, data):
        reanalyze = getattr(server, "reanalyze_soundcheck_stage", None)
        if not callable(reanalyze):
            result = {
                "type": "soundcheck_stage_reanalysis_result",
                "status": "blocked",
                "success": False,
                "reason": "soundcheck_reanalysis_path_unavailable",
                "live_authority_changed": False,
            }
        else:
            result = await reanalyze(
                run_id=data.get("run_id"),
                approved=data.get("approved") is True,
                approval_id=data.get("approval_id"),
            )
        await server.send_to_client(websocket, result)

    async def handle_finalize_soundcheck_readiness(websocket, data):
        run_id = str(data.get("run_id") or "").strip()
        reconcile = getattr(server, "reconcile_soundcheck_readiness", None)
        if not run_id or not callable(reconcile):
            await server.send_to_client(
                websocket,
                {
                    "type": "soundcheck_finalize_result",
                    "status": "blocked",
                    "success": False,
                    "reason": (
                        "soundcheck_run_id_required"
                        if not run_id
                        else "soundcheck_reconciliation_path_unavailable"
                    ),
                    "run_id": run_id or None,
                },
            )
            return
        readiness = await reconcile(run_id)
        if not isinstance(readiness, dict):
            await server.send_to_client(
                websocket,
                {
                    "type": "soundcheck_finalize_result",
                    "status": "blocked",
                    "success": False,
                    "reason": "soundcheck_run_not_found",
                    "run_id": run_id,
                },
            )
            return
        await server.send_to_client(
            websocket,
            {
                "type": "soundcheck_finalize_result",
                "status": "ok",
                "success": True,
                "run_id": run_id,
                "live_write": False,
                "soundcheck_readiness": readiness,
            },
        )

    async def handle_start_auto_engine(websocket, data):
        """Start the headless auto-soundcheck engine."""
        await _start_engine(websocket, data, soundcheck_events=False)

    async def handle_stop_auto_engine(websocket, data):
        """Stop the headless auto-soundcheck engine."""
        await server._stop_auto_soundcheck_engine_async(observe_only=False)
        await server.send_to_client(websocket, {
            "type": "auto_engine_status",
            "status": "stopped",
            "is_running": False,
            "running": False,
        })

    async def handle_get_auto_engine_status(websocket, data):
        """Get auto-soundcheck engine status."""
        if server.auto_soundcheck_engine:
            status = await asyncio.to_thread(
                server.auto_soundcheck_engine.get_status
            )
        else:
            status = {"state": "idle", "mixer_connected": False, "audio_running": False}
        await server.send_to_client(websocket, {
            "type": "auto_engine_status",
            "is_running": server.auto_soundcheck_running,
            "running": server.auto_soundcheck_running,
            **status
        })

    async def handle_update_muq_stem_scores(websocket, data):
        """Accept a batch of per-stem MuQ scores for EWMA drift visualization."""

        engine = server.auto_soundcheck_engine
        if engine is None or not hasattr(engine, "update_muq_stem_score_batch"):
            await server.send_to_client(websocket, {
                "type": "muq_stem_drift",
                "status": "unavailable",
                "enabled": False,
                "stems": {},
                "error": "Auto soundcheck engine is not running",
            })
            return

        stem_scores = (
            data.get("stem_scores")
            or data.get("scores")
            or data.get("stems")
            or {}
        )
        if not isinstance(stem_scores, dict):
            await server.send_to_client(websocket, {
                "type": "muq_stem_drift",
                "status": "error",
                "enabled": False,
                "stems": {},
                "error": "stem_scores must be an object",
            })
            return

        result = engine.update_muq_stem_score_batch(
            stem_scores,
            data.get("dt"),
            params_by_stem=data.get("params_by_stem") or data.get("params") or {},
        )
        payload = {
            "type": "muq_stem_drift",
            "status": "updated",
            **result,
        }
        await server.send_to_client(websocket, payload)
        await server.broadcast(payload)

    return {
        "start_auto_soundcheck": handle_start_auto_soundcheck,
        "start_m110_soundcheck_analysis": (
            handle_start_m110_soundcheck_analysis
        ),
        "stop_m110_soundcheck_analysis": (
            handle_stop_m110_soundcheck_analysis
        ),
        "start_per_channel_soundcheck": handle_start_per_channel_soundcheck,
        "soundcheck_next_channel": handle_soundcheck_next_channel,
        "soundcheck_skip_channel": handle_soundcheck_skip_channel,
        "soundcheck_repeat_channel": handle_soundcheck_repeat_channel,
        "stop_auto_soundcheck": handle_stop_auto_soundcheck,
        "get_auto_soundcheck_status": handle_get_auto_soundcheck_status,
        "accept_soundcheck_replan": handle_accept_soundcheck_replan,
        "reanalyze_soundcheck_stage": handle_reanalyze_soundcheck_stage,
        "finalize_soundcheck_readiness": handle_finalize_soundcheck_readiness,
        "start_auto_engine": handle_start_auto_engine,
        "stop_auto_engine": handle_stop_auto_engine,
        "get_auto_engine_status": handle_get_auto_engine_status,
        "update_muq_stem_scores": handle_update_muq_stem_scores,
        "muq_stem_scores": handle_update_muq_stem_scores,
    }
