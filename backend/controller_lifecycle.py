"""
Controller lifecycle management: cleanup and initialization.

Extracted from server.py to reduce server size and separate concerns.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def cleanup_all_controllers(server: Any) -> None:
    """
    Stop and clear all controllers, disconnect mixer.

    Modifies server attributes in place. Call before shutdown or on error.
    """
    logger.info("Cleaning up all controllers...")

    # Stop gain staging
    if server.gain_staging:
        try:
            server.gain_staging.stop()
            logger.info("Gain staging stopped")
        except Exception as e:
            logger.error(f"Error stopping gain staging: {e}")
        server.gain_staging = None

    # Stop voice control
    if server.voice_control:
        try:
            server.voice_control.stop_listening()
            logger.info("Voice control stopped")
        except Exception as e:
            logger.error(f"Error stopping voice control: {e}")
        server.voice_control = None

    # Stop Auto-EQ
    if server.auto_eq_controller:
        try:
            server.auto_eq_controller.stop()
            logger.info("Auto-EQ stopped")
        except Exception as e:
            logger.error(f"Error stopping Auto-EQ: {e}")
        server.auto_eq_controller = None

    # Stop Multi-channel Auto-EQ
    if server.multi_channel_auto_eq_controller:
        try:
            server.multi_channel_auto_eq_controller.stop_all()
            logger.info("Multi-channel Auto-EQ stopped")
        except Exception as e:
            logger.error(f"Error stopping multi-channel Auto-EQ: {e}")
        server.multi_channel_auto_eq_controller = None

    # Stop Phase alignment
    if server.phase_alignment_controller:
        try:
            server.phase_alignment_controller.stop()
            logger.info("Phase alignment stopped")
        except Exception as e:
            logger.error(f"Error stopping phase alignment: {e}")
        server.phase_alignment_controller = None

    # Stop Auto Fader
    if server.auto_fader_controller:
        try:
            server.auto_fader_controller.stop()
            logger.info("Auto Fader stopped")
        except Exception as e:
            logger.error(f"Error stopping Auto Fader: {e}")
        server.auto_fader_controller = None

    # Stop Auto Soundcheck
    if server.auto_soundcheck_running:
        try:
            server.auto_soundcheck_running = False
            if server.auto_soundcheck_task:
                server.auto_soundcheck_task.cancel()
            logger.info("Auto Soundcheck stopped")
        except Exception as e:
            logger.error(f"Error stopping Auto Soundcheck: {e}")
        server.auto_soundcheck_task = None
        server.auto_soundcheck_websocket = None

    # Stop Auto Compressor
    if server.auto_compressor_controller:
        try:
            server.auto_compressor_controller.stop()
            logger.info("Auto Compressor stopped")
        except Exception as e:
            logger.error(f"Error stopping Auto Compressor: {e}")
        server.auto_compressor_controller = None

    # Stop Feedback Detector
    if getattr(server, "_feedback_audio_capture", None):
        try:
            server._feedback_audio_capture.stop()
            logger.info("Feedback detector audio capture stopped")
        except Exception as e:
            logger.error(f"Error stopping feedback audio capture: {e}")
        server._feedback_audio_capture = None
    server.feedback_detector = None
    server._feedback_channel_mapping = {}

    # MixingAgent (sync best-effort — async disconnect prefers _stop_mixing_agent_internal)
    if getattr(server, "mixing_agent", None):
        try:
            server.mixing_agent.stop()
        except Exception as e:
            logger.debug("mixing_agent stop: %s", e)
    for attr in ("_mixing_agent_task", "_agent_observe_task"):
        task = getattr(server, attr, None)
        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:
                pass
        setattr(server, attr, None)
    server.mixing_agent = None
    if getattr(server, "_agent_audio_capture", None):
        try:
            server._agent_audio_capture.stop()
        except Exception as e:
            logger.debug("agent audio capture stop: %s", e)
        server._agent_audio_capture = None

    # Disconnect mixer
    if server.mixer_client:
        try:
            server.mixer_client.disconnect()
            logger.info("Mixer disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting mixer: {e}")
        server.mixer_client = None

    logger.info("All controllers cleaned up")


def get_initial_controller_state() -> Dict[str, Optional[Any]]:
    """Return dict of controller attribute names -> None for initial state."""
    return {
        "voice_control": None,
        "gain_staging": None,
        "safe_gain_calibrator": None,
        "auto_eq_controller": None,
        "multi_channel_auto_eq_controller": None,
        "phase_alignment_controller": None,
        "auto_fader_controller": None,
        "auto_compressor_controller": None,
        "feedback_detector": None,
    }
