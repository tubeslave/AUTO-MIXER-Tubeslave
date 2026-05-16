from unittest.mock import MagicMock

from phase_alignment import PhaseAlignmentController


def test_apply_corrections_uses_live_apply_adapter_when_present():
    mixer = MagicMock()
    controller = PhaseAlignmentController(mixer_client=mixer)
    controller.reference_channel = 1
    controller.channels_to_align = [2]
    controller.live_apply_adapter = MagicMock(return_value=True)

    success = controller.apply_corrections(
        {
            (1, 2): {
                "optimal_delay_ms": 0.8,
                "phase_invert": 1,
                "coherence": 0.91,
            }
        }
    )

    assert success is True
    mixer.set_channel_delay.assert_not_called()
    mixer.set_channel_phase_invert.assert_not_called()
    controller.live_apply_adapter.assert_called_once()
