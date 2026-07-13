"""Tests for audio_capture module."""
import sys
import os
import threading
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pytest
import audio_capture as audio_capture_module
from audio_capture import AudioCapture, AudioSourceType


class TestAudioCapture:
    def test_init(self):
        capture = AudioCapture(num_channels=8, sample_rate=48000)
        assert capture is not None
        assert capture.num_channels == 8
        assert capture.sample_rate == 48000

    def test_test_mode(self):
        """Test that test signal generation works."""
        capture = AudioCapture(num_channels=2, sample_rate=48000)
        assert capture.source_type is not None

    @pytest.mark.parametrize(
        "source_type",
        [AudioSourceType.DANTE, AudioSourceType.SOUNDGRID],
    )
    def test_network_audio_backends_report_fresh_stream_as_real_input(
        self,
        source_type,
    ):
        capture = AudioCapture(
            num_channels=48,
            sample_rate=48000,
            source_type=source_type,
            device_name=7,
        )
        capture._running = True
        capture._stream = object()
        capture._last_callback_monotonic = time.monotonic()
        capture._captured_sample_count = 1024

        status = capture.get_status()

        assert status["source_type"] == source_type.value
        assert status["real_input"] is True
        assert status["captured_sample_count"] == 1024

    def test_real_input_can_refuse_synthetic_silence_fallback(self, monkeypatch):
        class FailingInputStream:
            def __init__(self, **_kwargs):
                raise RuntimeError("device unavailable")

        monkeypatch.setattr(audio_capture_module, "HAS_SOUNDDEVICE", True)
        monkeypatch.setattr(audio_capture_module.sd, "InputStream", FailingInputStream)
        capture = AudioCapture(
            num_channels=48,
            sample_rate=48000,
            source_type=AudioSourceType.SOUNDDEVICE,
            device_name=1,
        )
        capture.allow_silence_fallback = False

        with pytest.raises(RuntimeError, match="refusing synthetic silence fallback"):
            capture.start()

        assert capture.running is False
        assert capture.source_type == AudioSourceType.SOUNDDEVICE

    def test_real_input_start_failure_closes_created_stream(self, monkeypatch):
        calls = []

        class FailingStartInputStream:
            def __init__(self, **_kwargs):
                calls.append("created")

            def start(self):
                calls.append("start")
                raise RuntimeError("start failed")

            def stop(self):
                calls.append("stop")

            def close(self):
                calls.append("close")

        monkeypatch.setattr(audio_capture_module, "HAS_SOUNDDEVICE", True)
        monkeypatch.setattr(
            audio_capture_module.sd, "InputStream", FailingStartInputStream
        )
        capture = AudioCapture(
            num_channels=48,
            sample_rate=48000,
            source_type=AudioSourceType.SOUNDDEVICE,
            device_name=1,
        )
        capture.allow_silence_fallback = False

        with pytest.raises(RuntimeError, match="refusing synthetic silence fallback"):
            capture.start()

        assert calls == ["created", "start", "stop", "close"]
        assert capture._stream is None
        assert capture.running is False
        assert capture.source_type == AudioSourceType.SOUNDDEVICE

    def test_stop_still_closes_stream_when_stream_stop_raises(self):
        calls = []

        class StopFailingStream:
            def stop(self):
                calls.append("stop")
                raise RuntimeError("stop failed")

            def close(self):
                calls.append("close")

        capture = AudioCapture(num_channels=2, sample_rate=48000)
        capture._running = True
        capture._stream = StopFailingStream()

        with pytest.raises(RuntimeError, match="AudioCapture stop incomplete"):
            capture.stop()

        assert calls == ["stop", "close"]
        assert capture._stream is not None
        assert capture.running is False

    def test_start_failure_retains_stream_when_cleanup_is_incomplete(self, monkeypatch):
        class OrphanedStartStream:
            def start(self):
                raise RuntimeError("start failed")

            def stop(self):
                raise RuntimeError("stop failed")

            def close(self):
                raise RuntimeError("close failed")

        monkeypatch.setattr(audio_capture_module, "HAS_SOUNDDEVICE", True)
        monkeypatch.setattr(
            audio_capture_module.sd,
            "InputStream",
            lambda **_kwargs: OrphanedStartStream(),
        )
        capture = AudioCapture(
            num_channels=2,
            sample_rate=48000,
            source_type=AudioSourceType.SOUNDDEVICE,
        )
        capture.allow_silence_fallback = False

        with pytest.raises(RuntimeError, match="startup cleanup incomplete"):
            capture.start()

        assert capture._stream is not None
        assert capture.running is False

    def test_synchronized_buffers_wait_for_complete_multichannel_callback(self):
        capture = AudioCapture(
            num_channels=2,
            sample_rate=4,
            buffer_seconds=2,
            block_size=2,
            source_type=AudioSourceType.SOUNDDEVICE,
        )
        first_channel_written = threading.Event()
        release_callback = threading.Event()
        original_write = capture._buffers[1].write

        def delayed_first_channel_write(samples):
            original_write(samples)
            first_channel_written.set()
            assert release_callback.wait(1.0)

        capture._buffers[1].write = delayed_first_channel_write
        callback = threading.Thread(
            target=capture._audio_callback,
            args=(
                np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
                2,
                None,
                None,
            ),
        )
        callback.start()
        assert first_channel_written.wait(1.0)

        result = {}

        def snapshot():
            result.update(capture.get_synchronized_buffers([1, 2], 2))

        reader = threading.Thread(target=snapshot)
        reader.start()
        time.sleep(0.02)
        assert reader.is_alive()

        release_callback.set()
        callback.join(1.0)
        reader.join(1.0)

        assert callback.is_alive() is False
        assert reader.is_alive() is False
        np.testing.assert_array_equal(result["buffers"][1], [1.0, 1.0])
        np.testing.assert_array_equal(result["buffers"][2], [2.0, 2.0])
        assert result["captured_sample_count"] == 2

    def test_stop_reports_capture_thread_that_does_not_terminate(self):
        capture = AudioCapture(num_channels=2, sample_rate=48000)

        class StuckThread:
            def is_alive(self):
                return True

            def join(self, timeout):
                assert timeout == 2.0

        capture._running = True
        capture._capture_thread = StuckThread()

        with pytest.raises(RuntimeError, match="capture_thread"):
            capture.stop()

        assert capture._capture_thread is not None
        assert capture.running is False
