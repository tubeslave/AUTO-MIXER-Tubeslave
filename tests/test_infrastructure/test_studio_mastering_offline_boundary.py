"""Render studio masters in a fresh process with network operations prohibited."""
from pathlib import Path
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("mode", ["builtin", "reference_fallback", "safety_only"])
def test_studio_mastering_needs_no_live_transport_or_audio_device(mode):
    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(r'''
        import json
        import socket
        import sys
        attempts = []

        def deny_network(*args, **kwargs):
            attempts.append("network_attempt")
            raise AssertionError("Offline mastering must not use network/OSC")

        for name in ("connect", "connect_ex", "bind", "send", "sendall", "sendto"):
            setattr(socket.socket, name, deny_network)
        socket.create_connection = deny_network
        socket.getaddrinfo = deny_network

        import numpy as np
        from auto_mastering import AutoMaster
        from studio_mastering_metrics import StudioMasteringMeter

        mode = sys.argv[1]
        sr = 48000
        t = np.arange(sr // 2, dtype=np.float64) / sr
        mono = (0.18 * np.sin(2 * np.pi * 997 * t)).astype(np.float32)
        audio = np.column_stack([mono, mono * 0.5]).astype(np.float32)
        if mode == "safety_only":
            audio *= 10.0
        original = audio.copy()
        audio.setflags(write=False)
        reference = original * 0.6
        original_reference = reference.copy()
        reference.setflags(write=False)
        processor = AutoMaster(sample_rate=sr, target_lufs=-18.0, true_peak_limit=-1.0)
        processor._matchering_available = False
        meter = StudioMasteringMeter(sr)

        if mode == "builtin":
            result = processor.master(audio)
            assert result.success
            rendered = result.audio
            assert abs(result.lufs - meter.integrated_lufs(rendered)) < 0.02
        elif mode == "reference_fallback":
            rendered = processor.master(audio, reference=reference)
        else:
            rendered, reduction = meter.limit_true_peak(audio, ceiling_dbtp=-1.0)
            assert reduction > 0
            assert np.max(np.abs(rendered)) < np.max(np.abs(audio))

        np.testing.assert_array_equal(audio, original)
        np.testing.assert_array_equal(reference, original_reference)
        assert rendered.shape == original.shape
        assert np.isfinite(rendered).all()
        # Linked processing and static safety gain must preserve this L/R ratio.
        np.testing.assert_allclose(rendered[:, 1], rendered[:, 0] * 0.5,
                                   rtol=1e-5, atol=1e-6)
        measurement = meter.measure(rendered)
        assert measurement.true_peak_dbtp <= -0.98
        assert not attempts, attempts
        for name in ("sounddevice", "pyaudio", "soundcard"):
            assert name not in sys.modules, name
        print(json.dumps({"mode": mode, "frames": len(rendered),
                          "channels": rendered.shape[1],
                          "integrated_lufs": measurement.integrated_lufs,
                          "true_peak_dbtp": measurement.true_peak_dbtp,
                          "network_attempts": len(attempts),
                          "input_unchanged": True,
                          "stereo_ratio_preserved": True}))
    ''')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "backend"), str(root)])
    env["OSC_DISABLED"] = "true"
    result = subprocess.run([sys.executable, "-c", script, mode], cwd=root,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    print(result.stdout.strip())
