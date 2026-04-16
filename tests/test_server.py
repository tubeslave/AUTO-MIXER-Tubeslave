"""
Tests for backend/server.py — convert_numpy_types utility and
AutoMixerServer basic initialization.

All tests work without hardware or network (all external connections mocked).
"""

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

try:
    from server import convert_numpy_types, AutoMixerServer
except ImportError:
    pytest.skip("server module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# convert_numpy_types tests
# ---------------------------------------------------------------------------

class TestConvertNumpyTypes:

    def test_int_types(self):
        for dtype in [np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64]:
            val = dtype(42)
            result = convert_numpy_types(val)
            assert isinstance(result, int)
            assert result == 42

    def test_float_types(self):
        for dtype in [np.float32, np.float64]:
            val = dtype(3.14)
            result = convert_numpy_types(val)
            assert isinstance(result, float)
            assert abs(result - 3.14) < 1e-5

    def test_bool_type(self):
        val = np.bool_(True)
        result = convert_numpy_types(val)
        assert isinstance(result, bool)
        assert result is True

    def test_ndarray(self):
        arr = np.array([1, 2, 3])
        result = convert_numpy_types(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_dict_recursive(self):
        data = {"a": np.int32(1), "b": np.float64(2.5), "c": "hello"}
        result = convert_numpy_types(data)
        assert isinstance(result["a"], int)
        assert isinstance(result["b"], float)
        assert result["c"] == "hello"

    def test_list_recursive(self):
        data = [np.int32(1), np.float64(2.5), "hello"]
        result = convert_numpy_types(data)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert result[2] == "hello"

    def test_nested_dict(self):
        data = {"outer": {"inner": np.int64(10)}}
        result = convert_numpy_types(data)
        assert isinstance(result["outer"]["inner"], int)

    def test_native_types_unchanged(self):
        assert convert_numpy_types(42) == 42
        assert convert_numpy_types(3.14) == 3.14
        assert convert_numpy_types("hello") == "hello"
        assert convert_numpy_types(True) is True
        assert convert_numpy_types(None) is None

    def test_tuple_recursive(self):
        data = (np.int32(1), np.float64(2.0))
        result = convert_numpy_types(data)
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)

    def test_json_serializable(self):
        """Result should be JSON-serializable."""
        data = {
            "level": np.float32(-5.0),
            "channel": np.int64(3),
            "muted": np.bool_(False),
            "values": np.array([1.0, 2.0, 3.0]),
        }
        result = convert_numpy_types(data)
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# AutoMixerServer initialization tests (with mocks)
# ---------------------------------------------------------------------------

class TestAutoMixerServerInit:

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_basic_init(self, mock_config, mock_bleed):
        """Server should initialize with default host and port."""
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        assert server.ws_host == "localhost"
        assert server.ws_port == 8765
        assert server.mixer_client is None
        assert server.connection_mode is None
        assert server.connected_clients == set()
        assert server.live_mode is False

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_custom_host_port(self, mock_config, mock_bleed):
        """Server should accept custom host and port."""
        server = AutoMixerServer(ws_host="0.0.0.0", ws_port=9999)
        assert server.ws_host == "0.0.0.0"
        assert server.ws_port == 9999

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_controllers_none_initially(self, mock_config, mock_bleed):
        """All controllers should be None after initialization."""
        server = AutoMixerServer()
        assert server.voice_control is None
        assert server.gain_staging is None
        assert server.auto_eq_controller is None
        assert server.phase_alignment_controller is None
        assert server.auto_fader_controller is None
        assert server.auto_compressor_controller is None
        assert server.mixing_agent is None
        assert server.mixing_agent_task is None

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_auto_soundcheck_not_running(self, mock_config, mock_bleed):
        """Auto soundcheck should not be running initially."""
        server = AutoMixerServer()
        assert server.auto_soundcheck_running is False
        assert server.auto_soundcheck_task is None

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_shutdown_flags(self, mock_config, mock_bleed):
        """Shutdown flags should be properly initialized."""
        server = AutoMixerServer()
        assert server._is_shutting_down is False
        assert server._shutdown_event is None
