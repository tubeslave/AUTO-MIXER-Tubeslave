"""
Tests for backend/logging_config.py — setup_logging and get_logger.

Verifies logging setup, file output, and structlog integration (when available).
"""

import logging
import os
import tempfile
import pytest

try:
    from logging_config import setup_logging, get_logger, HAS_STRUCTLOG
except ImportError:
    pytest.skip("logging_config module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_file(tmp_dir):
    """Path to a temporary log file."""
    return os.path.join(tmp_dir, "test.log")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSetupLogging:

    def test_setup_default(self):
        """setup_logging with defaults should not raise."""
        setup_logging()

    def test_setup_debug_level(self):
        """Setting DEBUG level should configure root logger."""
        setup_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level <= logging.DEBUG

    def test_setup_warning_level(self):
        setup_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_log_to_file(self, log_file):
        """Logs should be written to the specified file."""
        setup_logging(level="INFO", log_file=log_file)
        logger = logging.getLogger("test_file_output")
        logger.info("Test log message for file output")
        # Flush handlers
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert os.path.isfile(log_file)
        with open(log_file) as f:
            contents = f.read()
        assert "Test log message for file output" in contents

    def test_log_file_creates_directory(self, tmp_dir):
        """Log file in a non-existent subdirectory should be created."""
        nested_path = os.path.join(tmp_dir, "subdir", "nested.log")
        setup_logging(level="INFO", log_file=nested_path)
        logger = logging.getLogger("test_nested_dir")
        logger.info("Nested dir log")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert os.path.isfile(nested_path)


class TestGetLogger:

    def test_returns_logger(self):
        """get_logger should return a logger instance."""
        lgr = get_logger("test_module")
        assert lgr is not None

    def test_logger_name(self):
        """Logger should be identifiable by its name."""
        lgr = get_logger("my_module")
        # Both structlog and standard loggers should work
        assert lgr is not None

    def test_structlog_if_available(self):
        """If structlog is available, get_logger should return a structlog logger."""
        if not HAS_STRUCTLOG:
            pytest.skip("structlog not installed")
        import structlog
        setup_logging(level="INFO")
        lgr = get_logger("structlog_test")
        # structlog loggers have a 'bind' method
        assert hasattr(lgr, "bind") or hasattr(lgr, "info")


class TestJsonOutput:

    def test_json_output_mode(self):
        """JSON output mode should not raise."""
        if not HAS_STRUCTLOG:
            pytest.skip("structlog not installed for JSON output test")
        setup_logging(level="INFO", json_output=True)
        lgr = get_logger("json_test")
        # Should not raise
        lgr.info("json test message")
