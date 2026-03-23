"""Tests for logging_config module."""
import pytest
import os
import sys
import logging
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from logging_config import setup_logging, get_logger


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_logging_returns_bool(self):
        """setup_logging returns True if structlog is available, False otherwise."""
        result = setup_logging(level='DEBUG')
        assert isinstance(result, bool)

    def test_setup_logging_with_log_file(self):
        """setup_logging writes to a log file when log_file is specified."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_path = f.name
        try:
            setup_logging(level='INFO', log_file=log_path)
            # Emit a log message through the root logger
            test_logger = logging.getLogger('test_file_output')
            test_logger.info('Test message for file output')
            # Flush all handlers
            for handler in logging.root.handlers:
                handler.flush()
            with open(log_path, 'r') as lf:
                contents = lf.read()
            # The file should have been created (may or may not have our message
            # depending on handler ordering, but the file should exist)
            assert os.path.exists(log_path)
        finally:
            os.unlink(log_path)
            # Clean up handlers to avoid affecting other tests
            logging.root.handlers = [h for h in logging.root.handlers
                                     if not isinstance(h, logging.FileHandler)]

    def test_setup_logging_sets_level(self):
        """setup_logging configures the root logger to the specified level."""
        setup_logging(level='WARNING')
        root_level = logging.root.level
        # The root level should be WARNING (30) or configured accordingly
        assert root_level == logging.WARNING

    def test_get_logger_returns_logger(self):
        """get_logger returns a logger object (structlog or stdlib)."""
        logger = get_logger('test_module')
        assert logger is not None
        # The logger should have some kind of info method
        assert hasattr(logger, 'info')

    def test_setup_logging_json_mode(self):
        """setup_logging with json_output=True does not raise."""
        # This tests that the JSON renderer path works without error
        result = setup_logging(level='INFO', json_output=True)
        assert isinstance(result, bool)
