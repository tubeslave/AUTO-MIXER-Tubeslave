"""
Structured logging configuration.

Uses structlog for JSON output in production, console output for development.
Falls back to standard logging if structlog is unavailable.
"""

import logging
import sys
from typing import Optional

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def setup_logging(level: str = "INFO", json_output: bool = False,
                  log_file: Optional[str] = None) -> None:
    """Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, output JSON-formatted logs (production).
        log_file: Optional file path for log output.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)

    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    if HAS_STRUCTLOG:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
        ]

        if json_output:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logging.getLogger(__name__).info("structlog configured (json=%s)", json_output)
    else:
        logging.getLogger(__name__).info(
            "structlog not available, using standard logging"
        )


def get_logger(name: str):
    """Get a logger instance. Uses structlog if available."""
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)
