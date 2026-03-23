"""
Structured logging configuration using structlog.
Falls back to standard logging if structlog unavailable.
"""
import logging
import sys
import os
from typing import Optional

def setup_logging(level: str = 'INFO', json_output: bool = False, log_file: Optional[str] = None):
    """Configure application logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    try:
        import structlog

        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
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

        # Also configure stdlib logging for libraries
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            level=numeric_level,
            stream=sys.stdout,
        )

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            logging.root.addHandler(file_handler)

        structlog.get_logger().info("Structured logging initialized",
                                     level=level, json=json_output)
        return True

    except ImportError:
        # Fallback to standard logging
        handlers = [logging.StreamHandler(sys.stdout)]

        if log_file:
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            level=numeric_level,
            handlers=handlers,
        )
        logging.getLogger().info("Standard logging initialized (structlog not available)")
        return False

def get_logger(name: str):
    """Get a logger, preferring structlog if available."""
    try:
        import structlog
        return structlog.get_logger(name)
    except ImportError:
        return logging.getLogger(name)
