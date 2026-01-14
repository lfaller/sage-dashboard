"""Logging configuration for SAGE Dashboard.

Provides a configured logger instance for use throughout the application.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: The name of the logger (usually __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler with a simple format
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger
