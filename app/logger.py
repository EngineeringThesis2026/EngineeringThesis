"""
Logging configuration for the Legal Advisory System.

Provides centralized logging with console and optional file output.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "legal_advisory", level: int = logging.INFO, log_to_file: bool = True) -> logging.Logger:
    """
    Set up and configure logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_to_file: Whether to log to file (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        try:
            # Create logs directory if it doesn't exist
            logs_dir = Path(__file__).parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)

            log_file = logs_dir / "app.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, just use console
            logger.warning(f"Could not set up file logging: {e}")

    return logger


# Create default logger instance
logger = setup_logger()
