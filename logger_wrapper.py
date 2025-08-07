"""
Logger wrapper to handle missing loguru gracefully
"""

import logging
import sys
from pathlib import Path

# Try to use loguru, fall back to standard logging
try:
    from loguru import logger
except ImportError:
    # Create a compatible logger that mimics loguru's interface
    class LoguruCompatible:
        def __init__(self):
            self._logger = logging.getLogger("notioniq")
            self._logger.setLevel(logging.INFO)

            # Add console handler if not already present
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)

        def info(self, message):
            self._logger.info(message)

        def debug(self, message):
            self._logger.debug(message)

        def warning(self, message):
            self._logger.warning(message)

        def error(self, message):
            self._logger.error(message)

        def exception(self, message):
            self._logger.exception(message)

        def success(self, message):
            self._logger.info(f"✓ {message}")

        def remove(self):
            """Compatibility method - no-op for standard logging"""
            pass

        def add(self, *args, **kwargs):
            """Compatibility method - no-op for standard logging"""
            pass

    logger = LoguruCompatible()
