"""
Phi-2 interface package with enhanced performance tracking.
"""

from .logging import logger, perf_logger, timer
from .cache import SmartCache

__all__ = ['logger', 'perf_logger', 'timer', 'SmartCache']