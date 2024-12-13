"""
Performance logging configuration for Phi-2 interface.
"""

import logging
from pathlib import Path
import time
from contextlib import contextmanager, asynccontextmanager

# Set up main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phi2")

# Set up performance logger
perf_logger = logging.getLogger("phi2.performance")
perf_logger.setLevel(logging.INFO)

# Create logs directory
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
perf_handler = logging.FileHandler(log_dir / "phi2_performance.log")
perf_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s'
))
perf_logger.addHandler(perf_handler)

@contextmanager
def sync_timer(operation: str):
    """Synchronous context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        perf_logger.info(f"{operation}: {elapsed*1000:.2f}ms")

@asynccontextmanager
async def timer(operation: str):
    """Asynchronous context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        perf_logger.info(f"{operation}: {elapsed*1000:.2f}ms")