"""
Performance logging utility for MaZe application.
Provides timing decorators and logging for identifying bottlenecks.
Logs with full stack traces go to daily log files.
Console shows clean, readable timing info.
"""

import time
import functools
import inspect
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Callable, Any
from datetime import datetime

# Configure logging directory
log_dir = Path("app/logs")
log_dir.mkdir(exist_ok=True)

# Create logger with propagation for full stack traces
logger = logging.getLogger("maze_performance")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Don't show in uvicorn console

# Daily rotating file handler - logs with full details
log_filename = log_dir / "performance.log"
file_handler = TimedRotatingFileHandler(
    log_filename,
    when="midnight",
    interval=1,
    backupCount=30,  # Keep 30 days of logs
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.suffix = "%Y-%m-%d"  # Rotated files: performance.log.2025-12-05
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s\n%(pathname)s:%(lineno)d'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


def log_timing(message: str, level: str = "info"):
    """Log timing message to daily file and print cleanly to console."""
    if level == "error":
        logger.error(message, stack_info=True)  # Full stack trace in file
    else:
        logger.info(message)
    print(message)  # Clean console output


def timeit(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    Works with both sync and async functions.
    
    Usage:
        @timeit
        def my_function():
            ...
        
        @timeit
        async def my_async_function():
            ...
    """
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
                log_timing(f"⏱️  {func.__name__} took {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                log_timing(f"❌ {func.__name__} failed after {elapsed:.2f}ms: {e}")
                raise
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
                log_timing(f"⏱️  {func.__name__} took {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                log_timing(f"❌ {func.__name__} failed after {elapsed:.2f}ms: {e}")
                raise
        return sync_wrapper


class PerformanceTimer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with PerformanceTimer("My Operation"):
            # ... code to time
    """
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        log_timing(f"▶️  Starting: {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        if exc_type is None:
            log_timing(f"✅ {self.name} took {elapsed:.2f}ms")
        else:
            log_timing(f"❌ {self.name} failed after {elapsed:.2f}ms: {exc_val}")
        return False


# Example usage
if __name__ == "__main__":
    @timeit
    def slow_function():
        time.sleep(0.5)
        return "Done"
    
    @timeit
    async def async_slow_function():
        import asyncio
        await asyncio.sleep(0.3)
        return "Async Done"
    
    # Test sync
    result = slow_function()
    print(result)
    
    # Test async
    import asyncio
    result = asyncio.run(async_slow_function())
    print(result)
    
    # Test context manager
    with PerformanceTimer("Manual timing test"):
        time.sleep(0.2)
