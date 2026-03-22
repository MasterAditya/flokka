"""Timing utilities for observability."""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any


def timed(logger: logging.Logger | None = None) -> Callable:
    """Decorator that logs execution time of a function."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or logging.getLogger(fn.__module__)
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _logger.debug("%s completed in %.2f ms", fn.__name__, elapsed_ms)
            return result

        return wrapper

    return decorator


def measure_ms(fn: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms
