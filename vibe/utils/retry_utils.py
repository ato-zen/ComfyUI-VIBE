"""Retry utils."""

import functools
import time
from collections.abc import Callable
from typing import Any

import loguru  # noqa: TC002


def retry_decorator(
    logger: "loguru.Logger",
    max_retries: int = 5,
    delay: int = 0,
    exception: type[Exception] = Exception,
) -> Callable:
    """Retry a function up to retries times with a delay between retries if a specified exception is raised.

    Args:
        logger (loguru.Logger): The logger to use for logging.
        max_retries (int): Number of times to retry the function.
        delay (int): Time to wait between retries (in seconds).
        exception (Exception): The type of exception to catch and retry on.

    Returns:
        function: The wrapped function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except exception as exp:
                    attempts += 1
                    logger.debug(f"Attempt {attempts}/{max_retries} failed: {exp}")
                    if attempts == max_retries:
                        logger.exception("Max retries reached.")
                        raise
                    time.sleep(delay)

            message = "Retry loop exited without returning or raising."
            raise RuntimeError(message)

        return wrapper

    return decorator
