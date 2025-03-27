from typing import Callable
import time
from functools import wraps
from utils.logger import logger


def time_execution(func):
    """
    Decorator to measure and print the execution time of a function.

    Args:
        func: The function to be timed

    Returns:
        Wrapped function that prints execution time
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
        return result

    return wrapper


def async_time_execution(func: Callable):
    """Decorator to log execution time of FastAPI endpoints."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}"
            )
            raise

    return async_wrapper
