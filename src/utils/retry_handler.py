"""Retry handler with exponential backoff for rate limit handling."""
import backoff
from typing import Callable
from functools import wraps
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def is_rate_limit_error(e: Exception) -> bool:
    """Check if exception is a rate limit error.
    
    Args:
        e: Exception to check
        
    Returns:
        True if it's a rate limit error, False otherwise
    """
    # Check for common rate limit error indicators
    error_str = str(e).lower()

    # LiteLLM rate limit errors
    if hasattr(e, 'status_code'):
        if e.status_code == 429:
            return True
    
    # Check error message for rate limit indicators
    rate_limit_indicators = [
        'rate limit',
        'ratelimit',
        'too many requests',
        '429',
        'quota exceeded',
        'throttled',
        'requests per minute'
    ]
    
    return any(indicator in error_str for indicator in rate_limit_indicators)


def is_retryable_error(e: Exception) -> bool:
    """Check if exception is retryable.
    
    Args:
        e: Exception to check
        
    Returns:
        True if error should be retried, False otherwise
    """
    # Rate limit errors are retryable
    if is_rate_limit_error(e):
        return True
    
    # Check for temporary network/service errors
    error_str = str(e).lower()
    retryable_indicators = [
        'timeout',
        'connection',
        'service unavailable',
        '503',
        '502',
        '500',
        'internal server error',
        'temporary'
    ]
    
    return any(indicator in error_str for indicator in retryable_indicators)


def on_backoff(details):
    """Callback for backoff events.
    
    Args:
        details: Backoff details dictionary
    """
    logger.warning(
        f"⏳ Backing off {details['wait']:.1f}s after {details['tries']} tries "
        f"calling {details['target'].__name__} due to {details['exception']}"
    )


def on_giveup(details):
    """Callback when giving up after max retries.
    
    Args:
        details: Backoff details dictionary
    """
    logger.error(
        f"❌ Gave up calling {details['target'].__name__} after {details['tries']} tries. "
        f"Last exception: {details['exception']}"
    )


def retry_with_backoff(
    max_tries: int = 5,
    max_time: int = 60,
    base_delay: float = 1.0,
    max_delay: float = 30.0
):
    """Decorator to retry function with exponential backoff on rate limits.
    
    Args:
        max_tries: Maximum number of retry attempts
        max_time: Maximum total time to spend retrying (seconds)
        base_delay: Base delay for exponential backoff (seconds)
        max_delay: Maximum delay between retries (seconds)
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=max_tries,
            max_time=max_time,
            base=base_delay,
            max_value=max_delay,
            giveup=lambda e: not is_retryable_error(e),
            on_backoff=on_backoff,
            on_giveup=on_giveup
        )
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error type for debugging
                if is_rate_limit_error(e):
                    logger.warning(f"🚦 Rate limit encountered in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


def retry_llm_call(func: Callable) -> Callable:
    """Decorator specifically for LLM API calls with appropriate retry settings.
    
    This uses more aggressive retry settings suitable for LLM APIs:
    - Up to 5 retries
    - Max 120 seconds total
    - Exponential backoff starting at 2 seconds
    - Max delay of 60 seconds between retries
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with LLM-specific retry logic
    """
    return retry_with_backoff(
        max_tries=5,
        max_time=120,
        base_delay=2.0,
        max_delay=60.0
    )(func)

