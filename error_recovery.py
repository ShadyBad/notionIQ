"""
Error Recovery Module for NotionIQ
Implements resilient error handling with circuit breakers, retries, and graceful degradation
"""

import asyncio
import json
import random
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from rich.console import Console

from logger_wrapper import logger

console = Console()

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"  # Can continue with degraded functionality
    MEDIUM = "medium"  # Should retry with backoff
    HIGH = "high"  # Should fail fast
    CRITICAL = "critical"  # Stop all operations


@dataclass
class ErrorContext:
    """Context for error tracking"""

    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime
    stack_trace: str
    retry_count: int = 0
    recovery_attempted: bool = False
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    half_open_requests: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker"""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_requests = 0

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_requests >= self.config.half_open_requests:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is testing recovery"
                )
            self.half_open_requests += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return False

        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.timeout_seconds

    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' is now CLOSED")
        else:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self.name}' is now OPEN after {self.failure_count} failures"
            )

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker '{self.name}' returning to OPEN state")


class CircuitOpenError(Exception):
    """Exception raised when circuit is open"""

    pass


class RetryStrategy:
    """Retry strategy with exponential backoff and jitter"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry strategy"""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for retry attempt"""
        if retry_count <= 0:
            return 0

        # Exponential backoff
        delay = min(
            self.base_delay * (self.exponential_base ** (retry_count - 1)),
            self.max_delay,
        )

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


class ErrorRecoveryManager:
    """Manages error recovery strategies across the application"""

    def __init__(self):
        """Initialize error recovery manager"""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.dead_letter_queue: List[Dict[str, Any]] = []
        self.retry_strategy = RetryStrategy()

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name)
        return self.circuit_breakers[name]

    def with_recovery(
        self,
        func: Callable[..., T],
        recovery_func: Optional[Callable[..., T]] = None,
        circuit_name: Optional[str] = None,
        max_retries: int = 3,
    ) -> Callable[..., T]:
        """Decorator for functions with error recovery"""

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Use circuit breaker if specified
            if circuit_name:
                breaker = self.get_circuit_breaker(circuit_name)
                try:
                    return breaker.call(
                        self._execute_with_retry,
                        func,
                        recovery_func,
                        max_retries,
                        *args,
                        **kwargs,
                    )
                except CircuitOpenError:
                    if recovery_func:
                        logger.warning(
                            f"Circuit open for {circuit_name}, using recovery function"
                        )
                        return recovery_func(*args, **kwargs)
                    raise
            else:
                return self._execute_with_retry(
                    func, recovery_func, max_retries, *args, **kwargs
                )

        return wrapper

    def _execute_with_retry(
        self,
        func: Callable[..., T],
        recovery_func: Optional[Callable[..., T]],
        max_retries: int,
        *args,
        **kwargs,
    ) -> T:
        """Execute function with retry logic"""
        last_error = None

        for retry_count in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e

                # Log error
                error_context = ErrorContext(
                    error_type=type(e).__name__,
                    message=str(e),
                    severity=self._classify_error_severity(e),
                    timestamp=datetime.now(),
                    stack_trace=traceback.format_exc(),
                    retry_count=retry_count,
                )
                self.error_history.append(error_context)

                # Check if we should retry
                if retry_count < max_retries:
                    delay = self.retry_strategy.get_delay(retry_count + 1)
                    logger.warning(
                        f"Retry {retry_count + 1}/{max_retries} after {delay:.2f}s delay: {e}"
                    )
                    time.sleep(delay)
                else:
                    # Max retries reached
                    if recovery_func:
                        logger.warning(f"Max retries reached, using recovery function")
                        try:
                            return recovery_func(*args, **kwargs)
                        except Exception as recovery_error:
                            logger.error(
                                f"Recovery function also failed: {recovery_error}"
                            )
                            self._add_to_dead_letter_queue(
                                func, args, kwargs, last_error
                            )
                            raise last_error
                    else:
                        self._add_to_dead_letter_queue(func, args, kwargs, last_error)
                        raise last_error

        raise last_error

    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and message"""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Critical errors - stop everything
        if any(
            term in error_msg for term in ["api key", "authentication", "unauthorized"]
        ):
            return ErrorSeverity.CRITICAL

        # High severity - fail fast
        if any(term in error_msg for term in ["rate limit", "quota", "banned"]):
            return ErrorSeverity.HIGH

        # Medium severity - retry with backoff
        if any(term in error_msg for term in ["timeout", "connection", "network"]):
            return ErrorSeverity.MEDIUM

        # Low severity - can continue with degraded functionality
        return ErrorSeverity.LOW

    def _add_to_dead_letter_queue(
        self, func: Callable, args: tuple, kwargs: dict, error: Exception
    ):
        """Add failed operation to dead letter queue for later processing"""
        self.dead_letter_queue.append(
            {
                "function": func.__name__,
                "args": str(args)[:100],  # Truncate for storage
                "kwargs": str(kwargs)[:100],
                "error": str(error),
                "timestamp": datetime.now().isoformat(),
            }
        )
        logger.error(f"Added to dead letter queue: {func.__name__}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        if not self.error_history:
            return {
                "total_errors": 0,
                "by_severity": {},
                "by_type": {},
                "recent_errors": [],
            }

        # Count by severity
        by_severity = {}
        by_type = {}

        for error in self.error_history:
            # Count by severity
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # Count by type
            error_type = error.error_type
            by_type[error_type] = by_type.get(error_type, 0) + 1

        # Get recent errors
        recent_errors = [
            {
                "type": e.error_type,
                "message": e.message[:100],
                "severity": e.severity.value,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in list(self.error_history)[-10:]
        ]

        return {
            "total_errors": len(self.error_history),
            "by_severity": by_severity,
            "by_type": by_type,
            "recent_errors": recent_errors,
            "circuit_breakers": {
                name: breaker.state.value
                for name, breaker in self.circuit_breakers.items()
            },
            "dead_letter_queue_size": len(self.dead_letter_queue),
        }

    def process_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Process items in dead letter queue (manual intervention)"""
        processed = []

        while self.dead_letter_queue:
            item = self.dead_letter_queue.pop(0)
            # Log for manual review
            logger.info(f"Processing dead letter item: {item}")
            processed.append(item)

        return processed


class GracefulDegradation:
    """Provides graceful degradation strategies"""

    @staticmethod
    def with_fallback(primary_func: Callable, fallback_func: Callable):
        """Execute with fallback on failure"""

        def wrapper(*args, **kwargs):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed, using fallback: {e}")
                return fallback_func(*args, **kwargs)

        return wrapper

    @staticmethod
    def with_cache(func: Callable, cache_key: str, cache_ttl: int = 3600):
        """Use cached result on failure"""
        cache = {}

        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                cache[cache_key] = {"value": result, "timestamp": time.time()}
                return result
            except Exception as e:
                logger.warning(f"Function failed, checking cache: {e}")
                if cache_key in cache:
                    cached = cache[cache_key]
                    age = time.time() - cached["timestamp"]
                    if age < cache_ttl:
                        logger.info(f"Using cached result (age: {age:.0f}s)")
                        return cached["value"]
                raise

        return wrapper

    @staticmethod
    def with_default(func: Callable, default_value: Any):
        """Return default value on failure"""

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function failed, returning default: {e}")
                return default_value

        return wrapper


# Global error recovery manager instance
error_recovery = ErrorRecoveryManager()


# Decorator for easy use
def with_recovery(
    recovery_func: Optional[Callable] = None,
    circuit_name: Optional[str] = None,
    max_retries: int = 3,
):
    """Decorator to add error recovery to functions"""

    def decorator(func):
        return error_recovery.with_recovery(
            func, recovery_func, circuit_name, max_retries
        )

    return decorator


if __name__ == "__main__":
    # Example usage
    import random

    # Simulate an unreliable API
    @with_recovery(circuit_name="test_api", max_retries=3)
    def unreliable_api_call():
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("API connection failed")
        return "Success!"

    # Test the recovery system
    console.print("[bold]Testing Error Recovery System[/bold]")

    for i in range(10):
        try:
            result = unreliable_api_call()
            console.print(f"✅ Call {i+1}: {result}")
        except Exception as e:
            console.print(f"❌ Call {i+1} failed: {e}")

        time.sleep(0.5)

    # Show statistics
    stats = error_recovery.get_error_statistics()
    console.print("\n[bold]Error Statistics:[/bold]")
    console.print(json.dumps(stats, indent=2))
