"""Circuit breaker pattern for provider resilience.

Implements the circuit breaker pattern to handle provider failures gracefully:
- Prevents cascade failures when a provider is down
- Allows automatic recovery after failures
- Provides metrics for monitoring

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Provider is failing, requests fail fast
- HALF_OPEN: Testing if provider has recovered
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from mlpal_assistants_service.core.exceptions import ProviderUnavailableError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and requests are blocked."""

    def __init__(self, provider: str, retry_after: float):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for {provider}. "
            f"Retry after {retry_after:.1f} seconds."
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 2       # Successes in half-open before closing
    recovery_timeout: float = 30.0   # Seconds before trying half-open
    half_open_max_calls: int = 3     # Max concurrent calls in half-open


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    throttled_calls: int = 0  # provider 429/quota/overloaded — not counted as failures
    state_changes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for a single provider.

    Usage:
        breaker = CircuitBreaker("openai")

        try:
            async with breaker:
                result = await adapter.chat(...)
        except CircuitBreakerOpen as e:
            # Use fallback or return error
            pass
    """

    provider: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    stats: CircuitBreakerStats = field(default_factory=CircuitBreakerStats)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _half_open_calls: int = 0

    async def __aenter__(self) -> "CircuitBreaker":
        """Enter the circuit breaker context."""
        await self._before_call()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the circuit breaker context."""
        if exc_type is None:
            await self._on_success()
        elif exc_type is CircuitBreakerOpen:
            pass  # our own fast-fail, not a provider signal
        elif exc_type is not None and issubclass(exc_type, ProviderUnavailableError):
            # Provider THROTTLING (429 rate limit / quota / overloaded) means the
            # provider is up but busy — not a health failure. Counting it toward
            # the failure threshold lets one caller's rate-limit burst OPEN the
            # per-provider breaker and fail-fast EVERY caller for recovery_timeout.
            # Per-request retry+backoff already handles throttling; the breaker is
            # reserved for genuine faults (5xx / timeouts / connection errors,
            # which surface as ProviderError). Treat throttling as neutral.
            self.stats.throttled_calls += 1
        else:
            await self._on_failure(exc_val)
        return False  # Don't suppress exceptions

    async def _before_call(self) -> None:
        """Check if call is allowed before executing."""
        async with self._lock:
            self.stats.total_calls += 1

            if self.state == CircuitState.CLOSED:
                return  # Allow call

            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._should_attempt_recovery():
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 1
                    logger.info(f"Circuit breaker {self.provider}: OPEN -> HALF_OPEN")
                    return  # Allow test call
                else:
                    # Reject call
                    self.stats.rejected_calls += 1
                    retry_after = self._get_retry_after()
                    raise CircuitBreakerOpen(self.provider, retry_after)

            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return  # Allow call
                else:
                    # Too many concurrent half-open calls
                    self.stats.rejected_calls += 1
                    raise CircuitBreakerOpen(self.provider, 1.0)

    async def _on_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0

            if self.state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(f"Circuit breaker {self.provider}: HALF_OPEN -> CLOSED (recovered)")

    async def _on_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_failure_time = time.time()
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0

            logger.warning(
                f"Circuit breaker {self.provider}: failure #{self.stats.consecutive_failures} - {error}"
            )

            if self.state == CircuitState.CLOSED:
                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.error(
                        f"Circuit breaker {self.provider}: CLOSED -> OPEN "
                        f"(threshold {self.config.failure_threshold} reached)"
                    )

            elif self.state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                # Single failure in half-open returns to open
                self._transition_to(CircuitState.OPEN)
                logger.warning(f"Circuit breaker {self.provider}: HALF_OPEN -> OPEN (failed recovery)")

    def _should_attempt_recovery(self) -> bool:
        """Check if we should try recovery (transition to half-open)."""
        if self.stats.last_failure_time is None:
            return True
        elapsed = time.time() - self.stats.last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _get_retry_after(self) -> float:
        """Get seconds until retry is allowed."""
        if self.stats.last_failure_time is None:
            return 0.0
        elapsed = time.time() - self.stats.last_failure_time
        return max(0.0, self.config.recovery_timeout - elapsed)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self.state != new_state:
            self.state = new_state
            self.stats.state_changes += 1
            if new_state == CircuitState.CLOSED:
                self.stats.consecutive_failures = 0
            elif new_state == CircuitState.OPEN:
                self.stats.consecutive_successes = 0

    def reset(self) -> None:
        """Reset circuit breaker to closed state (for testing)."""
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._half_open_calls = 0

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return {
            "provider": self.provider,
            "state": self.state.value,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "rejected_calls": self.stats.rejected_calls,
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "recovery_timeout": self.config.recovery_timeout,
            },
        }


class CircuitBreakerRegistry:
    """
    Registry for managing circuit breakers across providers.

    Usage:
        registry = CircuitBreakerRegistry()

        # Get circuit breaker for a provider
        breaker = registry.get("openai")

        # Get all breaker statuses
        statuses = registry.get_all_status()
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self._config = config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(self, provider: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        if provider not in self._breakers:
            async with self._lock:
                if provider not in self._breakers:
                    self._breakers[provider] = CircuitBreaker(
                        provider=provider,
                        config=self._config,
                    )
        return self._breakers[provider]

    def get_sync(self, provider: str) -> CircuitBreaker:
        """Synchronous version of get (for non-async contexts)."""
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(
                provider=provider,
                config=self._config,
            )
        return self._breakers[provider]

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            provider: breaker.get_status()
            for provider, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers (for testing)."""
        for breaker in self._breakers.values():
            breaker.reset()


# Decorator for wrapping adapter methods with circuit breaker
def with_circuit_breaker(
    registry: CircuitBreakerRegistry,
    provider_attr: str = "provider_name",
) -> Callable:
    """
    Decorator to wrap adapter methods with circuit breaker.

    Usage:
        @with_circuit_breaker(registry)
        async def chat(self, ...):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            provider = getattr(self, provider_attr, "unknown")
            breaker = await registry.get(provider)

            async with breaker:
                return await func(self, *args, **kwargs)

        return wrapper
    return decorator


_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """The process-wide registry. Routers are built per request; a breaker
    only means something if its failure count outlives the request that
    tripped it, so every router shares this one instance."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry
