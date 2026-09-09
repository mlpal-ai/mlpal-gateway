"""Custom exceptions for the Assistants service."""

from typing import Any


class AssistantsServiceError(Exception):
    """Base exception for all service errors.

    ``http_status`` lets a subclass declare its response code explicitly instead
    of relying on the message-substring heuristics in the global handler. When
    None, the handler falls back to those heuristics (legacy behavior).
    """

    http_status: int | None = None

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(AssistantsServiceError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(AssistantsServiceError):
    """Raised when authorization fails."""

    pass


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid or revoked."""

    def __init__(self, message: str = "Invalid or revoked API key") -> None:
        super().__init__(message)


class RateLimitExceededError(AssistantsServiceError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        limit_type: str = "requests",
    ) -> None:
        super().__init__(message, {"retry_after": retry_after, "limit_type": limit_type})
        self.retry_after = retry_after
        self.limit_type = limit_type


class WalletEmptyError(AssistantsServiceError):
    """Wallet balance exhausted — maps to HTTP 402 with the wallet_empty
    envelope on the OpenAI wire (Anthropic wire handles it in messages_v2)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class QuotaExceededError(AssistantsServiceError):
    """Raised when user quota is exceeded."""

    def __init__(
        self,
        message: str = "Quota exceeded",
        current_usage: float | None = None,
        limit: float | None = None,
    ) -> None:
        super().__init__(message, {"current_usage": current_usage, "limit": limit})
        self.current_usage = current_usage
        self.limit = limit


class ModelAccessDeniedError(AssistantsServiceError):
    """Raised when a key's policy forbids the requested model (403).

    Distinct from ModelNotFound (404): the model exists and is served, but this
    key is not permitted to use it. Carries the machine-readable reason so a
    client can surface which model was denied.
    """

    http_status = 403

    def __init__(self, model: str, reason: str = "not in this key's allowed models") -> None:
        super().__init__(
            f"Model '{model}' is not permitted for this API key ({reason}).",
            {"model": model, "reason": reason, "code": "model_access_denied"},
        )
        self.model = model


class BudgetExceededError(AssistantsServiceError):
    """Raised when a key's spend budget for some window is exhausted (402).

    402 (Payment Required) distinguishes a monetary/budget block from a rate
    limit (429) or a model-access denial (403). Details carry the specific
    budget that tripped so the client knows the window, cap, spend and reset.
    """

    http_status = 402

    def __init__(
        self,
        *,
        window: str,
        unit: str,
        limit: float,
        spent: float,
        reset_at: str | None,
        budget_id: str | None = None,
    ) -> None:
        super().__init__(
            f"Spend budget exhausted: {spent:.4f}/{limit:.4f} {unit} for the "
            f"{window} window" + (f" (resets {reset_at})" if reset_at else ""),
            {
                "code": "budget_exceeded",
                "budget_id": budget_id,
                "window": window,
                "unit": unit,
                "limit": limit,
                "spent": spent,
                "reset_at": reset_at,
            },
        )
        self.window = window
        self.unit = unit
        self.limit = limit
        self.spent = spent
        self.reset_at = reset_at


class ProviderError(AssistantsServiceError):
    """Raised when a provider returns an error."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        original_error: str | None = None,
    ) -> None:
        super().__init__(
            message,
            {
                "provider": provider,
                "status_code": status_code,
                "original_error": original_error,
            },
        )
        self.provider = provider
        self.status_code = status_code
        self.original_error = original_error


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is temporarily unavailable."""

    pass


def http_status_for_provider_error(exc: ProviderError) -> int:
    """Map a ProviderError to the HTTP status the gateway should return.

    A provider error is only surfaced as a client error when the client's own
    request was at fault:
      * 400 / 422  -> same status (e.g. invalid params like max_output_tokens)
      * 429        -> 429 (provider rate limit / quota)
    Everything else — provider 5xx, our-side auth (401/403), model/config issues
    (404), connection failures, or an unknown status — is a gateway/upstream
    failure and maps to 502 (or 503 when explicitly unavailable). This keeps a
    bad client request from masquerading as an outage, without leaking our
    gateway-to-provider auth/config problems as client errors.
    """
    if isinstance(exc, ProviderUnavailableError):
        return 503
    code = exc.status_code
    if code in (400, 422):
        return code
    if code == 429:
        return 429
    return 502


class ModelNotFoundError(AssistantsServiceError):
    """Raised when requested model is not found."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Model '{model}' not found", {"model": model})
        self.model = model


class ModelNotAvailableError(AssistantsServiceError):
    """Raised when model exists but is not available."""

    def __init__(self, model: str, reason: str | None = None) -> None:
        super().__init__(
            f"Model '{model}' is not available",
            {"model": model, "reason": reason},
        )
        self.model = model
        self.reason = reason


class UnsupportedCapabilityError(AssistantsServiceError):
    """Raised when a model doesn't support a requested capability."""

    def __init__(
        self,
        capability: str,
        model: str,
        suggestions: list[str] | None = None,
    ) -> None:
        suggestion_text = ""
        if suggestions:
            suggestion_text = f" Try: {', '.join(suggestions)}"
        super().__init__(
            f"Model '{model}' does not support {capability}.{suggestion_text}",
            {"capability": capability, "model": model, "suggestions": suggestions or []},
        )
        self.capability = capability
        self.model = model
        self.suggestions = suggestions or []


class UnsupportedModelKwargsError(AssistantsServiceError):
    """model_kwargs contained keys the serving adapter can't take. LOUD by
    design: the request is rejected before any provider call — a param is
    never silently dropped (the gateway's inversion of OpenRouter's
    require_parameters-off default)."""

    def __init__(
        self, offending: list[str], allowed: list[str], provider: str
    ) -> None:
        super().__init__(
            f"model_kwargs not supported by the serving adapter ({provider}): "
            f"{', '.join(sorted(offending))}. Supported keys: "
            f"{', '.join(sorted(allowed)) or 'none'}. Nothing was sent to the "
            "provider.",
            {
                "offending": sorted(offending),
                "allowed": sorted(allowed),
                "provider": provider,
            },
        )
        self.offending = sorted(offending)
        self.allowed = sorted(allowed)
        self.provider = provider


class ValidationError(AssistantsServiceError):
    """Raised when request validation fails."""

    pass


class UnsupportedEffortError(AssistantsServiceError):
    """`reasoning_effort_strict` was set and the model does not accept the
    requested rung (no clamp allowed). Nothing was sent to the provider."""

    def __init__(self, model: str, requested: str, supported: list[str]) -> None:
        super().__init__(
            f"reasoning_effort {requested!r} is not supported by {model}; "
            f"supported: {', '.join(supported) or 'none (no effort lever)'}. "
            "Drop reasoning_effort_strict to clamp to the nearest supported rung.",
            {"model": model, "requested": requested, "supported": supported},
        )
        self.model = model
        self.requested = requested
        self.supported = supported


class DatabaseError(AssistantsServiceError):
    """Raised when database operation fails."""

    pass
