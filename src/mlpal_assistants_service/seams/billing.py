"""Billing seam — the open-core boundary for spend/wallet enforcement.

The inference services depend on a small `BillingGate` interface (4 methods).
Two implementations are selected once, at the composition root, by
`settings.billing_backend`:

  * "managed" (default) → `repositories.BillingRepository` — the MLPal payments/
    wallet integration (outbound HTTP to backend/payments, `UserBillingStatus`).
  * "local" (OSS) → `LocalBillingGate` — allow-all, ZERO outbound calls. This is
    what lets a self-hosted box run with only Postgres + Redis + provider keys.

Per-key SPEND BUDGETS (`services/policy.py`) are enforced separately and remain
active in both modes — the billing gate is only the user-level wallet/quota
concern, which self-hosters don't have. Auth/policy fail closed; this gate, in
local mode, is intentionally permissive (there is no wallet to check).

`BillingRepository` already satisfies this Protocol structurally, so nothing
about the managed path changes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

import httpx

from mlpal_assistants_service.core.config import get_settings


def is_insufficient_wallet_error(exc: Exception) -> bool:
    """Classify a debit failure as "insufficient wallet balance" (HTTP 402).

    Lives on the seam (not in the managed wallet repository) so services can
    import it without pulling the payments/wallet data-access module into the
    OSS build. Under `LocalBillingGate` the debit never calls out, so this is
    always False in self-hosted mode."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code == 402
    return False


@runtime_checkable
class BillingGate(Protocol):
    """What the inference services need from the billing/wallet layer."""

    async def can_make_request_cached(
        self, user_id: int, estimated_cu: Decimal = Decimal("0")
    ) -> tuple[bool, str | None, bool]:
        """(allowed, reason, billing_existed). billing_existed=False asks the
        caller to ensure_billing_status in the background."""
        ...

    async def is_wallet_debit_active(self) -> bool:
        """Whether a wallet debit should actually be performed post-request."""
        ...

    async def ensure_billing_status(self, user_id: int) -> Any:
        """Ensure a billing record exists (managed only; no-op locally)."""
        ...

    async def debit_wallet_usage(
        self, user_id: int, compute_units: Decimal, usage_ref: str
    ) -> bool:
        """Debit the user's wallet for a completed request (managed only)."""
        ...

    async def reserve_inflight(self, user_id: int) -> tuple[str | None, str | None]:
        """(slot, block_reason). A slot is held while the request is in flight
        when the user's balance is low; block_reason set = refuse (402)."""
        ...

    async def release_inflight(self, slot: str | None) -> None:
        """Give back a slot from reserve_inflight (no-op for None)."""
        ...


class LocalBillingGate:
    """OSS default: self-hosted, no wallet. Allows every request and never calls
    out. Spend control in this mode is per-key budgets via services/policy.py."""

    async def can_make_request_cached(
        self, user_id: int, estimated_cu: Decimal = Decimal("0")
    ) -> tuple[bool, str | None, bool]:
        # allowed, no block reason, billing_existed=True (so the caller never
        # schedules ensure_billing_status).
        return True, None, True

    async def is_wallet_debit_active(self) -> bool:
        return False

    async def ensure_billing_status(self, user_id: int) -> Any:
        return None

    async def debit_wallet_usage(
        self, user_id: int, compute_units: Decimal, usage_ref: str
    ) -> bool:
        return True

    async def reserve_inflight(self, user_id: int) -> tuple[str | None, str | None]:
        return None, None

    async def release_inflight(self, slot: str | None) -> None:
        return None


def build_billing_gate(session: Any, redis: Any, settings: Any = None) -> BillingGate:
    """Composition root for the billing seam. 'managed' (default) preserves
    current behavior; 'local' returns the no-callout OSS gate."""
    settings = settings or get_settings()
    if getattr(settings, "billing_backend", "managed") == "local":
        return LocalBillingGate()
    # Imported lazily so the OSS build never imports the payments/wallet code path.
    from mlpal_assistants_service.repositories import BillingRepository

    return BillingRepository(session, redis)
