"""Owner notifications, delivered by the backend (the gateway has no mail).

The gateway only decides *that* someone must hear something — a key it
auto-revoked, an account it suspended — and hands the facts to the backend's
internal notify endpoint, which owns templates, SES and the console inbox
(planning/handoff-backend-hold-and-notify.md). Best effort: a failed
notification is an ERROR log plus a metric, never an exception on the caller
(the enforcement itself has already happened and is recorded in
abuse_events). Local billing deployments have no backend: no-op.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from mlpal_assistants_service.core.config import get_settings

logger = logging.getLogger(__name__)

KIND_KEY_REVOKED = "key-revoked"
KIND_ACCOUNT_SUSPENDED = "account-suspended"
_TIMEOUT_SECONDS = 5.0


async def notify_owner(kind: str, payload: dict[str, Any]) -> bool:
    """POST ``/api/v1/internal/notify/<kind>`` on the backend. True when the
    backend acknowledged (2xx)."""
    settings = get_settings()
    if getattr(settings, "billing_backend", "managed") == "local":
        return False
    try:
        from mlpal_assistants_service.repositories.billing_repository import (
            _service_auth_headers,
        )

        async with httpx.AsyncClient(
            base_url=settings.backend_base_url,
            timeout=httpx.Timeout(_TIMEOUT_SECONDS),
            headers=_service_auth_headers(settings),
        ) as client:
            response = await client.post(f"/api/v1/internal/notify/{kind}", json=payload)
            response.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001 — logged + metered, never raised
        logger.error(
            "owner notification failed kind=%s user_id=%s error=%s",
            kind, payload.get("user_id"), f"{type(e).__name__}: {e}",
        )
        try:
            from mlpal_assistants_service.core.metrics import get_metrics

            get_metrics().put_metric_sync("OwnerNotifyFailed", 1, dimensions={"kind": kind})
        except Exception:  # noqa: BLE001
            logger.debug("OwnerNotifyFailed metric emit failed", exc_info=True)
        return False
