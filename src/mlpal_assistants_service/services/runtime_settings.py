"""Runtime-overridable settings: change safe knobs from the console without
restarting the gateway.

Most configuration is env-only by design (credentials, DB endpoints — things
that NEED a restart and should live in the deployment, not a database). But a
small whitelist of operational knobs is hot-swappable: the value is stored in
the `gateway_meta` KV table, loaded into an in-process store at startup, and
propagated to every worker via the existing cache-invalidation pub/sub — no
restart, effective within a second.

Precedence (highest wins): runtime override > env var > code default. An
override survives restarts (it's in the DB); clearing it falls back to env.

Adding a knob = one entry in HOT_SETTINGS with a validator. Everything not
listed here stays env-only on purpose.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_META_PREFIX = "setting:"

# In-process store of active overrides: {name: value}. Loaded at startup,
# updated on writes and on "settings" pub/sub invalidations. Reads are plain
# dict lookups — hot-path safe.
_store: dict[str, str] = {}


def _validate_backends_csv(family: str) -> Callable[[str], str]:
    def validate(raw: str) -> str:
        from mlpal_assistants_service.adapters.factory import get_adapter_factory

        factory = get_adapter_factory()
        names = [n.strip() for n in raw.split(",") if n.strip()]
        if not names:
            raise ValueError("priority list cannot be empty")
        valid = {"first_party"} | {
            b for (f, b) in factory._backend_classes if f == family
        }
        bad = [n for n in names if n not in valid]
        if bad:
            raise ValueError(
                f"unknown backend(s) {bad} for {family} — valid: {sorted(valid)}"
            )
        return ",".join(names)

    return validate


# name -> {validate, description, family?}. The whitelist IS the contract:
# only these can be changed at runtime.
HOT_SETTINGS: dict[str, dict[str, Any]] = {
    "openai_backends": {
        "validate": _validate_backends_csv("openai"),
        "description": "Serving-backend priority for OpenAI models",
        "family": "openai",
    },
    "google_backends": {
        "validate": _validate_backends_csv("google"),
        "description": "Serving-backend priority for Google models",
        "family": "google",
    },
    "anthropic_backends": {
        "validate": _validate_backends_csv("anthropic"),
        "description": "Serving-backend priority for Anthropic models",
        "family": "anthropic",
    },
}


def get(name: str) -> str | None:
    """Active runtime override for a setting, or None. Hot-path safe."""
    return _store.get(name)


def source_of(name: str) -> str:
    """Where the effective value comes from: runtime | env | default."""
    if name in _store:
        return "runtime"
    from mlpal_assistants_service.core.config import Settings, get_settings

    current = getattr(get_settings(), name)
    default = Settings.model_fields[name].default
    return "env" if current != default else "default"


async def load(session: Any) -> int:
    """Load all overrides from gateway_meta into the store (startup/reload)."""
    from mlpal_assistants_service.services.catalog_feed import get_meta

    loaded = {}
    for name in HOT_SETTINGS:
        value = await get_meta(session, _META_PREFIX + name)
        if value is not None:
            loaded[name] = value
    _store.clear()
    _store.update(loaded)
    _invalidate_dependents()
    if loaded:
        logger.info("Runtime setting overrides loaded", extra={"names": list(loaded)})
    return len(loaded)


async def set_value(session: Any, name: str, value: str | None) -> None:
    """Set (or clear, with None) a runtime override. Validates, persists,
    applies locally. Caller is responsible for publishing the "settings"
    invalidation so other workers reload."""
    if name not in HOT_SETTINGS:
        raise KeyError(f"'{name}' is not runtime-overridable")
    from mlpal_assistants_service.services.catalog_feed import set_meta

    if value is not None:
        value = HOT_SETTINGS[name]["validate"](value)
    await set_meta(session, _META_PREFIX + name, value)
    if value is None:
        _store.pop(name, None)
    else:
        _store[name] = value
    _invalidate_dependents()
    logger.info(
        "Runtime setting changed", extra={"name": name, "value": value or "(cleared)"}
    )


def _invalidate_dependents() -> None:
    """Drop caches derived from these settings so changes apply immediately."""
    from mlpal_assistants_service.adapters.factory import get_adapter_factory
    from mlpal_assistants_service.services.messages_v2 import anthropic_backend

    get_adapter_factory()._resolution.clear()
    # Native-wire backend lists are keyed by the effective priority, so a
    # changed override already misses; clearing keeps memory bounded and
    # makes the switch explicit in one place.
    anthropic_backend._backends.clear()
    anthropic_backend._backend_lists.clear()
