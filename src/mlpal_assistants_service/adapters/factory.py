"""Factory for creating provider adapters.

The AdapterFactory provides a centralized way to create and manage
provider adapters, supporting:
- Lazy initialization (adapters created on first use)
- Singleton pattern (one adapter instance per provider)
- Runtime registration of new adapters, plus pip-installable plugins via the
  `mlpal.adapters` entry-point group
- Configuration-based provider enabling/disabling
- Multi-cloud serving backends: per-family priority lists
  (`MLPAL_<FAMILY>_BACKENDS`) resolved to a concrete adapter per model at
  first use, cached in-process — the hot path is a dict lookup, never I/O.
"""

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from mlpal_assistants_service.core.config import get_settings

if TYPE_CHECKING:
    from mlpal_assistants_service.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)

# Families that support serving backends, and the valid backend names for
# each. `first_party` is always valid and always the default.
FAMILY_BACKENDS: dict[str, tuple[str, ...]] = {
    "openai": ("first_party", "azure"),
    "google": ("first_party", "vertex"),
    "anthropic": ("first_party", "bedrock", "vertex", "azure"),
}


class AdapterFactory:
    """
    Factory for creating and managing provider adapters.

    Usage:
        factory = AdapterFactory()

        # Get adapter (created on first access)
        openai_adapter = factory.get("openai")

        # Get all enabled adapters
        adapters = factory.get_all()

        # Check if provider is available
        if factory.is_available("anthropic"):
            adapter = factory.get("anthropic")
    """

    # Registry of adapter classes (provider name -> class)
    _adapter_classes: dict[str, type["BaseAdapter"]] = {}

    # Singleton instances (provider name -> instance)
    _instances: dict[str, "BaseAdapter"] = {}

    # Backend adapter classes: (family, backend_name) -> class. first_party
    # entries alias _adapter_classes; cloud backends and plugins land here.
    _backend_classes: dict[tuple[str, str], type["BaseAdapter"]] = {}

    # Resolution cache: (family, provider_model_id) -> (adapter, wire_id).
    # Filled lazily, valid for process lifetime (backend config is env-only).
    _resolution: dict[tuple[str, str, frozenset[str]], tuple["BaseAdapter", str]] = {}

    def __init__(self) -> None:
        """Initialize the factory with default adapters."""
        self._register_default_adapters()
        self._load_entry_point_adapters()

    def _register_default_adapters(self) -> None:
        """Register all built-in adapter classes."""
        # Import here to avoid circular imports
        from mlpal_assistants_service.adapters.anthropic import AnthropicAdapter
        from mlpal_assistants_service.adapters.bedrock import BedrockAdapter
        from mlpal_assistants_service.adapters.google import GoogleAdapter
        from mlpal_assistants_service.adapters.openai import OpenAIAdapter
        from mlpal_assistants_service.adapters.serving import (
            AzureAnthropicAdapter,
            AzureOpenAIAdapter,
            BedrockAnthropicAdapter,
            VertexAnthropicAdapter,
            VertexGoogleAdapter,
        )

        self._adapter_classes = {
            "openai": OpenAIAdapter,
            "anthropic": AnthropicAdapter,
            "google": GoogleAdapter,
            "bedrock": BedrockAdapter,
        }
        self._backend_classes = {
            ("openai", "azure"): AzureOpenAIAdapter,
            ("anthropic", "azure"): AzureAnthropicAdapter,
            ("google", "vertex"): VertexGoogleAdapter,
            ("anthropic", "bedrock"): BedrockAnthropicAdapter,
            ("anthropic", "vertex"): VertexAnthropicAdapter,
        }

    def _load_entry_point_adapters(self) -> None:
        """Discover pip-installed adapter plugins.

        A plugin declares, in its pyproject:

            [project.entry-points."mlpal.adapters"]
            myprovider = "my_pkg.adapter:MyAdapter"          # new provider
            "openai:mycloud" = "my_pkg.adapter:MyBackend"    # serving backend

        Plain names register a new provider; `family:backend` names register
        a serving backend usable in MLPAL_<FAMILY>_BACKENDS. Load failures
        are logged, never fatal — one broken plugin must not take the
        gateway down.
        """
        for ep in entry_points(group="mlpal.adapters"):
            try:
                cls = ep.load()
            except Exception as e:  # noqa: BLE001 — isolate plugin failures
                logger.error(f"Failed to load adapter plugin {ep.name!r}: {e}")
                continue
            if ":" in ep.name:
                family, backend = ep.name.split(":", 1)
                self._backend_classes[(family, backend)] = cls
                logger.info(f"Registered plugin backend {backend!r} for family {family!r}")
            else:
                self._adapter_classes[ep.name] = cls
                logger.info(f"Registered plugin provider {ep.name!r}")

    @classmethod
    def register(cls, provider: str, adapter_class: type["BaseAdapter"]) -> None:
        """
        Register a new adapter class.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            adapter_class: Adapter class (must extend BaseAdapter)

        Example:
            AdapterFactory.register("custom", CustomAdapter)
        """
        cls._adapter_classes[provider] = adapter_class
        logger.info(f"Registered adapter for provider: {provider}")

    def get(self, provider: str) -> "BaseAdapter":
        """
        Get an adapter instance for a provider.

        Creates the adapter on first access (lazy initialization).
        Returns cached instance on subsequent calls.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            Adapter instance

        Raises:
            ValueError: If provider is not registered
            RuntimeError: If adapter fails to initialize
        """
        # Return cached instance if exists
        if provider in self._instances:
            return self._instances[provider]

        # Get adapter class
        adapter_class = self._adapter_classes.get(provider)
        if not adapter_class:
            available = ", ".join(sorted(self._adapter_classes.keys()))
            raise ValueError(
                f"Unknown provider: '{provider}'. "
                f"Available providers: {available}"
            )

        # Create instance
        try:
            instance = adapter_class()
            self._instances[provider] = instance
            logger.info(f"Created adapter instance for provider: {provider}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create adapter for {provider}: {e}")
            raise RuntimeError(f"Failed to initialize {provider} adapter: {e}") from e

    # =========================================================================
    # Serving-backend resolution (multi-cloud)
    # =========================================================================

    # Backend instances: (family, backend) -> instance, or None if the
    # backend failed to construct (unconfigured). Cached for process life.
    _backend_instances: dict[tuple[str, str], "BaseAdapter | None"] = {}

    def _priority(self, family: str) -> list[str]:
        """Parse and validate the priority CSV for a family. A runtime
        override (console-set, DB-persisted) wins over the env value."""
        from mlpal_assistants_service.services import runtime_settings

        settings = get_settings()
        raw = runtime_settings.get(f"{family}_backends") or {
            "openai": settings.openai_backends,
            "google": settings.google_backends,
            "anthropic": settings.anthropic_backends,
        }.get(family, "first_party")
        names = [n.strip() for n in raw.split(",") if n.strip()]
        valid = []
        for n in names:
            if n == "first_party" or (family, n) in self._backend_classes:
                valid.append(n)
            else:
                logger.error(
                    f"Unknown backend {n!r} in MLPAL_{family.upper()}_BACKENDS — skipping"
                )
        return valid or ["first_party"]

    def _get_backend(self, family: str, backend: str) -> "BaseAdapter | None":
        """Instance for (family, backend), or None if unconfigured.

        first_party configured-ness is the provider key check; cloud/plugin
        backends own their config knowledge — construction failure means
        unconfigured, logged once and cached.
        """
        if backend == "first_party":
            if not self.is_enabled(family):
                return None
            try:
                return self.get(family)
            except RuntimeError:
                return None
        key = (family, backend)
        if key in self._backend_instances:
            return self._backend_instances[key]
        try:
            instance = self._backend_classes[key]()
            logger.info(f"Created {backend} backend for family {family}")
        except Exception as e:  # noqa: BLE001 — unconfigured/broken backend ≠ fatal
            logger.warning(f"Backend {backend!r} for {family!r} unavailable: {e}")
            instance = None
        self._backend_instances[key] = instance
        return instance

    def resolve(
        self,
        family: str,
        provider_model_id: str,
        exclude: frozenset[str] = frozenset(),
    ) -> tuple["BaseAdapter", str]:
        """Pick the adapter that serves `provider_model_id` for `family`.

        Walks the family's priority list; first backend that is configured
        and serves the model wins. Result is cached — the steady-state cost
        is one dict lookup. Families without backend support (bedrock
        open-weights, plugin providers) resolve to their sole adapter.

        Returns (adapter, wire_model_id) where wire_model_id is what goes
        on the provider call (deployment name on Azure, mapped ID on
        Bedrock/Vertex Claude, unchanged for first-party).

        `exclude` skips backends by name — backend failover retries the same
        model on the next backend in priority order without the one that
        just failed. Cached per (family, model, exclude).

        Raises:
            ValueError: no configured backend serves this model.
        """
        cache_key = (family, provider_model_id, exclude)
        hit = self._resolution.get(cache_key)
        if hit is not None:
            return hit
        if family not in FAMILY_BACKENDS and not any(
            f == family for f, _ in self._backend_classes
        ):
            if not self.is_enabled(family):
                raise ValueError(f"Provider {family!r} is not configured")
            result = (self.get(family), provider_model_id)
            self._resolution[cache_key] = result
            return result
        tried = []
        for backend in self._priority(family):
            if backend in exclude:
                tried.append(f"{backend} (excluded)")
                continue
            adapter = self._get_backend(family, backend)
            if adapter is None:
                tried.append(f"{backend} (unconfigured)")
                continue
            if not adapter.serves(provider_model_id):
                tried.append(f"{backend} (does not serve model)")
                continue
            result = (adapter, adapter.backend_model_id(provider_model_id))
            self._resolution[cache_key] = result
            logger.info(
                f"Resolved {family}/{provider_model_id} -> backend {backend}"
            )
            return result
        raise ValueError(
            f"No configured backend serves {family}/{provider_model_id} "
            f"(tried: {', '.join(tried) or 'none'})"
        )

    def backend_priority(self, family: str) -> list[str]:
        """The validated priority list for a family (['first_party'] for
        providers without backend support). For operator-facing surfaces."""
        if family not in FAMILY_BACKENDS:
            return ["first_party"]
        return self._priority(family)

    def serving_backend_for(self, family: str, provider_model_id: str) -> str | None:
        """Backend name that would serve this model, or None if unserved.
        Used by /v1/models so the console can mark unserved models."""
        try:
            adapter, _ = self.resolve(family, provider_model_id)
        except (ValueError, RuntimeError):
            return None
        return adapter.backend_name

    def get_all(self) -> dict[str, "BaseAdapter"]:
        """
        Get all adapter instances.

        Creates adapters that haven't been initialized yet.

        Returns:
            Dict mapping provider names to adapter instances
        """
        for provider in self._adapter_classes:
            if provider not in self._instances:
                try:
                    self.get(provider)
                except Exception as e:
                    logger.warning(f"Could not initialize {provider} adapter: {e}")

        return dict(self._instances)

    def primary(self, provider: str) -> "BaseAdapter | None":
        """Highest-priority configured backend for a provider/family, or
        None when nothing is configured. This is what health checks and the
        adapter map should use — with only Azure configured, the openai
        family's primary is the Azure backend, not a keyless first-party
        adapter."""
        if provider in FAMILY_BACKENDS:
            for backend in self._priority(provider):
                adapter = self._get_backend(provider, backend)
                if adapter is not None:
                    return adapter
            return None
        if not self.is_enabled(provider):
            return None
        try:
            return self.get(provider)
        except RuntimeError:
            return None

    def get_enabled(self) -> dict[str, "BaseAdapter"]:
        """
        Get the primary adapter for every provider with a configured backend.

        Returns:
            Dict of provider name -> adapter for providers with valid config
        """
        enabled = {}
        for provider in self._adapter_classes:
            adapter = self.primary(provider)
            if adapter is not None:
                enabled[provider] = adapter
        return enabled

    def is_available(self, provider: str) -> bool:
        """
        Check if a provider is registered and can be initialized.

        Args:
            provider: Provider name

        Returns:
            True if provider is available
        """
        return provider in self._adapter_classes

    def is_enabled(self, provider: str) -> bool:
        """
        Check if a provider has valid configuration.

        Args:
            provider: Provider name

        Returns:
            True if provider is configured and enabled
        """
        settings = get_settings()

        provider_keys = {
            "openai": settings.openai_api_key,
            "anthropic": settings.anthropic_api_key,
            "google": settings.google_api_key,
            "bedrock": settings.enable_bedrock,
        }

        if provider_keys.get(provider):
            return True
        # A family without a first-party key is still enabled when a cloud
        # backend is configured (e.g. Azure-only or Vertex-only boxes).
        if provider in FAMILY_BACKENDS:
            return any(
                backend != "first_party"
                and self._get_backend(provider, backend) is not None
                for backend in self._priority(provider)
            )
        if provider in provider_keys:
            return False
        # Plugin providers own their config knowledge: constructible == enabled.
        if provider in self._adapter_classes:
            try:
                self.get(provider)
                return True
            except RuntimeError:
                return False
        return False

    @property
    def providers(self) -> list[str]:
        """Get list of all registered provider names."""
        return list(self._adapter_classes.keys())

    @property
    def enabled_providers(self) -> list[str]:
        """Get list of providers with valid configuration."""
        return [p for p in self.providers if self.is_enabled(p)]

    def clear_instances(self) -> None:
        """Clear all cached adapter instances (for testing)."""
        self._instances.clear()
        self._backend_instances.clear()
        self._resolution.clear()


# Singleton factory instance
_factory: AdapterFactory | None = None


def get_adapter_factory() -> AdapterFactory:
    """Get the singleton AdapterFactory instance."""
    global _factory
    if _factory is None:
        _factory = AdapterFactory()
    return _factory


def get_adapter(provider: str) -> "BaseAdapter":
    """
    Convenience function to get an adapter.

    Args:
        provider: Provider name

    Returns:
        Adapter instance
    """
    return get_adapter_factory().get(provider)
