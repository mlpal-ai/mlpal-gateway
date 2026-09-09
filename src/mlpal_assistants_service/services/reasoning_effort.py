"""Universal reasoning effort: one ordinal ladder for every provider.

The wire vocabulary is a FIXED ordered set of names, not a shared provider
vocabulary — names mean relative position. Each model's supported rungs live
in the catalog (`capabilities.effort_levels`, probe-verified at onboarding);
provider spellings are adapter mapping tables. A provider adding a rung is a
catalog entry plus one adapter line, never a client change.

Unsupported rungs CLAMP deterministically toward the model's nearest rung in
the direction of intent, so tier fallbacks keep working when the fallback
model has fewer rungs. Clamps are never silent: the resolution rides on the
response metadata and the usage log, and strict mode turns one into a 400.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlpal_assistants_service.core.exceptions import (
    UnsupportedEffortError,
    ValidationError,
)

LADDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_RANK = {rung: i for i, rung in enumerate(LADDER)}

# Provider-native effort knobs a client may send via model_kwargs. Combining
# one with the universal field is a contradiction, not a merge.
NATIVE_EFFORT_KWARGS = frozenset({"reasoning", "thinking", "output_config", "thinking_config"})


@dataclass(frozen=True)
class EffortResolution:
    requested: str | None
    applied: str | None
    clamped: bool

    def as_metadata(self) -> dict[str, Any]:
        return {"requested": self.requested, "applied": self.applied, "clamped": self.clamped}


NO_EFFORT = EffortResolution(None, None, False)


def supported_levels(capabilities: Any) -> tuple[str, ...]:
    """The model's rungs in ladder order (catalog `capabilities.effort_levels`)."""
    caps = capabilities if isinstance(capabilities, dict) else {}
    declared = caps.get("effort_levels") or ()
    return tuple(rung for rung in LADDER if rung in declared)


def default_effort(capabilities: Any) -> str | None:
    caps = capabilities if isinstance(capabilities, dict) else {}
    value = caps.get("default_effort")
    return value if value in _RANK else None


def resolve_effort(
    requested: str | None,
    capabilities: Any,
    *,
    strict: bool = False,
    model: str = "",
) -> EffortResolution:
    """Map a requested rung onto what `model` accepts.

    - supported → as is
    - above the ceiling → ceiling; below the floor → floor; inside a gap →
      the nearest LOWER rung (cost-conservative)
    - model declares no rungs → nothing is sent (applied=None), reported as a clamp
    - strict → any deviation is a 400 naming the supported set
    """
    if requested is None:
        return NO_EFFORT
    if requested not in _RANK:
        raise ValidationError(f"reasoning_effort must be one of {list(LADDER)}, got {requested!r}")
    supported = supported_levels(capabilities)
    if requested in supported:
        return EffortResolution(requested, requested, False)
    if strict:
        raise UnsupportedEffortError(model, requested, list(supported))
    if not supported:
        return EffortResolution(requested, None, True)
    rank = _RANK[requested]
    lower = [s for s in supported if _RANK[s] < rank]
    higher = [s for s in supported if _RANK[s] > rank]
    applied = lower[-1] if lower else higher[0]
    return EffortResolution(requested, applied, True)


def check_no_native_conflict(requested: str | None, model_kwargs: dict[str, Any] | None) -> None:
    """A universal effort next to a provider-native one is a 400, not a guess."""
    if not requested or not model_kwargs:
        return
    clash = sorted(NATIVE_EFFORT_KWARGS & set(model_kwargs))
    if clash:
        raise ValidationError(
            f"reasoning_effort conflicts with model_kwargs {clash}; send one or the other"
        )
