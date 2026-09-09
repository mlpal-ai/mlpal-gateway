"""Model catalog for GET /v2/catalog (schema 2).

Selection is driven by PER-MODEL ATTRIBUTES, not by a four-point ordinal scale.
Cost, context window, capabilities and latency are independent axes, so a caller
can ask for "long context, cheap" rather than picking a tier that happens to be
"mid" for unrelated reasons. `build_model_attributes` describes every served
model; tiers remain as a curated view derived over them.

The split that matters: OBJECTIVE attributes are merged live from the systems
that already own them, so onboarding a model makes it routable with zero
curation. The one judgment axis, `quality`, is MEASURED not hand-labelled —
delegation feedback where we have it, published benchmarks (curated.json) as the
seed — and is non-exclusionary: a model with no signal has quality=None.

Live-merged at serve time:

  - rel_cost   from model_pricing (blended 3:1 input:output — agent workloads
                are context-heavy — normalized to the top tier's served model
                = 100), so a ledger price change updates the catalog with no
                curation edit.
  - caps       from model_registry capabilities (the flags we live-verify at
                onboarding).
  - availability + failover: each tier serves the first candidate in
                [model, *alternates] that is active, unpaused, and priced. A
                provider suspending a model (it has happened) flips the tier
                to its alternate with zero code change and zero client release.

Alternates are returned in the response (with availability + rel_cost) so
callers can see the whole ladder, pin overrides, or build their own fallback.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from mlpal_assistants_service.core.exceptions import (
    ModelNotAvailableError,
    ModelNotFoundError,
)

_CURATED_PATH = Path(__file__).resolve().parent.parent / "catalog" / "curated.json"

# Capability flags surfaced to clients, in display order.
_CAP_KEYS = ("tools", "vision", "pdf", "audio", "streaming")

# Blend weights for rel_cost: agent traffic is input-heavy (context re-sent
# every turn), so weight input 3:1 over output.
_INPUT_WEIGHT = Decimal(3)
_TOTAL_WEIGHT = Decimal(4)


# Task-QUALITY dimensions the catalog scores models on. Deliberately excludes
# anything already objective in the payload — long-context is `context`, vision
# is `caps`, speed is `throughput` — so this is only genuine, hard-to-observe
# quality. A model's score on a dimension is MEASURED from delegation feedback
# where we have enough of it, else SEEDED from a published benchmark; both carry
# provenance. Absence means unmeasured (non-exclusionary), never "bad at it".
QUALITY_DIMENSIONS = (
    "coding",          # agentic code editing / repo work
    "reasoning",       # deep multi-step problem solving
    "tool_use",        # reliable function calling / agentic loops
    "extraction",      # structured pull-out from documents
    "classification",  # labelling / routing / triage
)

# Throughput classes rank the served fleet on observed generation speed
# (output tokens/sec) into terciles, NOT against invented absolute thresholds —
# so the labels stay meaningful as the fleet and providers change. This is
# sustained decode speed, which dominates total wait for a coding agent emitting
# long edits; it is NOT time-to-first-token (a separate axis we don't yet record).
_THROUGHPUT_CLASSES = ("fast", "medium", "slow")
_MIN_MODELS_FOR_RANKING = 3


def load_curated() -> dict[str, Any]:
    return json.loads(_CURATED_PATH.read_text())


def available_profiles() -> list[str]:
    return sorted((load_curated().get("profiles") or {}).keys())


# Key-scoped view: a predicate over model tags (the key's model_policy).
# None = unrestricted (dashboard/JWT callers).
Allowed = Callable[[str], bool] | None


def _blended(pricing: Any) -> Decimal:
    return (_INPUT_WEIGHT * pricing.input_rate + pricing.output_rate) / _TOTAL_WEIGHT


async def _resolve_candidate(
    tag: str, router: Any, pricing_service: Any, allowed: Allowed = None
) -> dict[str, Any]:
    """One ladder entry → availability + live cost/caps. Unavailable models
    (unknown, inactive, paused, unpriced, or outside the caller's key policy)
    still appear with available=False so callers see the full ladder."""
    entry: dict[str, Any] = {"model": tag, "available": False}
    if allowed is not None and not allowed(tag):
        return entry
    try:
        model = await router.get_model(tag)
        pricing = await pricing_service.get_pricing(tag, "chat")
    except (ModelNotFoundError, ModelNotAvailableError):
        return entry
    if pricing is None:
        return entry
    caps = model.capabilities if isinstance(model.capabilities, dict) else {}
    entry.update(
        available=True,
        provider=model.provider,
        _blended=_blended(pricing),
        _caps=[k for k in _CAP_KEYS if caps.get(k) is True],
        _context={"input": model.context_length, "output": model.max_output_tokens},
    )
    return entry


def _cu_per_1m(rate: Any, rate_unit: str) -> float | None:
    """Normalize a per-rate_unit CU rate to CU per 1M tokens."""
    if rate is None:
        return None
    r = Decimal(rate)
    if rate_unit == "per_1k_tokens":
        r *= Decimal(1000)
    elif rate_unit != "per_1m_tokens":
        return None  # non-token pricing (per_image/per_minute) has no per-1M meaning
    return float(round(r, 4))


def _throughput_classes(stats: dict[str, dict]) -> dict[str, str]:
    """Rank models by observed generation speed into fast/medium/slow terciles.

    Relative, not absolute: 'fast' means fast *for our served fleet*. Ranked on
    ms-per-output-token ascending (== tokens/sec descending), so lowest ms/token
    is 'fast'. Needs at least a few measured models to be meaningful — below that
    everyone is unclassified rather than mislabelled by a degenerate ranking.
    """
    measured = [(t, s["ms_per_output_token"]) for t, s in stats.items()
                if s.get("ms_per_output_token") is not None]
    if len(measured) < _MIN_MODELS_FOR_RANKING:
        return {}
    measured.sort(key=lambda kv: kv[1])
    n = len(measured)
    out: dict[str, str] = {}
    for i, (tag, _) in enumerate(measured):
        out[tag] = _THROUGHPUT_CLASSES[min(i * len(_THROUGHPUT_CLASSES) // n, len(_THROUGHPUT_CLASSES) - 1)]
    return out


def _throughput_entry(cls: str | None, stats: dict | None) -> dict[str, Any]:
    """Shape the throughput axis: tokens/sec is the headline (higher = faster);
    ms/token is its inverse; total ms is context only (length-confounded)."""
    stats = stats or {}
    mspt = stats.get("ms_per_output_token")
    return {
        "class": cls,
        "output_tokens_per_sec": round(1000 / mspt, 1) if mspt else None,
        "ms_per_output_token": mspt,
        "p50_total_ms": stats.get("p50_ms"),
        "p95_total_ms": stats.get("p95_ms"),
        "samples": stats.get("samples", 0),
    }


def _card(card_seed: dict[str, Any]) -> dict[str, Any] | None:
    """The DETERMINISTIC model card: a short summary + published benchmark scores
    (sourced + dated). Never mutated by feedback — a benchmark is a fact about
    abstract capability, not our situational signal. None when neither exists."""
    summary = card_seed.get("summary")
    benchmarks = {
        dim: b for dim, b in (card_seed.get("benchmarks") or {}).items()
        if dim in QUALITY_DIMENSIONS and b.get("score") is not None
    }
    if not summary and not benchmarks:
        return None
    return {"summary": summary, "benchmarks": benchmarks or None}


def _measured(feedback: dict[str, dict] | None) -> dict[str, Any] | None:
    """ORTHOGONAL measured quality from delegation feedback — OUR observed,
    situational signal, kept entirely separate from the deterministic card.
    None until enough feedback accrues (non-exclusionary)."""
    if not feedback:
        return None
    return {
        dim: {"score": s["score"], "accept_rate": s["accept_rate"],
              "escalation_rate": s["escalation_rate"], "samples": s["samples"]}
        for dim, s in feedback.items() if dim in QUALITY_DIMENSIONS
    } or None


def _served_lineage(curated_lineage: dict[str, dict], served: list) -> dict[str, dict]:
    """Join declared lineage (generation/tier/tier_rank) with each served model's
    provider (from the registry), and annotate latest_in_tier: within a
    (provider, tier) the highest generation is the current version of that class.
    Only served models with a declared lineage appear."""
    joined = {
        m.model_tag: {**curated_lineage[m.model_tag], "provider": m.provider}
        for m in served if m.model_tag in curated_lineage
    }
    top: dict[tuple, float] = {}
    for ln in joined.values():
        key = (ln["provider"], ln["tier"])
        top[key] = max(top.get(key, float("-inf")), ln.get("generation", 0))
    for ln in joined.values():
        ln["latest_in_tier"] = ln.get("generation", 0) == top[(ln["provider"], ln["tier"])]
    return joined


def _benchmark_rankings(models: dict[str, Any]) -> dict[str, list[dict]]:
    """Deterministic leaderboard per dimension from card benchmarks. Separate from
    measured (which is per-model) — this ranks published capability, not our
    situational signal. A new model with a benchmark slots in automatically."""
    out: dict[str, list[dict]] = {}
    for dim in QUALITY_DIMENSIONS:
        scored = []
        for tag, m in models.items():
            b = (m.get("card") or {}).get("benchmarks") or {}
            if dim in b:
                scored.append({"model": tag, "score": b[dim]["score"], "source": b[dim]["source"]})
        scored.sort(key=lambda r: r["score"], reverse=True)
        if scored:
            out[dim] = scored
    return out


def _flagships(models: dict[str, Any]) -> dict[str, str]:
    """Per provider, the deterministic 'best' served model: the rank-1 (flagship)
    tier, latest generation of it. Rank dominates generation — Google's flagship
    `pro` can sit on an older generation number than a newer `flash`, so we must
    NOT just take the newest model. Rank-then-version gives the flagship correctly."""
    best: dict[str, tuple] = {}  # provider -> (tier_rank, generation, tag)
    for tag, m in models.items():
        ln = m.get("lineage")
        if not ln:
            continue
        prov, rank, gen = ln["provider"], ln.get("tier_rank", 99), ln.get("generation", 0)
        cur = best.get(prov)  # lower rank wins; ties -> higher generation
        if cur is None or (rank, -gen) < (cur[0], -cur[1]):
            best[prov] = (rank, gen, tag)
    return {prov: t[2] for prov, t in best.items()}


async def build_model_attributes(
    router: Any,
    pricing_service: Any,
    latency_stats: dict[str, dict] | None = None,
    feedback_quality: dict[str, dict] | None = None,
    allowed: Allowed = None,
) -> dict[str, Any]:
    """Every served chat model with the attributes a caller can route on.

    Objective axes (cost, context, caps, throughput) come from the systems that
    own them. `lineage` is deterministic structure (provider/generation/tier/
    rank). `card` is the deterministic benchmark+summary card. `measured` is the
    orthogonal feedback loop — never merged into the card. All judgment fields are
    non-exclusionary (None when absent), so onboarding a model makes it routable.
    """
    latency_stats = latency_stats or {}
    feedback_quality = feedback_quality or {}
    classes = _throughput_classes(latency_stats)
    curated = load_curated()
    cards = curated.get("cards") or {}

    all_models = await router.list_models(operation="chat", include_deprecated=False)
    served = [m for m in all_models
              if not getattr(m, "is_paused", False) and getattr(m, "is_active", True)
              and (allowed is None or allowed(m.model_tag))]
    lineage = _served_lineage(curated.get("lineage") or {}, served)

    out: dict[str, Any] = {}
    for m in served:
        tag = m.model_tag
        pricing = await pricing_service.get_pricing(tag, "chat")
        caps = m.capabilities if isinstance(m.capabilities, dict) else {}
        stats = latency_stats.get(tag)
        ln = lineage.get(tag)
        entry: dict[str, Any] = {
            "provider": m.provider,
            "lineage": {"provider": ln["provider"], "generation": ln["generation"],
                        "tier": ln["tier"], "tier_rank": ln["tier_rank"],
                        "latest_in_tier": ln["latest_in_tier"]} if ln else None,
            "context": {"input": m.context_length, "output": m.max_output_tokens},
            "caps": [k for k in _CAP_KEYS if caps.get(k) is True],
            # Absolute CU/1M — the axis callers actually budget against. (Tiers
            # keep their own top-tier-normalized rel_cost; we deliberately don't
            # publish a second relative scale here.)
            "cost": None if pricing is None else {
                "input_cu_per_1m": _cu_per_1m(pricing.input_cu_rate, pricing.rate_unit),
                "output_cu_per_1m": _cu_per_1m(pricing.output_cu_rate, pricing.rate_unit),
            },
            # Sustained decode SPEED (not time-to-first-token). For a coding
            # agent emitting long edits this dominates total wait, so it leads in
            # tokens/sec. `p50/p95_total_ms` is the whole-request time and is
            # confounded by how many tokens were asked for — kept for tail
            # awareness, not for cross-model speed comparison (use tokens/sec).
            "throughput": _throughput_entry(classes.get(tag), stats),
            "card": _card(cards.get(tag) or {}),        # deterministic
            "measured": _measured(feedback_quality.get(tag)),  # orthogonal loop
        }
        out[tag] = entry
    return out


async def build_catalog(
    profile: str,
    router: Any,
    pricing_service: Any,
    latency_stats: dict[str, dict] | None = None,
    feedback_quality: dict[str, dict] | None = None,
    allowed: Allowed = None,
) -> dict[str, Any] | None:
    """Assemble the served catalog for a profile, or None if unknown.

    Returns `models` (every served model + routable attributes — the source of
    truth), `benchmark_rankings` (per-dimension leaderboard), `flagships`, and
    `tiers` — the curated ladder, now the AUTHORITATIVE subtask-routing interface:
    each tier is guaranteed current-generation, cost-ordered, and carries `caps`
    + real `context` so a client routes a subtask by mapping its complexity
    estimate onto `routing_ladder` (cheapest -> most capable) with no client-side
    "which model is current" logic. The ladder depends only on the registry and
    pricing (not the slowly-refreshing latency/feedback stats), so the routing
    decision is deterministic within the response's cache TTL.
    """
    curated = load_curated()
    prof = (curated.get("profiles") or {}).get(profile)
    if prof is None:
        return None

    tier_names = list(prof["tiers"].keys())  # curation order; first = top tier
    resolved: dict[str, dict[str, Any]] = {}
    for tier in tier_names:
        spec = prof["tiers"][tier]
        ladder = [spec["model"], *spec.get("alternates", [])]
        candidates = [await _resolve_candidate(tag, router, pricing_service, allowed) for tag in ladder]
        resolved[tier] = {
            "spec": spec,
            "candidates": candidates,
            "served": next((c for c in candidates if c["available"]), None),
        }

    # Normalize rel_cost to the top tier's served model (=100); if the whole
    # top ladder is dark, normalize to the priciest served model instead.
    base: Decimal | None = None
    top_served = resolved[tier_names[0]]["served"]
    if top_served is not None:
        base = top_served["_blended"]
    else:
        blends = [r["served"]["_blended"] for r in resolved.values() if r["served"]]
        base = max(blends) if blends else None

    def rel_cost(blend: Decimal) -> int | None:
        if base is None or base == 0:
            return None
        return int(round(Decimal(100) * blend / base))

    tiers_out: dict[str, Any] = {}
    for tier in tier_names:
        spec = resolved[tier]["spec"]
        served = resolved[tier]["served"]
        alternates = []
        for cand in resolved[tier]["candidates"]:
            if cand is served:
                continue
            alt: dict[str, Any] = {"model": cand["model"], "available": cand["available"]}
            if cand["available"]:
                alt["provider"] = cand["provider"]
                alt["rel_cost"] = rel_cost(cand["_blended"])
                # caps + context so a client can pick this fallback for a subtask
                # the tier model can't serve (vision / long context) without a
                # second round-trip.
                alt["caps"] = cand["_caps"]
                alt["context"] = cand["_context"]
            alternates.append(alt)

        if served is None:
            # Whole ladder unavailable — surface it loudly rather than dropping
            # the tier (a silent hole would look like a curation mistake).
            tiers_out[tier] = {
                "model": None,
                "error": "no candidate in this tier is currently available",
                "good_for": spec.get("good_for", ""),
                "alternates": alternates,
            }
            continue

        tiers_out[tier] = {
            "model": served["model"],
            "provider": served["provider"],
            "rel_cost": rel_cost(served["_blended"]),
            # Real registry window, so a client hard-filters a long-context
            # subtask against the tier without guessing.
            "context": served["_context"],
            "good_for": spec.get("good_for", ""),
            "caps": served["_caps"],
            "served_alternate": served["model"] != spec["model"],
            "alternates": alternates,
        }

    models = await build_model_attributes(
        router, pricing_service, latency_stats, feedback_quality, allowed=allowed)
    return {
        "schema": curated.get("schema", 1),
        "profile": profile,
        "updated": curated.get("updated"),
        "quality_dimensions": list(QUALITY_DIMENSIONS),
        # Deterministic 'provider best' — the rank-1 tier's latest served version.
        "flagships": _flagships(models),
        # Deterministic benchmark leaderboard per dimension (from cards; the
        # measured loop is per-model under models[].measured, kept separate).
        "benchmark_rankings": _benchmark_rankings(models),
        # The authoritative subtask ladder, cheapest -> most capable. A client
        # maps its complexity estimate straight onto this (index 0 = cheapest,
        # clamped) and uses tiers[name] — no client-side "which model is current"
        # logic. Curation order is top-first, so we reverse it here. The tiers it
        # names are guaranteed current-generation, cost-ordered, and carry
        # caps + real context so the client can hard-filter special subtasks.
        "routing_ladder": list(reversed(tier_names)),
        "models": models,
        "tiers": tiers_out,
    }
