"""Tests for the curated /v2/catalog builder: live merge of curation with
registry availability + pricing, rel_cost normalization, tier failover, and
the alternates ladder exposed to callers."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlpal_assistants_service.core.exceptions import ModelNotAvailableError
from mlpal_assistants_service.services.catalog import build_catalog, load_curated

CAPS = {"pdf": True, "audio": False, "tools": True, "vision": True, "operation": "chat", "streaming": True}

# Real ledger rates so rel_cost assertions track the actual curation.
RATES = {
    "claude-fable-5": ("anthropic", "10", "50"),      # blended 20.000
    "claude-opus-5": ("anthropic", "5", "25"),        # blended 10.000
    "claude-opus-4-8": ("anthropic", "5", "25"),      # blended 10.000
    "gpt-5.6-sol": ("openai", "5", "30"),
    "gpt-6-astra": ("openai", "10", "50"),             # blended 11.250
    "gpt-5.6-terra": ("openai", "2.50", "15"),        # blended  5.625
    "claude-sonnet-5": ("anthropic", "3", "15"),      # blended  6.000
    "gpt-5.5": ("openai", "2.50", "15"),
    "gpt-5.6-luna": ("openai", "1", "6"),             # blended  2.250
    "gemini-3.5-flash": ("google", "0.30", "2.50"),   # blended  0.850
    # Google flagship `pro` sits on an OLDER generation (3.1) than the newer
    # `flash` (3.5) — the edge case: rank must beat generation in flagship pick.
    "gemini-3.1-pro-preview": ("google", "1.25", "10"),
    "claude-haiku-4-5-20251001": ("anthropic", "1", "5"),
    # Served but deliberately ABSENT from curated.json's models map — proves a
    # model needs no curation entry to be fully routable (schema-2 invariant).
    "gpt-4o": ("openai", "2.50", "10"),
}


def _model(tag: str, provider: str):
    return SimpleNamespace(
        model_tag=tag, provider=provider, capabilities=dict(CAPS),
        context_length=200000, max_output_tokens=64000,
        is_active=True, is_paused=False,
    )


def _mocks(unavailable: set[str] = frozenset()):
    router = MagicMock()

    async def get_model(tag):
        if tag in unavailable:
            raise ModelNotAvailableError(tag, "paused")
        provider, _, _ = RATES[tag]
        return _model(tag, provider)

    router.get_model = AsyncMock(side_effect=get_model)

    async def list_models(operation=None, include_deprecated=False):
        return [_model(t, p) for t, (p, _, _) in RATES.items() if t not in unavailable]

    router.list_models = AsyncMock(side_effect=list_models)
    pricing = MagicMock()

    async def get_pricing(tag, op):
        _, i, o = RATES[tag]
        return SimpleNamespace(
            input_rate=Decimal(i), output_rate=Decimal(o),
            input_cu_rate=Decimal(i), output_cu_rate=Decimal(o),
            rate_unit="per_1m_tokens",
        )

    pricing.get_pricing = AsyncMock(side_effect=get_pricing)
    return router, pricing


@pytest.mark.asyncio
async def test_catalog_happy_path_rel_cost_and_alternates():
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing)
    t = cat["tiers"]
    # 4 tiers in curation order, primaries served
    assert list(t.keys()) == ["max", "frontier", "mid", "cheap"]
    assert t["max"]["model"] == "gpt-6-astra" and not t["max"]["served_alternate"]
    assert t["frontier"]["model"] == "claude-opus-5"
    assert t["mid"]["model"] == "gpt-5.6-terra"
    assert t["cheap"]["model"] == "gpt-5.6-luna"
    # rel_cost normalized to max=100 from the blended 3:1 ledger rates
    assert t["max"]["rel_cost"] == 100
    assert t["frontier"]["rel_cost"] == 50    # opus-5 10/20
    assert t["mid"]["rel_cost"] == 28         # 5.625/20
    assert t["cheap"]["rel_cost"] == 11       # 2.25/20
    # alternates exposed with availability + their own rel_cost
    alts = {a["model"]: a for a in t["cheap"]["alternates"]}
    assert alts["gemini-3.5-flash"]["available"] is True
    assert alts["gemini-3.5-flash"]["rel_cost"] == 4    # 0.85/20
    assert alts["claude-haiku-4-5-20251001"]["available"] is True
    # caps derived from registry capabilities (audio false → excluded)
    assert t["max"]["caps"] == ["tools", "vision", "pdf", "streaming"]


@pytest.mark.asyncio
async def test_catalog_tier_failover_marks_alternate_and_lists_primary():
    # e.g. provider suspensions: primary AND first alternate dark
    router, pricing = _mocks(unavailable={"gpt-6-astra", "claude-fable-5"})
    cat = await build_catalog("coding", router, pricing)
    top = cat["tiers"]["max"]
    assert top["model"] == "claude-opus-5"         # second alternate served
    assert top["served_alternate"] is True
    # the dark primary still appears in the ladder, flagged unavailable
    alts = {a["model"]: a for a in top["alternates"]}
    assert alts["gpt-6-astra"]["available"] is False
    assert alts["claude-fable-5"]["available"] is False
    # normalization follows the served top model (opus blended 10)
    assert top["rel_cost"] == 100
    # frontier's own primary is now opus-5 too (also 100 here); its gpt-5.6-sol
    # ALTERNATE is pricier than the served max — rel_cost can exceed 100, honestly.
    fr = cat["tiers"]["frontier"]
    assert fr["model"] == "claude-opus-5" and fr["rel_cost"] == 100
    sol_alt = next(a for a in fr["alternates"] if a["model"] == "gpt-5.6-sol")
    assert sol_alt["rel_cost"] == 112   # 11.25/10=112.5, banker's rounding


@pytest.mark.asyncio
async def test_catalog_unknown_profile_is_none():
    router, pricing = _mocks()
    assert await build_catalog("nope", router, pricing) is None


@pytest.mark.asyncio
async def test_models_map_covers_every_served_model_with_routable_attributes():
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing)
    models = cat["models"]
    # Every served model is present — not just the ones named in a tier ladder.
    assert set(models) == set(RATES)
    m = models["gpt-5.6-terra"]
    assert m["provider"] == "openai"
    assert m["context"] == {"input": 200000, "output": 64000}
    assert m["caps"] == ["tools", "vision", "pdf", "streaming"]
    # Absolute CU cost — the axis callers budget on (no second relative scale).
    assert m["cost"]["input_cu_per_1m"] == 2.5 and m["cost"]["output_cu_per_1m"] == 15.0
    assert cat["schema"] == 3
    assert "coding" in cat["quality_dimensions"]
    assert "flagships" in cat and "benchmark_rankings" in cat


@pytest.mark.asyncio
async def test_uncurated_model_is_still_fully_routable():
    """Onboarding a model makes it selectable with no curation edit. gpt-4o is in
    no tier ladder and has no card/measured/lineage, yet it must still be present
    and described on every objective axis (judgment fields None, non-exclusionary)."""
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing)

    uncurated = cat["models"]["gpt-4o"]
    assert uncurated["card"] is None and uncurated["measured"] is None  # no signal -> unknown
    assert uncurated["lineage"] is None                                 # not placed yet
    assert uncurated["cost"]["input_cu_per_1m"] == 2.5   # ...but fully routable
    assert uncurated["context"]["input"] == 200000
    assert uncurated["caps"] and uncurated["provider"] == "openai"

    for tag, m in cat["models"].items():
        assert m["context"]["input"] is not None, tag
        assert m["cost"] is not None, tag  # objective axes always populated
        # judgment fields are the only ones allowed None (hints, never gates)
        assert m["card"] is None or isinstance(m["card"], dict), tag


@pytest.mark.asyncio
async def test_card_is_deterministic_and_measured_is_orthogonal():
    """The card (summary + benchmarks) is deterministic and NEVER mutated by
    feedback; measured quality is served as a separate, orthogonal field."""
    router, pricing = _mocks()
    feedback = {
        "claude-opus-4-8": {"coding": {"score": 74.0, "samples": 120, "accept_rate": 0.72,
                                       "escalation_rate": 0.15}},
    }
    cat = await build_catalog("coding", router, pricing, feedback_quality=feedback)
    opus = cat["models"]["claude-opus-4-8"]
    # card benchmark stays the published 88.6 despite a lower measured score
    assert opus["card"]["benchmarks"]["coding"]["score"] == 88.6
    assert opus["card"]["benchmarks"]["coding"]["source"] == "SWE-bench Verified"
    assert opus["card"]["summary"]  # deterministic summary present
    # measured is a SEPARATE field, not merged into the card
    assert opus["measured"]["coding"]["score"] == 74.0
    assert opus["measured"]["coding"]["escalation_rate"] == 0.15
    # benchmark_rankings ranks the CARD (deterministic), unaffected by feedback
    top = cat["benchmark_rankings"]["coding"][0]
    assert top["model"] == "claude-fable-5" and top["score"] == 95.0


@pytest.mark.asyncio
async def test_lineage_and_flagship_resolution():
    """Deterministic 'provider best' = rank-1 tier, latest version — and rank
    dominates generation (Google's pro flagship sits below a newer flash number)."""
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing)
    # opus-5 (gen 5.0) is now the latest opus; opus-4-8 (4.8) is superseded
    assert cat["models"]["claude-opus-5"]["lineage"]["latest_in_tier"] is True
    opus48 = cat["models"]["claude-opus-4-8"]["lineage"]
    assert opus48["tier"] == "opus" and opus48["tier_rank"] == 2 and opus48["latest_in_tier"] is False
    # flagships: OpenAI -> newest gen's rank-1; Google -> pro (older gen) NOT flash;
    # Anthropic -> fable (rank 1, above the opus tier) even though opus-5 exists
    assert cat["flagships"]["openai"] == "gpt-6-astra"   # gen 6.0 rank-1 supersedes sol
    assert cat["flagships"]["google"] == "gemini-3.1-pro-preview"
    assert cat["flagships"]["anthropic"] == "claude-fable-5"


@pytest.mark.asyncio
async def test_throughput_is_unknown_without_telemetry_never_guessed():
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing, latency_stats=None)
    tp = cat["models"]["gpt-5.6-terra"]["throughput"]
    assert tp["class"] is None and tp["samples"] == 0
    assert tp["output_tokens_per_sec"] is None and tp["p50_total_ms"] is None


@pytest.mark.asyncio
async def test_throughput_class_ranks_fleet_and_reports_tokens_per_sec():
    router, pricing = _mocks()
    # Lowest ms/token == highest tokens/sec == fastest; ranking is fleet-relative.
    stats = {
        tag: {"samples": 100, "p50_ms": 900, "p95_ms": 1800, "ms_per_output_token": float(i)}
        for i, tag in enumerate(RATES, start=1)
    }
    cat = await build_catalog("coding", router, pricing, latency_stats=stats)
    models = cat["models"]
    classes = {t: m["throughput"]["class"] for t, m in models.items()}
    first, last = list(RATES)[0], list(RATES)[-1]
    assert classes[first] == "fast" and classes[last] == "slow"
    # tokens/sec is the headline and is the inverse of ms/token (i=1 -> 1000/s).
    assert models[first]["throughput"]["output_tokens_per_sec"] == 1000.0
    assert models[first]["throughput"]["samples"] == 100
    assert models[first]["throughput"]["p50_total_ms"] == 900  # total time kept, clearly named


def test_curated_cards_and_lineage_are_well_formed():
    from mlpal_assistants_service.services.catalog import QUALITY_DIMENSIONS
    curated = load_curated()
    for tag, card in (curated.get("cards") or {}).items():
        for dim, entry in (card.get("benchmarks") or {}).items():
            assert dim in QUALITY_DIMENSIONS, f"{tag}: '{dim}' not a quality dimension"
            # A benchmark is a fact from a source, on a date — never bare.
            assert entry.get("score") is not None, f"{tag}/{dim}: missing score"
            assert entry.get("source"), f"{tag}/{dim}: benchmark needs a source"
            assert entry.get("as_of"), f"{tag}/{dim}: benchmark needs an as_of date"
    # provider is joined from the registry at serve time, so it's NOT in the
    # curated file — only the declared structure is (generation/tier/tier_rank).
    for tag, ln in (curated.get("lineage") or {}).items():
        assert ln.get("tier"), tag
        assert isinstance(ln.get("generation"), (int, float)), tag
        assert isinstance(ln.get("tier_rank"), int) and ln["tier_rank"] >= 1, tag


def test_curated_file_shape():
    curated = load_curated()
    prof = curated["profiles"]["coding"]["tiers"]
    for tier, spec in prof.items():
        assert spec["model"], tier
        assert isinstance(spec.get("alternates", []), list)
        assert spec.get("good_for"), tier
        # every model referenced must be priced in the test rate table (keeps
        # the curation and this test honest together)
        for tag in [spec["model"], *spec["alternates"]]:
            assert tag in RATES, f"{tag} missing from RATES — update test with ledger rates"


# ---------------------------------------------------------------------------
# Authoritative routing-ladder guarantees (Option A). These turn the five
# promises the gateway makes to subtask-routing clients into CI gates, so a
# stale curation edit fails the build rather than shipping a superseded ladder.
# ---------------------------------------------------------------------------

_PROVIDER_SCOPE = {"openai", "anthropic", "google"}


@pytest.mark.asyncio
async def test_routing_ladder_is_explicit_and_cost_ordered():
    """G2 (cost-ordered) + the ordinal contract: `routing_ladder` is the tier
    names cheapest -> most capable, and rel_cost is non-decreasing along it, so a
    client maps complexity 0..N onto it directly."""
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing)
    ladder = cat["routing_ladder"]
    assert ladder == ["cheap", "mid", "frontier", "max"]
    tiers = cat["tiers"]
    assert set(ladder) == set(tiers), "ladder must name exactly the served tiers"
    costs = [tiers[name]["rel_cost"] for name in ladder]
    assert costs == sorted(costs), f"ladder not cost-ordered: {costs}"


@pytest.mark.asyncio
async def test_every_tier_entry_carries_caps_and_real_context():
    """G3: each tier the client can land on exposes caps + the real registry
    context window, so a special subtask (vision / long context) is hard-filtered
    against the tier with no guessing and no extra round-trip."""
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing)
    for name, tier in cat["tiers"].items():
        assert tier["model"], name  # every tier resolves in the happy path
        assert isinstance(tier["caps"], list), name
        assert isinstance(tier["context"]["input"], int) and tier["context"]["input"] > 0, name
        assert tier["context"]["output"] is not None, name
        # available alternates carry the same, so a fallback needs no second call
        for alt in tier["alternates"]:
            if alt["available"]:
                assert isinstance(alt["caps"], list), (name, alt["model"])
                assert alt["context"]["input"] is not None, (name, alt["model"])


@pytest.mark.asyncio
async def test_tier_primaries_are_current_generation():
    """G1: every curated tier PRIMARY is the latest generation of its (provider,
    tier) — never a superseded member like a dead tier's last model."""
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing)
    models = cat["models"]
    for tier, spec in load_curated()["profiles"]["coding"]["tiers"].items():
        ln = models[spec["model"]]["lineage"]
        assert ln is not None, f"{tier} primary {spec['model']} has no lineage"
        assert ln["latest_in_tier"] is True, (
            f"{tier} primary {spec['model']} is superseded (not latest_in_tier)"
        )


@pytest.mark.asyncio
async def test_tier_models_are_provider_scoped():
    """G5: every model a client can route to (tier primary + available alternates)
    is OpenAI / Anthropic / Google — never a weaker open-weight / bedrock model."""
    router, pricing = _mocks()
    cat = await build_catalog("coding", router, pricing)
    for name, tier in cat["tiers"].items():
        assert tier["provider"] in _PROVIDER_SCOPE, (name, tier["provider"])
        for alt in tier["alternates"]:
            if alt["available"]:
                assert alt["provider"] in _PROVIDER_SCOPE, (name, alt["model"], alt["provider"])


@pytest.mark.asyncio
async def test_routing_decision_is_stable_across_stats_refresh():
    """G4 (deterministic within TTL): the ladder + tiers depend only on the
    registry and pricing, so a latency/feedback refresh (which moves the `models`
    block) must NOT change the routing decision within the cache window."""
    router, pricing = _mocks()
    base = await build_catalog("coding", router, pricing)
    stats = {tag: {"samples": 50, "p50_ms": 800, "p95_ms": 1600,
                   "ms_per_output_token": float(i)} for i, tag in enumerate(RATES, 1)}
    feedback = {"gpt-5.6-luna": {"coding": {"score": 40.0, "samples": 99,
                                            "accept_rate": 0.4, "escalation_rate": 0.3}}}
    perturbed = await build_catalog("coding", router, pricing,
                                    latency_stats=stats, feedback_quality=feedback)
    assert base["routing_ladder"] == perturbed["routing_ladder"]
    assert base["tiers"] == perturbed["tiers"]
    # ...while the models block DID move, proving the perturbation was real
    assert base["models"] != perturbed["models"]
