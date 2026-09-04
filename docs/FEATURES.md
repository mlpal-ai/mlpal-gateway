# MLPal Gateway — Master Feature List

One codebase, two deployments. ✅ = present · ⚙️ = present, different default/backing · — = absent by design.
Updated 2026-09-04.

## Inference surfaces

| Feature | Managed | Self-hosted (OSS) |
|---|---|---|
| `POST /v1/messages` — universal Anthropic wire (anthropic native; openai/google via translating edge) | ✅ | ✅ |
| `POST /v1/chat/completions` (+ `/stream`) — OpenAI wire | ✅ | ✅ |
| `/v1/embeddings`, `/v1/images/generations`, `/v1/audio/speech`, `/v1/audio/transcriptions` | ✅ | ✅ |
| SSE streaming on both wires (heartbeats, byte-faithful passthrough) | ✅ | ✅ |
| Tools / structured output / MCP pass-through (gateway, not middleware) | ✅ | ✅ |
| **Prompt caching on both wires**: Anthropic `cache_control` breakpoints pass through natively on `/v1/messages` and via `cache_control` on OpenAI-wire messages (max 4/request); OpenAI + Gemini implicit caching surfaces as `cached_tokens` (Gemini needs a ≥4096-token prefix) | ✅ | ✅ |
| `fallback_models` (up to 3, tried on retriable serving failures, billed as-served) + `model_kwargs` (provider-native params, unknown keys rejected) on the OpenAI wire | ✅ | ✅ |
| `POST /v1/messages/count_tokens` passthrough | ✅ | ✅ |
| **Async image jobs**: `wait:false` returns a job id, poll `GET /v1/images/jobs/{id}`; image quality tiers (`lite`/standard/`hd`) priced per tier | ✅ | ✅ |
| `X-MLPal-Compute-Units` header on native responses | ✅ | ✅ |
| Deprecated `/v2/*` aliases (yodex transition) | ✅ until drained | — (`/v2` reserved) |
| Bedrock-mantle passthrough at `/mantle/v1/messages` | opt-in flag | — (module not shipped) |

## Models, catalog, routing

| Feature | Managed | Self-hosted |
|---|---|---|
| Curated registry (82 models: OpenAI, Anthropic, Google, Bedrock open-weights; deprecations carry a successor + shutdown date) | ✅ | ✅ same feed |
| Declarative catalog feed + reconcile (insert / update / **soft-retire**, provenance) | manual/CI run | ✅ every boot |
| Effective-dated pass-through pricing (markup 1.00, 1 CU = $10) | ✅ | ✅ |
| **Serving backends**: the same model served via first-party API, Bedrock, Vertex, or Azure Foundry in priority order (`MLPAL_<FAMILY>_BACKENDS`); third-party adapters via pip entry point | ✅ | ✅ |
| **Connections (BYOK / BYOM)**: per-account provider keys and custom OpenAI-compatible endpoints; connection-served requests bill zero CU (tokens still metered) | ✅ | ✅ |
| Router tags `mlpal` / `mlpal-flash` / `mlpal-lite` — **availability-aware**, feed-driven candidate lists | ✅ | ✅ (single-provider boxes resolve) |
| Curated catalog API `GET /v1/catalog` (tier ladder, live availability, `no-cache`+ETag) | ✅ | ✅ |
| Model cards + lineage (console) | ✅ | ✅ |
| Admin pause/retire per model | ✅ | ✅ |
| Provider adapters activate per key present | ✅ (all four) | ✅ (whatever keys you set) |
| Per-provider circuit breakers | ✅ | ✅ |

## Keys, policy, budgets

| Feature | Managed | Self-hosted |
|---|---|---|
| Key mint/revoke, hashed secrets, one-time display | ✅ | ✅ |
| Key management principal | Cognito JWT (frontend) **or** admin key | local admin key (bootstrap-printed) |
| Permission scopes (`messages`, `chat`, `embedding`, …, `admin`, `*`) | ✅ | ✅ |
| Model policy allow/deny globs (deny wins; follows router-tag resolution) | ✅ | ✅ |
| Spend budgets: **multiple calendar windows** (daily/weekly/monthly/lifetime), `cu`/`usd` at fixed peg, pre-flight enforcement, never cuts a running request, Redis counters re-seeded from `usage_logs`, fail-open | ✅ | ✅ |
| Rate limiting (Redis, per-tier) | ✅ | ✅ |
| Identical admission on ALL inference surfaces | ✅ | ✅ |
| **Per-key payload-capture policy** (`capture: inherit / on / off`) — a key owner opts its own traces in or out of body capture regardless of the box default | ✅ | ✅ |

## Billing & usage

| Feature | Managed | Self-hosted |
|---|---|---|
| Pass-through CU metering (single figure, no markup, no meter) | ✅ | ✅ |
| **Cache-aware metering reproduces list price exactly**: cache reads at the provider's tier (per-model `cache_read_rate`, e.g. Claude Fable 5.1 $0.25/MTok; else 0.10× OpenAI/Anthropic, 0.25× Gemini), Anthropic cache writes at 1.25× (5m) / 2× (1h); one usage convention on the OpenAI wire (`input_tokens` = whole prompt, `cached_tokens` + `cache_write_tokens` subsets) | ✅ | ✅ |
| `usage_logs` per request: tokens, CU, latency, status, error, surface tag | ✅ | ✅ |
| `GET /v1/usage/summary` + daily buckets | ✅ | ✅ |
| Wallet debits (payments service) + debit-retry worker + billing gate | ✅ | — (local gate: allow-all, no callouts; spend control = per-key budgets) |
| Platform fee (free up to 50 CU or 300M tokens/mo per account, cache reads excluded; $100/mo flat beyond) | mechanism shipped, enforcement off | n/a (self-hosted is free) |

## Observability

| Feature | Managed | Self-hosted |
|---|---|---|
| **Trace records** (every request, all metadata) — the Traces screen/API | ✅ always | ✅ always |
| **Payload capture** (request/response bodies: zlib, size-capped, retention-purged, runtime toggle) | ⚙️ default **OFF** (privacy: metrics always, bodies never) | ⚙️ default **ON** |
| `/admin/v1/*` control plane (traces+payloads, providers w/ live probes, models, router chains, config, capture toggle, budgets burn, latency stats, cache ops) | ✅ | ✅ |
| Structured logs w/ trace correlation | ✅ | ✅ |
| Metrics / tracing backends | CloudWatch EMF + X-Ray | console/log emitters (local mode) |
| `/health` + `/health/ready` with provider probes | ✅ | ✅ |

## Admin console (UI — ships in the OSS repo, works against either deployment with an admin key)

Overview dashboard · Traces (live tail, key/status/window filters, payload viewer, copy-as-curl) · Playground · Providers (live health, status dots) · Keys (scoped mint with focus-hints, stacked budgets, per-key stats + burn bars) · Models (registry cards, pricing, search) · Routing (router tags + curated tier ladder) · Usage charts · Settings (capture toggle, config view) · dark/light · collapsible sidebar · MLPal design system.

## Auth & ops

| Feature | Managed | Self-hosted |
|---|---|---|
| Auth backend | auth-service / Cognito | `local` (admin key) |
| Deploy | EKS, push-to-main CI | `docker compose up` (one command; seed prints admin key, catalog reconciles on boot) |
| Asset storage (image/audio outputs → presigned URLs) | S3 | optional (set AWS creds; degrades gracefully) |
| Migrations | manual `alembic upgrade head` | migrate container on boot |
| Secret hygiene | k8s secrets | gitignored `.env`; publish-semantics secret scanner |
| E2E verification | `the unit suite (`pytest tests/unit`)` | same script |

## SDK

`mlpal-assistants` (Python): v2 `MLPal`/`AsyncMLPal` — native messages (+streaming), catalog, feedback, usage, key admin, `Message.compute_units`; v1 `Assistant` facade for OpenAI-wire modalities. 0.2.0 on private PyPI; public-PyPI trusted publishing staged. Deployment-agnostic via `base_url`.

## The parity rule

Everything is one codebase with a composition root (`api/mounting.py`) and seams
(auth, billing, capture default, mantle, `/v2` aliases). `the OSS build pipeline`
derives the OSS tree; drift is structural, not accidental. If a feature lands in
one deployment only, it must be a **seam with a documented default** — never a fork.
