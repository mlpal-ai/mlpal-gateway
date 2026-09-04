<div align="center">

# MLPal Gateway

**One API for every model provider.**<br>
Anthropic, OpenAI, Google, and AWS Bedrock ship built in; new providers are an adapter away.<br>
Self-hosted. Model routing, per-request cost metering, per-key access control, admin console.

[![PyPI](https://img.shields.io/pypi/v/mlpal-gateway)](https://pypi.org/project/mlpal-gateway/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/mlpal-ai/mlpal-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/mlpal-ai/mlpal-gateway/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/mlpal-gateway)](https://pypi.org/project/mlpal-gateway/)

</div>

<br>

![Gateway console](https://raw.githubusercontent.com/mlpal-ai/mlpal-gateway/main/docs/console.png)

<br>

## Quick start

```bash
cp .env.oss.example .env      # add whichever provider keys you have
docker compose up             # Postgres + Redis + gateway + admin console
```

| | |
|---|---|
| Gateway | `http://localhost:8000` — Swagger at `/docs` |
| Admin console | `http://localhost:8080` |
| Bootstrap admin key | printed once: `docker compose logs seed` — lost it? mint another: `docker compose run --rm seed python scripts/bootstrap_admin_key.py --name admin-2 --force` |

Adapters activate based on which provider keys you set. `GET /v1/models` lists
only what your box can serve; a single-provider deployment works without any
extra configuration.

Don't want to run infrastructure? The managed deployment of this same codebase
runs at **[mlpal.ai](https://mlpal.ai)** — create a key and point the same
SDK/CLI at `https://models.mlpal.ai` instead of localhost. Pricing is simple:
**routing is free up to 50 CU or 300M tokens each month** (cache reads don't
count), then **$100 flat per month**; your tokens are always billed at provider
list price, passed through with no markup, either way.

Prebuilt multi-arch images are on GHCR if you'd rather not build from source:
`ghcr.io/mlpal-ai/mlpal-gateway` and `ghcr.io/mlpal-ai/mlpal-gateway-console`
(tags: `latest`, version, commit SHA).

## Calling it

Anthropic wire, any provider's model:

```bash
curl http://localhost:8000/v1/messages \
  -H "Authorization: Bearer $MLPAL_API_KEY" -H "content-type: application/json" \
  -d '{"model":"mlpal","max_tokens":512,"messages":[{"role":"user","content":"hi"}]}'
```

Existing OpenAI SDK code — point it at `http://localhost:8000/v1`. Python SDK —
[`pip install mlpal-assistants`](https://pypi.org/project/mlpal-assistants/):

```python
from mlpal_assistants import MLPal

client = MLPal(base_url="http://localhost:8000")   # key from MLPAL_API_KEY
msg = client.messages.create(
    model="mlpal", max_tokens=512,
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.text, msg.compute_units)
```

## What you get

- **One wire format for every provider.** `POST /v1/messages` speaks the
  Anthropic Messages format for all served models; the gateway translates to
  each provider's native API and relays the SSE stream without re-chunking.
  OpenAI-compatible endpoints (`/v1/chat/completions`, embeddings, images,
  audio) serve existing OpenAI SDK code unchanged.
- **Router tags.** `"model": "mlpal"` resolves to the best model your
  deployment serves, walking a curated candidate list that spans providers.
  Model retired? Provider key missing? Resolution falls through to the next
  candidate. Client code never changes.
- **Cost on every response.** Requests are metered in compute units
  (1 CU = $10 of provider list price, no markup) and returned in an
  `X-MLPal-Compute-Units` header. The meter reproduces provider list pricing
  exactly, including prompt-cache discounts.
- **Per-key control.** Model-policy globs (`allow: ["claude-*"]`), multi-window
  spend budgets, permissions — all enforced at admission, before the provider
  call. A running stream is never cut.
- **Observability.** Per-key cache hit rate, latency p50/p95,
  time-to-first-token, request traces, and optional payload capture
  (zlib-compressed, runtime toggle — your box, your data).
- **Provider semantics preserved.** Prompt caching (`cache_control`, on both
  wires), tools, structured output, and MCP config pass through untouched.
- **Serve through the cloud you already have.** The same models can be served
  via Azure AI Foundry, Vertex AI, or AWS Bedrock instead of (or in priority
  order with) the provider's own API — `MLPAL_ANTHROPIC_BACKENDS=bedrock,first_party`
  and Claude requests ride your AWS account; same IDs, same wire, zero client
  changes. `scripts/probe_backends.py` live-verifies what your credentials can
  serve and prints the exact config. See [docs/BACKENDS.md](docs/BACKENDS.md);
  third-party adapters plug in via a pip entry point
  ([docs/ADAPTERS.md](docs/ADAPTERS.md)).
- **Catalog that keeps itself current (opt-in, free).** By default your box
  ships with the bundled model catalog, frozen at this version — fully
  functional offline, zero calls home. Subscribe to the hosted feed (Models
  page, or `MLPAL_CATALOG_FEED=hosted`) and the gateway pulls the curated
  catalog daily — new models appear and retired ones are absorbed with no
  upgrade. Subscribing requires a free [mlpal.ai](https://mlpal.ai) account
  key purely as identity — we recommend a dedicated key with **no
  permissions** (it can authenticate the feed and nothing else; feed pulls
  are never billed). See [Telemetry](#telemetry).

## Measured

Gateway overhead isolated against a zero-latency fake upstream, so provider
variance can't hide anything — MLPal with its **full admission pipeline**
(auth, rate limit, billing, model policy, budgets, metering, capture) against
LiteLLM in both its bare mode and a production configuration with
database-backed virtual keys, budgets, and spend tracking (N=100 per system;
methodology in the [technical report](paper/mlpal-gateway-technical-report.pdf),
raw data and harness in [`paper/bench/`](paper/bench/)):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mlpal-ai/mlpal-gateway/main/docs/fig-overhead-dark.svg">
  <img alt="Gateway overhead benchmark" src="https://raw.githubusercontent.com/mlpal-ai/mlpal-gateway/main/docs/fig-overhead-light.svg">
</picture>

**+8.5 ms with everything on — less than a bare proxy checks one static key
for, and with the tightest tail (p95 33 ms vs 58/41 ms).** Admission-time
governance is computationally free. Provider semantics survive the hop too:
a 22k-token cached prefix passes through byte-faithfully and metered
0.002756 CU on write, 0.000224 CU on read (12.3×) — matching Anthropic's
list price to five decimals. The managed deployment of this same codebase,
measured the same night from the same client, served claude-haiku-4.5 at
642 ms median TTFT vs OpenRouter's 898 ms (report §5.3–5.4).

## Why a curated catalog

Production models retire on roughly a 12-month cycle now — from the providers'
own deprecation ledgers:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mlpal-ai/mlpal-gateway/main/docs/fig-lifetimes-dark.svg">
  <img alt="Model launch-to-retirement spans" src="https://raw.githubusercontent.com/mlpal-ai/mlpal-gateway/main/docs/fig-lifetimes-light.svg">
</picture>

Serving a model well — valid parameter ranges, per-model cache minimums,
reasoning budgets, provider quirks — is per-model engineering that does not
scale to a 1,600-entry catalog. This gateway serves a curated set (~75 models)
kept current by a data feed, and router tags absorb every retirement above
server-side. You can always pin any explicit model tag or register your own
adapter.

The full argument with benchmarks and sources:
**[Curation Over Breadth](paper/mlpal-gateway-technical-report.pdf)** ·
feature matrix vs. OpenRouter / LiteLLM / Portkey:
[docs/POSITIONING.md](docs/POSITIONING.md).

## Router tags vs. catalog

Two ways to use the curated set — they differ in who picks the model:

| | Router tags — `mlpal`, `mlpal-flash`, `mlpal-lite` | Catalog — `GET /v1/catalog` |
|---|---|---|
| Who decides | The gateway: tag resolves to the best served model for the operation | Your client: a ranked list with tiers, capabilities, per-token rates |
| Use when | You want a good default and zero model-name maintenance | You are writing routing logic (agents route sub-tasks this way) |
| One-provider box | Falls through to whatever your key serves | Unserved candidates are marked |

Both are driven by the same feed (`catalog/*.json`) and update as data, not
code.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /v1/messages` | Anthropic-wire inference, all providers, streaming SSE |
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `/v1/embeddings` · `/v1/images/generations` · `/v1/audio/*` | OpenAI-compatible modalities |
| `GET /v1/models` | Models this deployment serves |
| `GET /v1/catalog` | Ranked catalog: tiers, capabilities, rates |
| `POST /v1/feedback` | Outcome feedback for routing scores |
| `GET /v1/usage/*` · `/v1/keys/*` | Self-scoped usage, traces, per-key stats |
| `/admin/v1/*` | Keys, policies, budgets, capture, routing |

Details: [docs/API_SURFACE.md](docs/API_SURFACE.md).

## Point the Anthropic SDK or Claude Code at it

`/v1/messages` speaks the native Anthropic wire and accepts the SDK's default
`x-api-key` header — so it really is a drop-in:

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000", api_key="mlpal_sk_...")
```

For Claude Code, set `ANTHROPIC_BASE_URL=http://localhost:8000` and
`ANTHROPIC_API_KEY=mlpal_sk_...` — every request is then routed, priced, and
traced by your gateway.

## Use it with a coding agent

[Yodex](https://github.com/mlpal-ai/yodex) is a coding CLI built on this
gateway — it speaks the Anthropic wire and uses `GET /v1/catalog` to route
sub-tasks to cheaper models (~10× lower sub-agent cost in its
[benchmarks](https://github.com/mlpal-ai/yodex#benchmarks)):

```bash
npm install -g @mlpal/yodex
export YODEX_GATEWAY_URL=http://localhost:8000
export YODEX_API_KEY=mlpal_sk_...    # minted in the console
yodex "fix the failing test"
```

## Repository layout

```
src/                 # FastAPI gateway: adapters, services, api, seams
console/             # admin UI (React + Vite): keys, traces, catalog, usage
docker-compose.yaml  # one-command local deployment
alembic/             # database migrations
paper/               # technical report + benchmark harness + raw results
enterprise/          # commercial add-ons (separate license, NOT Apache)
docs/                # API surface, positioning, figures
```

Auth and billing sit behind composition-root seams (`api/mounting.py`); the
defaults (`MLPAL_AUTH_BACKEND=local`, `MLPAL_BILLING_BACKEND=local`) run fully
standalone with no external dependencies. `src/` never imports from
`enterprise/`.

## Telemetry

None by default. If you opt in to the hosted catalog feed, each daily pull
sends exactly three things to `models.mlpal.ai`: the mlpal.ai API key you
subscribed with (identity — links the install to your account), a random
per-install UUID, and your gateway version. Feed pulls are free and never
metered. No provider keys, no payloads, no usage data, ever. Stay in
`bundled` mode (the default) and the gateway makes zero calls home.

## License and contact

Apache-2.0, except the `enterprise/` directory (commercial — see
[`enterprise/LICENSE`](enterprise/LICENSE)). Contributions welcome:
[CONTRIBUTING.md](CONTRIBUTING.md) · security and everything else:
**contact@mlpal.ai**
