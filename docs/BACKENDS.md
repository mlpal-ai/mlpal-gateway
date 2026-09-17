# Serving backends: Azure, Vertex AI, and Bedrock

The gateway serves each model **family** (OpenAI, Google, Anthropic) through
one or more *serving backends*. `first_party` — the provider's own API — is
the default and needs nothing but the provider key. Cloud backends let you
serve the **same models, same IDs, same wire** through the cloud you already
have credentials, quota, or compliance requirements on.

| Family    | Backends                                   | Cloud credential                       |
| --------- | ------------------------------------------ | -------------------------------------- |
| openai    | `first_party`, `azure`                     | Azure AI Foundry resource key          |
| google    | `first_party`, `vertex`                    | GCP ADC (service account / user creds) |
| anthropic | `first_party`, `bedrock`, `vertex`, `azure`| AWS creds / GCP ADC / same Azure key   |

The `bedrock` *catalog* family (open-weight models) is unrelated — it keeps
working exactly as before via `MLPAL_ENABLE_BEDROCK`.

## Priority

`MLPAL_<FAMILY>_BACKENDS` is a priority-ordered list. For every model, the
first backend that is **configured** and **serves that model** takes the call:

```env
# Claude via Bedrock when possible, Anthropic API otherwise:
MLPAL_ANTHROPIC_BACKENDS=bedrock,first_party

# GPT models via your Azure deployment only:
MLPAL_OPENAI_BACKENDS=azure

# Gemini via Vertex first:
MLPAL_GOOGLE_BACKENDS=vertex,first_party
```

Resolution happens once per model and is cached in-process — the request
hot path pays a dict lookup, nothing more. Models no backend serves show as
**not served** in the console and on `/v1/models` (`"serving_backend": null`),
and requests to them fail fast with a clear error instead of a provider 401.

## Azure (OpenAI family)

One AI Foundry resource serves the OpenAI-compatible v1 surface:

```bash
az cognitiveservices account create -n <res> -g <rg> -l eastus2 \
  --kind AIServices --sku S0 --custom-domain <res> --yes
az cognitiveservices account deployment create -g <rg> -n <res> \
  --deployment-name gpt-5-mini --model-format OpenAI \
  --model-name gpt-5-mini --model-version 2025-08-07 \
  --sku-name GlobalStandard --sku-capacity 50   # 1 unit = 1,000 TPM
```

```env
MLPAL_OPENAI_BACKENDS=azure,first_party
MLPAL_AZURE_OPENAI_ENDPOINT=https://<res>.services.ai.azure.com
MLPAL_AZURE_OPENAI_API_KEY=...
```

Azure addresses models by **deployment name**. Name deployments after the
model IDs (deployment `gpt-5-mini` for model `gpt-5-mini`) and everything
passes through unchanged. Non-identity names — or exact "which models does
Azure actually have" console display — use the map:

```env
MLPAL_AZURE_DEPLOYMENTS='{"gpt-5-mini": "my-mini-deployment"}'
```

With the map set, Azure claims only mapped models and the rest fall through
to the next backend in priority. Without it, Azure claims the whole family.
Fresh subscriptions start with zero quota for most models — check
`az cognitiveservices usage list -l <region>` and request increases in the
portal's Quota blade.

### Claude on Microsoft Foundry (same resource)

The SAME AIServices resource serves Claude natively at `<endpoint>/anthropic`
— so Azure credits cover Claude too, with no extra credentials:

```env
MLPAL_ANTHROPIC_BACKENDS=azure,first_party
# optional exact map (identity convention otherwise):
MLPAL_AZURE_ANTHROPIC_DEPLOYMENTS='{"claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001"}'
```

Deploying Claude needs the Anthropic marketplace attestation, which the
Foundry portal doesn't always surface — use the ARM API (auto-signs the
Anthropic terms on your behalf; set values matching your real organization):

```bash
az rest --method put \
  --url "https://management.azure.com/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<res>/deployments/claude-haiku-4-5-20251001?api-version=2025-10-01-preview" \
  --body '{"sku": {"name": "GlobalStandard", "capacity": 50},
           "properties": {"model": {"format": "Anthropic", "name": "claude-haiku-4-5", "version": "20251001"},
                          "modelProviderData": {"industry": "technology", "organizationName": "<LegalName>", "countryCode": "US"}}}'
```

`industry` must be lowercase; Claude on Foundry requires the resource in
East US 2 or Sweden Central, and fresh subscriptions start at zero Claude
quota (request in the Quota blade). Claude gets the byte-faithful native
`/v1/messages` path; the OpenAI-wire endpoints work via translation.

## Vertex AI (Google + Anthropic families)

One service account (or your user ADC) serves both Gemini and Claude:

```bash
gcloud services enable aiplatform.googleapis.com
gcloud iam service-accounts create mlpal-vertex
gcloud projects add-iam-policy-binding <project> \
  --member="serviceAccount:mlpal-vertex@<project>.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
gcloud iam service-accounts keys create sa.json \
  --iam-account="mlpal-vertex@<project>.iam.gserviceaccount.com"
# If org policy blocks SA keys: `gcloud auth application-default login` instead.
```

```env
GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
MLPAL_VERTEX_PROJECT=<project>
MLPAL_VERTEX_LOCATION=global          # best availability; Claude at first-party price
MLPAL_GOOGLE_BACKENDS=vertex,first_party
```

Gemini model IDs are identical on Vertex — no mapping. Claude on Vertex
additionally needs per-model **Model Garden enablement** (console consent) and
often a quota-increase request on fresh projects; once enabled, list the
models explicitly (IDs are dashed and dateless for ≥4.6 generations):

```env
MLPAL_ANTHROPIC_BACKENDS=vertex,first_party
MLPAL_VERTEX_ANTHROPIC_MODELS='{"claude-sonnet-4-6": "claude-sonnet-4-6"}'
```

## Bedrock (Anthropic family)

Uses your existing AWS credentials (env/IRSA). Claude model IDs on Bedrock
are inference-profile IDs that don't follow one rule, so the map is explicit
and **generated by live verification**, never guessed:

```bash
uv run python scripts/probe_backends.py bedrock
```

makes a 1-token call per candidate and prints exactly:

```env
MLPAL_ANTHROPIC_BACKENDS=bedrock,first_party
MLPAL_BEDROCK_ANTHROPIC_MODELS='{"claude-opus-5": "global.anthropic.claude-opus-5", ...}'
MLPAL_BEDROCK_MANTLE_MODELS='["claude-opus-5", "claude-sonnet-5", ...]'
```

Two Bedrock paths exist and their model populations differ:

- **Native** (`MLPAL_BEDROCK_MANTLE_MODELS`): the Anthropic-wire Bedrock
  endpoint — byte-faithful `/v1/messages` passthrough, newest model
  generations only.
- **Adapter** (`MLPAL_BEDROCK_ANTHROPIC_MODELS`): the Bedrock runtime SDK —
  wider model coverage; `/v1/messages` still works via translation, plus all
  OpenAI-wire endpoints.

The gateway walks `MLPAL_ANTHROPIC_BACKENDS` in order and takes the FIRST
native backend that serves the model (mantle: its allowlist; first_party:
everything), so `bedrock,first_party` with an empty mantle list keeps the
Anthropic wire byte-faithful on first-party while the OpenAI wire's adapter
path serves the mapped models from Bedrock. `count_tokens` always uses a
backend that can count (Bedrock cannot). Probe with the region your account
has model access in (`MLPAL_BEDROCK_MANTLE_REGION`), and prefer `global.`
inference profiles: they are priced at Anthropic list; `us.` profiles cost
10% more. Probe once, paste both lines.

## Changing priorities at runtime (no restart)

Backend **priorities** are runtime-overridable: the console's Settings page
("Serving backends" card) or `PUT /admin/v1/settings/<family>_backends` change
them live — persisted in the DB, propagated to every worker via pub/sub within
about a second, surviving restarts. `value: null` clears the override back to
the env value. Precedence: runtime override > env > default.

Cloud **credentials** stay env-only on purpose (they're secrets and belong in
your deployment, not a database) — so the flow is: configure credentials once
in env, then move traffic between backends live whenever you need to (e.g.
shift Claude to Bedrock during a first-party incident, then reset).

## Automatic failover between backends

A backend outage never needs a config change. When the backend that serves a
model fails with a serving fault — a 5xx/529, a connection or timeout error,
a provider 429, or an open circuit breaker — the gateway retries the **same
model** once on the next backend in that family's priority list, before any
client-supplied `fallback_models`. So with `bedrock,first_party`, a Bedrock
incident quietly moves Claude to the Anthropic API per request and moves it
back as soon as Bedrock recovers.

Rules, on both wires:

- Never on a 4xx (the request would fail identically elsewhere).
- Never after a stream has emitted its first bytes — an open stream is
  committed to its backend.
- At most one hop per request; the hop's own failure is the one you see.
- Requests served through a tenant connection (your own provider key) are
  never moved onto the gateway's keys.
- The Anthropic wire stays byte-faithful: a native Bedrock request fails over
  to native first-party, never onto the translating path.

Circuit breakers are per backend (`anthropic:bedrock`, `anthropic:first_party`);
after 5 consecutive failures a backend is skipped for 30 s without paying its
timeout, then probed again. Attribution: `metadata.backend_fallback_from` on
`/v1/chat/completions`, the `X-MLPal-Backend-Fallback-From` header on
`/v1/messages`, and `backend_fallback_from` / `serving_backend` in every usage
row — the failed attempt is recorded under its own backend with its error.
`MLPAL_BACKEND_FAILOVER=false` turns the hop off (breakers stay per backend).

## Verifying a box

`scripts/probe_backends.py {bedrock,vertex,azure}` live-verifies each leg
with your credentials and prints ready-to-paste env. The console's Providers
page shows which backend serves each family and the priority order; the
Models page marks anything unserved.
