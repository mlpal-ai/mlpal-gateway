# API surface

The gateway exposes **one inference API in two wire formats** plus a **separate
management surface**. App teams standardize on the MLPal SDK (native wire);
existing OpenAI/Anthropic code can point its `base_url` here and work unchanged.
The wire you call is decoupled from the upstream provider — every wire can reach
every configured model, and the response comes back in the shape of the wire you
called, not the model that ran.

## Versioning policy

**A path version denotes a contract revision, never a wire dialect.** The whole
data plane lives under `/v1`; the two wires are told apart by **resource**, not
by version number:

- `POST /v1/messages` — the Anthropic-wire (native) endpoint.
- `POST /v1/chat/completions` (+ `/v1/embeddings`, `/v1/images/generations`,
  `/v1/audio/*`) — the OpenAI-wire. These modalities have no representation in
  the Anthropic Messages wire, so `/v1` is their natural and only home — they
  are **not** "legacy".

`/v2` is **reserved** for a genuine future breaking revision of the whole
surface. The "recommended" surface is signaled by docs + the SDK default, not by
inflating a version number.

The control plane (`/admin/v1/*`) is versioned **independently** of the data
plane, because it evolves on its own cadence.

## Inference (data plane) — `/v1`

| Surface | Path | For |
|---|---|---|
| **Native (Anthropic-wire)** | `POST /v1/messages` (+ `GET /v1/messages/models`, `POST /v1/messages/count_tokens`) | The canonical inference surface; what the MLPal SDK uses. Also a drop-in for Anthropic-SDK / Claude-Code users. |
| **OpenAI-compat** | `POST /v1/chat/completions` (+ `/stream`), `/v1/embeddings`, `/v1/images/generations` (+ `GET /v1/images/jobs/{id}` for `wait:false`), `/v1/audio/*` | The zero-friction migration door — point an existing OpenAI SDK at this `base_url`. |
| Curated catalog | `GET /v1/catalog`, `GET /v1/models` | Routable model attributes / tiers, filtered to the models this key can serve. |
| Feedback | `POST /v1/feedback` | Outcome signal that refines curation. |
| Account usage | `GET /v1/usage/summary`, `GET /v1/usage/daily` | This account's own usage. |
| Connections (BYOK / BYOM) | `/v1/connections`, `/v1/connections/{id}/verify`, `/v1/connections/{id}/models` | Bring your own provider keys or an OpenAI-compatible endpoint; requests served on a connection bill zero CU. |

## Management (control plane) — `/admin/v1/*`

The canonical home for administration; the SDK targets it via `client.admin.*`.

| Path | Purpose |
|---|---|
| `POST/GET/GET/PATCH/DELETE /admin/v1/keys[...]` | Issue and manage API keys + their **model_policy** and **spend budgets** |
| `POST /admin/v1/keys/cde` | Mint a scoped service key |
| `GET /admin/v1/keys/{id}/usage[/daily]` | Per-key usage |
| `POST /admin/v1/cache/invalidate`, `GET /admin/v1/cache/status` | Cache ops |
| `POST /admin/v1/models/{tag}/pause` | Pause/unpause a model |

**Auth:** management resolves the caller via the auth seam — Cognito JWT
(managed) or an `admin`-scoped API key (self-hosted `MLPAL_AUTH_BACKEND=local`).

Legacy `/v1/keys/*` and `/v1/admin/*` remain mounted for existing managed
clients; new integrations use `/admin/v1/*`.

## The `messages` seam (managed vs self-hosted)

`/v1/messages` has two possible backends, chosen at the composition root by
`MLPAL_ENABLE_BEDROCK_MANTLE_MESSAGES` (see `api/mounting.py`):

- **Self-hosted (`false`):** the universal translating core **is**
  `/v1/messages`. Nothing is mounted under `/v2`. This is the clean OSS surface.
- **Managed (`true`, default):** the Bedrock-mantle Anthropic passthrough owns
  `/v1/messages` (fronts Claude Code via `ANTHROPIC_BASE_URL`), and the universal
  core is served at `/v2/messages` **transitionally**, with `/v1/catalog` +
  `/v1/feedback` also keeping `/v2` aliases. The managed surface is therefore a
  strict **superset** of the historical one — every path that answered before
  still answers identically.

**Follow-up (managed):** once Claude Code is repointed off the Bedrock-mantle
`/v1/messages`, the universal core can own `/v1/messages` in managed too, the
transitional `/v2/*` mounts drop, and the SDK's `/v1/messages` becomes portable
across managed and self-hosted.

## API reference

Swagger UI (`/docs`) and ReDoc (`/redoc`) are served when `MLPAL_DEBUG=true`
(on by default in the self-hosted docker-compose); `GET /openapi.json` is always
available.
