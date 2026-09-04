# Publish checklist

This repo is assembled from the private `mlpal-assistants-service` by
`scripts/build-oss.sh` (in that repo). It is **local and private** — do not make
it public until the boxes below are checked.

## How it's built

- Source is synced by `scripts/build-oss.sh`, which **excludes** managed-only
  files (the Bedrock-mantle passthrough `api/v1/messages.py`) and **scrubs**
  internal config defaults (backend/payments URLs, Cognito pool/client IDs) and
  flips composition-root defaults to self-hosted (local auth/billing, no
  Bedrock-mantle, Bedrock provider off).
- `scripts/oss-secret-scan.sh` must pass (no secrets / internal identifiers).
- Scaffolding here (LICENSE, README, docs, console, docker-compose, `.github`,
  `enterprise/`) is authored directly and is **not** overwritten by the sync.

## Verified so far

- ✅ Public package imports with **no env flags** (defaults to self-hosted): 75
  routes (2026-09-04 sync), `/v1/messages` = universal core, **zero `/v2`**,
  zero `/mantle`.
- ✅ Secret scan clean (RDS/Cognito/account-IDs/provider-keys/`.env` all absent).
- ✅ `docker compose config` valid (gateway + console); clean-box brought up
  end-to-end previously in the source repo.

## Before flipping to public

- [ ] Run `scripts/build-oss.sh` + `scripts/oss-secret-scan.sh` once more on a
      clean checkout; scan must be green.
- [ ] `uv sync --all-extras && uv run pytest tests/unit/ -q` **inside this repo**
      (CI does this; run locally once to be sure deps resolve standalone).
- [ ] `cd console && npm ci && npm test && npm run build`.
- [ ] Review `pyproject.toml` for OSS: name/description/URLs/authors, and drop
      any managed-only dependency or entry point.
- [ ] Skim `README.md`, `docs/POSITIONING.md`, `docs/API_SURFACE.md` for tone +
      correctness (they're the public front door).
- [ ] Decide the GitHub org/repo name and create it **private** first
      (`gh repo create --private`), push, let CI go green, then flip public.

## Known follow-ups (safe to ship without, but worth doing)

- **Inert managed code still present** (no secrets, disabled by default):
  `repositories/billing_repository.py`, `services/debit_retry_worker.py`,
  `core/auth.py` (Cognito). They're unused under the OSS local backends; trim or
  move behind `enterprise/` later for a cleaner tree.
- **Package name** is still `mlpal_assistants_service` (repo is `mlpal-gateway`).
  A rename to `mlpal_gateway` is cosmetic but touches every import — do it as its
  own change.
- **Alembic** may create payments-owned tables; `scripts/init-db-oss.sql` stubs
  the user schema so migrations apply. Consider trimming payments migrations.
- **SDK** lives in its own repo (`mlpal-assistants-sdk`), published separately.
