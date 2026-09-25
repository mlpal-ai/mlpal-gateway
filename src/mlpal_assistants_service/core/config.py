"""Application configuration using pydantic-settings."""

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    for dist in ("mlpal-gateway", "mlpal-assistants-service"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    # Container images run from source without installing the project — read
    # the version straight from the bundled pyproject (same fallback as
    # catalog_feed.gateway_version).
    try:
        import pathlib
        import tomllib

        for parent in pathlib.Path(__file__).resolve().parents:
            pp = parent / "pyproject.toml"
            if pp.exists():
                return tomllib.loads(pp.read_text())["project"]["version"]
    except Exception:  # noqa: BLE001
        pass
    return "0.0.0-dev"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "MLpal Assistants Service"
    app_version: str = Field(
        # Single source of truth is pyproject/package metadata — a hardcoded
        # literal here shipped "0.1.0" in /health, OpenAPI, and OTEL while
        # the package was 0.2.x, so bug reports carried the wrong version.
        default_factory=lambda: _package_version()
    )
    # "local" = self-hosted OSS: metrics go to console (not AWS EMF) and no
    # X-Ray propagator is attached. (core/metrics.py already branches on "local".)
    environment: Literal["development", "staging", "production", "local"] = Field(
        default="development", alias="MLPAL_ENVIRONMENT"
    )
    debug: bool = Field(default=False, alias="MLPAL_DEBUG")
    log_level: str = Field(default="INFO", alias="MLPAL_LOG_LEVEL")

    # API
    api_v1_prefix: str = "/v1"
    allowed_hosts: list[str] = ["*"]
    # /v1/messages is the universal core everywhere (see api/mounting.py). This opt-in mounts the
    # Bedrock-mantle Claude-Code passthrough at the scoped /mantle/v1/messages
    # (prod showed zero mantle traffic — 30d usage_logs, 2026-08). The OSS
    # build doesn't ship the mantle module; keep this False there.
    enable_bedrock_mantle_messages: bool = Field(
        default=False, alias="MLPAL_ENABLE_BEDROCK_MANTLE_MESSAGES"
    )
    # Transitional: keep the historical /v2/{messages,catalog,feedback} paths
    # answering (same handlers, deprecated) while live consumers (yodex) move
    # to /v1. Flip to False once /v2 traffic in usage_logs drains to zero.
    # The OSS build scrubs this to False — /v2 stays reserved there.
    serve_legacy_v2_aliases: bool = Field(
        default=False, alias="MLPAL_SERVE_LEGACY_V2_ALIASES"
    )
    # Free-tier platform fee (managed only; locked pricing): when a user's
    # calendar-month tokens cross the threshold, ONE flat fee (in CU) is
    # charged via a synthetic usage row (services/platform_fee.py). 0 disables.
    platform_fee_threshold_tokens: int = Field(
        default=300_000_000, alias="MLPAL_PLATFORM_FEE_THRESHOLD_TOKENS"
    )
    platform_fee_cu: Decimal = Field(
        default=Decimal("5"), alias="MLPAL_PLATFORM_FEE_CU"
    )
    backend_base_url: str = Field(
        default="",
        alias="MLPAL_BACKEND_URL",
    )
    payments_base_url: str = Field(
        default="",
        alias="MLPAL_PAYMENTS_URL",
    )
    internal_service_api_key: str | None = Field(
        default=None,
        alias="INTERNAL_SERVICE_API_KEY",
    )
    # MSI service identity token for billing/wallet calls into backend.
    # Preferred over legacy internal_service_api_key when set. During
    # parallel-acceptance window, we send BOTH headers so calls work
    # whether or not backend's receiver has been migrated to MSI.
    # Inbound service identities (mlpal_svc_*, minted by mlpal-auth-service)
    # are validated against the auth service. Today the only inbound service
    # caller is the auth service itself, managing HOP-keyring keys.
    # Unset (OSS default) = service identities are not accepted anywhere.
    auth_service_url: str | None = Field(default=None, alias="MLPAL_AUTH_SERVICE_URL")
    service_identity_token: str | None = Field(
        default=None,
        alias="MLPAL_SERVICE_IDENTITY_TOKEN",
    )
    wallet_timeout_seconds: float = Field(
        default=2.0,
        alias="MLPAL_WALLET_TIMEOUT_SECONDS",
    )
    wallet_cache_ttl_seconds: int = Field(
        default=60,
        alias="MLPAL_WALLET_CACHE_TTL_SECONDS",
    )
    # Near-empty wallets cache for less time: the stale-allow window shrinks
    # exactly when the gate is about to close (gating design 2026-08-12).
    wallet_low_balance_cache_ttl_seconds: int = Field(
        default=10,
        alias="MLPAL_WALLET_LOW_BALANCE_CACHE_TTL_SECONDS",
    )
    wallet_retry_interval_seconds: int = Field(
        default=30,
        alias="MLPAL_WALLET_RETRY_INTERVAL_SECONDS",
    )
    # TEMPORARY kill switch for wallet debits, independent of the backend
    # walletGatingEnabled rollout flag. While the wallet billing system is being
    # reworked, set this false to skip all debits: serving paths record usage as
    # 'not_applicable' (usage is still logged for later reconciliation) and the
    # background retry worker retires any in-flight intents instead of charging.
    # Flip back to true (or drop the env var) once the rework lands.
    wallet_debit_enabled: bool = Field(
        default=True,
        alias="MLPAL_WALLET_DEBIT_ENABLED",
    )

    # Prompt-cache CU multipliers (relative to the base input-token rate).
    # Match Anthropic's published Claude-on-Bedrock cache pricing; env-tunable
    # so pricing changes don't need a code deploy. Move into ModelPricing when
    # that schema gains explicit cache columns.
    cache_5m_write_multiplier: Decimal = Field(
        default=Decimal("1.25"), alias="MLPAL_CACHE_5M_WRITE_MULTIPLIER"
    )
    cache_1h_write_multiplier: Decimal = Field(
        default=Decimal("2.00"), alias="MLPAL_CACHE_1H_WRITE_MULTIPLIER"
    )
    cache_read_multiplier: Decimal = Field(
        default=Decimal("0.10"), alias="MLPAL_CACHE_READ_MULTIPLIER"
    )

    # /v2/messages (universal Anthropic-Messages endpoint). All env-tunable.
    # Allowlist of model tags v2 will serve (config, not a code path). The
    # wildcard "*" (GA default) admits any served chat model — a chat-operation
    # model on a provider with an edge (anthropic/openai/google); non-chat
    # (image/embedding/tts) and edgeless providers (bedrock) are still rejected.
    # Replace with explicit tags to pin a narrower set.
    messages_v2_allowlist: list[str] = Field(
        default=["*"],
        alias="MLPAL_MESSAGES_V2_ALLOWLIST",
    )
    messages_v2_heartbeat_interval: float = Field(
        default=15.0, alias="MLPAL_MESSAGES_V2_HEARTBEAT_INTERVAL"
    )
    # Reserve at least this many output tokens on Gemini thinking models by
    # capping thinking_budget to (max_tokens - reserve), so a tight max_tokens
    # can't be fully consumed by reasoning and yield an empty answer. Applied
    # only when no tools are set (the tool path owns thinking config for
    # thought-signatures). 0 disables. Pro models keep their 128-token minimum.
    google_thinking_output_reserve: int = Field(
        default=256, alias="MLPAL_GOOGLE_THINKING_OUTPUT_RESERVE"
    )
    # --- Prompt caching on /v2/messages (cache_control is Anthropic-wire, so
    # this is a v2-only concern; v1 never triggers it). Caching differs by
    # provider, so only the one that needs new, billable machinery gets a flag:
    #   - Anthropic: client cache_control is passed through unchanged (always on;
    #     a flag would just mean discarding client intent — never wanted).
    #   - OpenAI: automatic prefix caching, provider-side (nothing to toggle).
    #   - Google: NO effective automatic caching for our shape, so we create
    #     explicit cachedContents. Billable + stateful -> flag, default OFF,
    #     canary per tenant. Rollback = flip the flag.
    messages_v2_cache_google: bool = Field(
        default=False, alias="MLPAL_MESSAGES_V2_CACHE_GOOGLE"
    )
    # Don't create a cachedContent for a prefix smaller than this (Gemini's own
    # minimum is ~2048; below it, create fails and there's no reuse win anyway).
    messages_v2_cache_min_tokens: int = Field(
        default=2048, alias="MLPAL_MESSAGES_V2_CACHE_MIN_TOKENS"
    )
    # Default cachedContent TTL when the client's cache_control gives none.
    messages_v2_cache_ttl_seconds: int = Field(
        default=3600, alias="MLPAL_MESSAGES_V2_CACHE_TTL_SECONDS"
    )
    # Stream reasoning/thinking text as Anthropic `thinking` content-block deltas
    # on the translating edges (OpenAI/Google), so clients render live activity
    # during the model's think phase. Env kill-switch for quick rollback.
    messages_v2_stream_thinking: bool = Field(
        default=True, alias="MLPAL_MESSAGES_V2_STREAM_THINKING"
    )
    # --- Multi-cloud serving backends ------------------------------------
    # Priority-ordered CSV per model FAMILY. The first backend that is
    # configured and serves the requested model wins. Valid names:
    #   openai:    first_party, azure
    #   google:    first_party, vertex
    #   anthropic: first_party, bedrock, vertex
    # (The `bedrock` catalog family — open-weight models — is unchanged.)
    openai_backends: str = Field(default="first_party", alias="MLPAL_OPENAI_BACKENDS")
    google_backends: str = Field(default="first_party", alias="MLPAL_GOOGLE_BACKENDS")
    anthropic_backends: str = Field(
        default="first_party", alias="MLPAL_ANTHROPIC_BACKENDS"
    )
    # Backend failover: when a request fails on its serving backend with a
    # serving fault (5xx / timeout / connection error / provider 429 / open
    # breaker), retry the SAME model once on the next backend in the family's
    # priority list. Never on 4xx, never after a stream has emitted. Circuit
    # breakers are per backend, so a dead backend is skipped without paying
    # its timeout once its breaker opens. Off = one backend per model, as before.
    backend_failover_enabled: bool = Field(default=True, alias="MLPAL_BACKEND_FAILOVER")
    # Azure OpenAI / AI Foundry, v1 surface (<endpoint>/openai/v1/). Azure
    # addresses models by DEPLOYMENT name; name deployments after the model
    # IDs (e.g. deployment "gpt-5.2" for gpt-5.2) and no mapping is needed.
    # For accurate serves()/console display, or non-identity names, provide
    # MLPAL_AZURE_DEPLOYMENTS as JSON {model_id: deployment_name}.
    azure_openai_endpoint: str | None = Field(
        default=None, alias="MLPAL_AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_api_key: str | None = Field(
        default=None, alias="MLPAL_AZURE_OPENAI_API_KEY"
    )
    azure_openai_deployments: str | None = Field(
        default=None, alias="MLPAL_AZURE_DEPLOYMENTS"
    )
    # Claude on Microsoft Foundry (same AIServices resource, native Anthropic
    # wire at <endpoint>/anthropic). Deployments are addressed by name — name
    # them after the Anthropic model IDs and no map is needed; this JSON map
    # covers non-identity names and makes serves()/console display exact.
    azure_anthropic_deployments: str | None = Field(
        default=None, alias="MLPAL_AZURE_ANTHROPIC_DEPLOYMENTS"
    )
    # Vertex AI. One service account (GOOGLE_APPLICATION_CREDENTIALS, ADC)
    # serves both Gemini (google-genai vertexai=True) and Claude
    # (AnthropicVertex). "global" location preferred: higher availability,
    # and Claude at first-party price (regional endpoints carry +10%).
    vertex_project: str | None = Field(default=None, alias="MLPAL_VERTEX_PROJECT")
    vertex_location: str = Field(default="global", alias="MLPAL_VERTEX_LOCATION")
    # Claude-on-{Bedrock,Vertex} model maps: JSON {anthropic_model_id:
    # backend_model_id}. EXPLICIT and empty by default — both clouds require
    # per-model enablement (Bedrock model access / Vertex Model Garden), so
    # we never guess what an account can serve. `scripts/probe_backends.py`
    # discovers and verifies the map for your credentials.
    bedrock_anthropic_models: str | None = Field(
        default=None, alias="MLPAL_BEDROCK_ANTHROPIC_MODELS"
    )
    # OpenAI proprietary models on Bedrock (Responses wire at
    # bedrock-runtime/openai/v1, SigV4): {provider_model_id: inference-profile
    # id}, e.g. {"gpt-6-astra": "global.openai.gpt-6-astra"}. Explicit for the
    # same reason as the Claude map; `scripts/probe_backends.py openai`
    # generates it. `global.` profiles are priced at OpenAI list.
    bedrock_openai_models: str | None = Field(
        default=None, alias="MLPAL_BEDROCK_OPENAI_MODELS"
    )
    # Models the bedrock-mantle NATIVE endpoint serves (JSON list of Anthropic
    # model IDs). Mantle's population is a subset of bedrock-runtime's (newest
    # dateless-ID generation only — live-verified 2026-08-14: opus-4-7/4-8,
    # opus-5, sonnet-5). Models on this list get the byte-faithful native
    # /v1/messages path; other Claude models fall back to the adapter path.
    bedrock_mantle_models: str | None = Field(
        default=None, alias="MLPAL_BEDROCK_MANTLE_MODELS"
    )
    vertex_anthropic_models: str | None = Field(
        default=None, alias="MLPAL_VERTEX_ANTHROPIC_MODELS"
    )
    # --- Connections (BYOK provider keys + BYOM custom endpoints) ----------
    # Feature gate: off by default everywhere until the rollout decision.
    connections_enabled: bool = Field(default=False, alias="MLPAL_CONNECTIONS_ENABLED")
    # Custody driver for tenant secrets: 'secrets_service' (managed; KMS
    # envelope + audit via mlpal-secrets-service) or 'local' (dev-grade
    # AES-GCM in the gateway DB).
    custody_driver: str = Field(default="local", alias="MLPAL_CUSTODY_DRIVER")
    custody_secrets_service_url: str | None = Field(
        default=None, alias="MLPAL_SECRETS_SERVICE_URL"
    )
    custody_secrets_service_token: str | None = Field(
        default=None, alias="MLPAL_SECRETS_SERVICE_TOKEN"
    )
    custody_local_key: str | None = Field(default=None, alias="MLPAL_CUSTODY_LOCAL_KEY")
    # Bounded pool of tenant-scoped adapter clients (LRU).
    connections_pool_size: int = Field(default=256, alias="MLPAL_CONNECTIONS_POOL_SIZE")
    # Hosted catalog feed: 'bundled' ships frozen; 'hosted' subscribes to
    # catalog_feed_url and keeps the catalog current (runtime-toggleable).
    catalog_feed_mode: str = Field(default="bundled", alias="MLPAL_CATALOG_FEED")
    catalog_feed_url: str = Field(
        default="https://models.mlpal.ai/v1/catalog/feed", alias="MLPAL_CATALOG_FEED_URL"
    )
    catalog_feed_interval_hours: float = Field(
        default=24.0, alias="MLPAL_CATALOG_FEED_INTERVAL_HOURS"
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com", alias="MLPAL_ANTHROPIC_BASE_URL"
    )
    anthropic_api_version: str = Field(
        default="2023-06-01", alias="MLPAL_ANTHROPIC_API_VERSION"
    )

    # Database. Defaults are the standard local-dev triple so a fresh checkout
    # can run tests / boot against a local postgres with zero env; every real
    # deployment (compose, k8s) sets these explicitly.
    db_host: str = Field(default="localhost", alias="MLPAL_DB_HOST")
    db_port: int = Field(default=5432, alias="MLPAL_DB_PORT")
    db_name: str = Field(default="postgres", alias="MLPAL_DB_NAME")
    db_user: str = Field(default="postgres", alias="MLPAL_DB_USER")
    db_password: str = Field(default="postgres", alias="MLPAL_DB_PASSWORD")
    db_schema: str = Field(default="assistants", alias="MLPAL_DB_SCHEMA")
    user_schema: str = Field(default="mlpal_test", alias="MLPAL_USER_SCHEMA")
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Construct async database URL."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                path=self.db_name,
            )
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_sync(self) -> str:
        """Construct sync database URL for alembic."""
        return str(
            PostgresDsn.build(
                scheme="postgresql",
                username=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                path=self.db_name,
            )
        )

    # Redis
    redis_host: str = Field(default="localhost", alias="MLPAL_REDIS_HOST")
    redis_port: int = Field(default=6379, alias="MLPAL_REDIS_PORT")
    redis_password: str | None = Field(default=None, alias="MLPAL_REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="MLPAL_REDIS_DB")
    redis_ssl: bool = Field(default=False, alias="MLPAL_REDIS_SSL")
    # Bounded command-connection pool. The default redis-py pool cap (100) is
    # implicit and unmonitored; we make it explicit so it can be tuned and
    # alerted on. The pub/sub listener uses a separate, tiny pool (below) so it
    # can never exhaust this one — see core/cache.py and ERRORS.md.
    redis_max_connections: int = Field(default=50, alias="MLPAL_REDIS_MAX_CONNECTIONS")
    # Idle-connection PINGs detect half-open sockets (e.g. after an ElastiCache
    # failover) before they surface as request errors.
    redis_health_check_interval: int = Field(
        default=30, alias="MLPAL_REDIS_HEALTH_CHECK_INTERVAL"
    )
    # Dedicated pool for the cache-invalidation pub/sub subscriber. Kept small
    # and isolated so a stuck/reconnecting subscriber cannot starve commands.
    redis_pubsub_max_connections: int = Field(
        default=4, alias="MLPAL_REDIS_PUBSUB_MAX_CONNECTIONS"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # AWS
    aws_region: str = Field(default="us-east-2", alias="MLPAL_AWS_REGION")
    aws_endpoint_url: str | None = Field(default=None, alias="AWS_ENDPOINT_URL")
    # Region for the Anthropic-format Bedrock endpoint (`bedrock-mantle`).
    # Distinct from `aws_region` because as of 2026-05 the mantle endpoint
    # is only model-populated in us-east-1, while general AWS resources
    # may live in us-east-2.
    bedrock_mantle_region: str = Field(
        default="us-east-1", alias="MLPAL_BEDROCK_MANTLE_REGION",
    )
    # Which Bedrock host serves the NATIVE Anthropic wire: "runtime"
    # (bedrock-runtime.<region>.amazonaws.com/anthropic/v1/messages — full
    # Claude population, addressed by inference-profile id from
    # MLPAL_BEDROCK_ANTHROPIC_MODELS; verified 2026-09-17: cache 5m/1h,
    # effort, adaptive thinking, tools, streaming, betas) or the legacy
    # "mantle" compatibility host (Haiku only in us-east-2 as of 09-2026).
    bedrock_native_endpoint: str = Field(
        default="runtime", alias="MLPAL_BEDROCK_NATIVE_ENDPOINT",
    )
    sqs_usage_queue_url: str | None = Field(default=None, alias="MLPAL_SQS_USAGE_QUEUE_URL")
    sqs_usage_batch_size: int = Field(
        default=10,  # 10 messages per poll (SQS max)
        alias="MLPAL_SQS_USAGE_BATCH_SIZE",
        description="Max messages to receive per SQS poll (1-10)",
    )
    sqs_long_poll_wait: int = Field(
        default=20,  # 20 seconds (SQS max, most cost-effective)
        alias="MLPAL_SQS_LONG_POLL_WAIT",
        description="Long poll wait time in seconds (1-20, 20 recommended)",
    )

    # Asset Storage (S3)
    s3_assets_bucket: str = Field(
        default="mlpal-assistants-assets",
        alias="MLPAL_S3_ASSETS_BUCKET",
    )
    s3_assets_region: str = Field(
        default="us-east-2",
        alias="MLPAL_S3_ASSETS_REGION",
    )
    asset_url_expiration: int = Field(
        default=3600,  # 1 hour, same as OpenAI
        alias="MLPAL_ASSET_URL_EXPIRATION",
    )

    # Cognito (for JWT auth)
    cognito_region: str = Field(default="us-east-2", alias="COGNITO_REGION")
    cognito_user_pool_id: str = Field(default="", alias="COGNITO_USER_POOL_ID")
    cognito_client_id: str = Field(default="", alias="COGNITO_CLIENT_ID")

    # Provider API Keys. Optional so a self-hosted box can run with only the
    # providers it has keys for — the AdapterFactory enables a provider only when
    # its key is present, and the served catalog filters to enabled providers.
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    # Rate Limiting
    rate_limit_enabled: bool = True
    default_rate_limit_rpm: int = 60  # requests per minute
    default_rate_limit_tpm: int = 100_000  # tokens per minute

    # API Key Settings
    api_key_prefix: str = "mlpal_sk_"
    api_key_bytes: int = 32  # 256 bits of randomness
    api_key_cache_ttl: int = Field(
        default=1800,  # 30 minutes — keys rarely change; admin invalidation on revoke/update
        alias="MLPAL_API_KEY_CACHE_TTL",
        description="TTL in seconds for API key validation cache in Redis",
    )
    api_key_last_used_flush_interval: int = Field(
        default=60,  # 1 minute
        alias="MLPAL_API_KEY_LAST_USED_FLUSH_INTERVAL",
        description="Interval in seconds for flushing last_used_at updates to DB",
    )

    # Cache TTL Settings
    model_cache_ttl: int = Field(
        default=3600,  # 1 hour — models change ~monthly; admin invalidation covers updates
        alias="MLPAL_MODEL_CACHE_TTL",
        description="TTL in seconds for model registry entries in Redis",
    )
    local_cache_ttl: int = Field(
        default=300,  # 5 minutes — prevents stale in-memory reads between instances
        alias="MLPAL_LOCAL_CACHE_TTL",
        description="TTL in seconds for in-memory TTLCache entries (models, pricing, routing)",
    )
    local_cache_maxsize: int = Field(
        default=10000,  # Max entries per cache to prevent OOM
        alias="MLPAL_LOCAL_CACHE_MAXSIZE",
        description="Maximum number of entries in each in-memory TTLCache",
    )
    routing_refresh_interval: int = Field(
        default=120,  # 2 minutes
        alias="MLPAL_ROUTING_REFRESH_INTERVAL",
        description="Interval in seconds for background meta-model routing table refresh",
    )


    # Billing seam (open-core). "managed" = MLPal payments/wallet (default,
    # preserves current prod behavior). "local" = self-hosted OSS: allow-all
    # gate with NO outbound calls to the MLPal backend/payments services — per-key
    # spend budgets (services/policy.py) remain the local enforcement primitive.
    billing_backend: str = Field(
        default="local",
        alias="MLPAL_BILLING_BACKEND",
        description="'managed' (MLPal payments/wallet) or 'local' (OSS, no callout).",
    )
    # Bedrock uses AWS IAM/creds, not an API key. Default True preserves prod
    # (where AWS is configured); a self-hosted box with no AWS sets this False so
    # Bedrock models aren't listed/routed (they'd fail at call otherwise).
    enable_bedrock: bool = Field(
        default=False,
        alias="MLPAL_ENABLE_BEDROCK",
        description="Enable the Bedrock provider (requires AWS creds). Set False on OSS boxes without AWS.",
    )
    # Auth seam for MANAGEMENT endpoints (create/manage keys, policies, budgets).
    # "managed" (default) = Cognito JWT + user-schema lookup (unchanged prod).
    # "local" (OSS) = an API key carrying the 'admin' permission manages the box;
    # no Cognito, no users table. Inference endpoints are API-key auth in both.
    auth_backend: str = Field(
        default="local",
        alias="MLPAL_AUTH_BACKEND",
        description="'managed' (Cognito JWT) or 'local' (admin API key) for mgmt endpoints.",
    )

    # Per-key policy engine (model access + spend budgets).
    # USD<->CU conversion for USD-denominated budgets. Compute units are the unit
    # we actually record; a USD budget is normalized to CU as usd / cu_to_usd.
    # INVARIANT: must match the `cu_to_dollar` used in the model_pricing ledger
    # (currently 10.0 for every row) or a $-budget will not equal its true $ spend.
    cu_to_usd: float = Field(
        default=10.0,
        alias="MLPAL_CU_TO_USD",
        description="Dollars per compute unit; must equal model_pricing.cu_to_dollar.",
    )
    # Timezone for calendar-aligned budget windows (daily midnight / weekly Monday
    # / monthly 1st). IANA zone; default UTC.
    budget_timezone: str = Field(
        default="UTC",
        alias="MLPAL_BUDGET_TIMEZONE",
        description="IANA timezone for calendar budget window boundaries.",
    )

    # Observability - Tracing
    tracing_enabled: bool = Field(default=True, alias="TRACING_ENABLED")
    otel_service_name: str = Field(
        default="mlpal-assistants-service",
        alias="OTEL_SERVICE_NAME",
    )
    otel_exporter_endpoint: str = Field(
        default="http://localhost:4317",
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_debug: bool = Field(default=False, alias="OTEL_DEBUG")

    # Observability - Metrics
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")
    metrics_namespace: str = Field(default="MLPal/Assistants", alias="AWS_EMF_NAMESPACE")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()  # type: ignore[call-arg]
