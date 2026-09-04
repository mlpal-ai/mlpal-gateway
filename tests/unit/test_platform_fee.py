"""Platform fee: one 5-CU charge per (user, month) on crossing the free tier.

Contract confirmed with the backend session 2026-08-11: synthetic usage row
operation='platform_fee', model_tag='platform-fee-300m', deterministic
trace_id, managed-only. Real-SQL tests (sqlite; the pg partial unique index
is the concurrency backstop on top of the pre-check exercised here).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import BYTEA, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles

from mlpal_assistants_service.core.config import get_settings


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003
    return "JSON"


@compiles(BYTEA, "sqlite")
def _bytea_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003
    return "BLOB"


from mlpal_assistants_service.db.models import APIKey, Base
from mlpal_assistants_service.db.models.usage_log import UsageLog
from mlpal_assistants_service.services.platform_fee import (
    FEE_MODEL_TAG,
    FEE_OPERATION,
    maybe_charge_platform_fee,
    month_progress,
)

UsageLog.__table__.c.id.autoincrement = False
_ids = iter(range(50_000, 60_000))
# The service's fee insert relies on DB identity for `id` (fine on Postgres);
# SQLite with the composite PK needs a Python-side default in tests.
_svc_ids = iter(range(90_000, 99_000))
from sqlalchemy import ColumnDefault  # noqa: E402

UsageLog.__table__.c.id.default = ColumnDefault(lambda: next(_svc_ids))


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        execution_options={"schema_translate_map": {"assistants": None, "users": None, "mlpal_test": None}},
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda c: Base.metadata.create_all(c, tables=[UsageLog.__table__, APIKey.__table__])
        )
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def managed(monkeypatch):
    """Managed billing + tiny threshold so tests cross it cheaply."""
    settings = get_settings()
    monkeypatch.setattr(settings, "billing_backend", "managed")
    monkeypatch.setattr(settings, "platform_fee_threshold_tokens", 1000)
    return settings


def _usage_row(user_id: int, tokens: int) -> UsageLog:
    return UsageLog(
        id=next(_ids), user_id=user_id, api_key_id=1, trace_id=f"t-{next(_ids)}",
        model_tag="claude-haiku-4-5-20251001", provider="anthropic", operation="chat",
        input_tokens=tokens, output_tokens=0, compute_units=Decimal("0.001"),
        status="success", wallet_debit_status="not_applicable",
        created_at=datetime.now(UTC),
    )


async def _fee_rows(session, user_id: int) -> list[UsageLog]:
    return list((await session.execute(
        select(UsageLog).where(
            UsageLog.user_id == user_id, UsageLog.operation == FEE_OPERATION
        )
    )).scalars())


@pytest.mark.asyncio
async def test_below_threshold_no_fee(session, managed):
    charged = await maybe_charge_platform_fee(session, None, 1, 500, api_key_id=1)
    assert charged is False
    assert await _fee_rows(session, 1) == []


@pytest.mark.asyncio
async def test_crossing_charges_exactly_once(session, managed):
    # 900 tokens in the ledger; the 200-token crossing request is persisted
    # first (mirrors record_usage ordering: write, then accrue).
    session.add(_usage_row(1, 900))
    session.add(_usage_row(1, 200))
    await session.commit()

    charged = await maybe_charge_platform_fee(session, None, 1, 200, api_key_id=1)
    assert charged is True
    rows = await _fee_rows(session, 1)
    assert len(rows) == 1
    fee = rows[0]
    assert fee.model_tag == FEE_MODEL_TAG
    assert fee.provider == "mlpal"
    assert fee.compute_units == Decimal("5")
    assert fee.input_tokens == 0 and fee.output_tokens == 0
    assert fee.wallet_debit_status == "not_applicable"
    assert fee.trace_id.startswith("fee_1_")
    assert fee.cc_metadata["threshold_tokens"] == 1000

    # Idempotent: further traffic past the threshold never double-charges.
    charged_again = await maybe_charge_platform_fee(session, None, 1, 5000, api_key_id=1)
    assert charged_again is False
    assert len(await _fee_rows(session, 1)) == 1


@pytest.mark.asyncio
async def test_fee_rows_do_not_count_toward_threshold(session, managed):
    session.add(_usage_row(2, 999))
    session.add(_usage_row(2, 1))
    await session.commit()
    await maybe_charge_platform_fee(session, None, 2, 1, api_key_id=1)
    # fee row exists; its 0 tokens (and its exclusion) must not affect others
    progress = await month_progress(session, 2)
    assert progress["monthly_tokens"] == 1000
    assert progress["platform_fee_charged"] is True
    assert progress["monthly_token_limit"] == 1000


@pytest.mark.asyncio
async def test_local_billing_never_charges(session, managed, monkeypatch):
    monkeypatch.setattr(managed, "billing_backend", "local")
    session.add(_usage_row(3, 10_000))
    await session.commit()
    charged = await maybe_charge_platform_fee(session, None, 3, 10_000, api_key_id=1)
    assert charged is False
    assert await _fee_rows(session, 3) == []


@pytest.mark.asyncio
async def test_zero_threshold_disables(session, managed, monkeypatch):
    monkeypatch.setattr(managed, "platform_fee_threshold_tokens", 0)
    charged = await maybe_charge_platform_fee(session, None, 4, 10_000_000, api_key_id=1)
    assert charged is False


@pytest.mark.asyncio
async def test_users_are_independent(session, managed):
    session.add(_usage_row(5, 2000))
    await session.commit()
    await maybe_charge_platform_fee(session, None, 5, 2000, api_key_id=1)
    assert len(await _fee_rows(session, 5)) == 1
    assert await _fee_rows(session, 6) == []
    progress6 = await month_progress(session, 6)
    assert progress6["platform_fee_charged"] is False


def _cached_usage_row(user_id: int, tokens: int, cache_read: int) -> UsageLog:
    row = _usage_row(user_id, tokens)
    row.cc_metadata = {"cache_read_input_tokens": cache_read}
    return row


@pytest.mark.asyncio
async def test_cache_reads_excluded_from_db_reseed(session, managed):
    """2026-08-29 decision: cache READS don't count toward the free tier.
    1,500 input tokens of which 900 are cache reads = 600 countable — below
    the 1,000 threshold, so no fee."""
    session.add(_cached_usage_row(9, 1500, cache_read=900))
    await session.commit()

    charged = await maybe_charge_platform_fee(session, None, 9, 600, api_key_id=1)
    assert charged is False
    assert await _fee_rows(session, 9) == []


@pytest.mark.asyncio
async def test_cache_writes_still_count(session, managed):
    """Cache-write tokens are part of input and DO count: 1,200 input with
    zero cache reads crosses the 1,000 threshold."""
    session.add(_cached_usage_row(10, 1200, cache_read=0))
    await session.commit()

    charged = await maybe_charge_platform_fee(session, None, 10, 1200, api_key_id=1)
    assert charged is True


@pytest.mark.asyncio
async def test_accrual_path_subtracts_cache_reads(session, managed, monkeypatch):
    """UsageService._accrue_platform_fee passes net tokens (input+output −
    cache_read) into the counter."""
    from mlpal_assistants_service.services.usage import UsageService

    seen = {}

    async def fake_fee(sess, redis, user_id, tokens_delta, api_key_id):
        seen.update(user_id=user_id, tokens=tokens_delta)
        return False

    monkeypatch.setattr(
        "mlpal_assistants_service.services.platform_fee.maybe_charge_platform_fee",
        fake_fee,
    )
    svc = UsageService.__new__(UsageService)
    svc.session = session
    svc.redis = None
    await svc._accrue_platform_fee({
        "status": "success", "operation": "chat", "user_id": 7, "api_key_id": 1,
        "input_tokens": 1000, "output_tokens": 100,
        "cc_metadata": {"cache_read_input_tokens": 800},
    })
    assert seen == {"user_id": 7, "tokens": 300}
