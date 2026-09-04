"""Async image jobs: submit idempotency, runner lifecycle, worker_lost.

Real-SQL tests on sqlite (JSONB compiled to JSON, same shim as
test_platform_fee); the runner and heartbeat are exercised with a faked
ImageService and a monkeypatched session factory.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.dialects.postgresql import BYTEA, JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles

import mlpal_assistants_service.services.image_jobs as jobs_mod
from mlpal_assistants_service.core.exceptions import (
    ModelNotFoundError,
    ProviderError,
    WalletEmptyError,
)
from mlpal_assistants_service.db.models import Base
from mlpal_assistants_service.db.models.image_job import ImageJob
from mlpal_assistants_service.schemas.images import ImageGenerationRequest
from mlpal_assistants_service.services.image_jobs import (
    IdempotencyConflictError,
    get_job,
    submit_job,
)


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003
    return "JSON"


@compiles(BYTEA, "sqlite")
def _bytea_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003
    return "BLOB"


@pytest_asyncio.fixture
async def session_factory(tmp_path):
    # File-backed sqlite, NOT :memory: — the runner and heartbeat open their
    # own sessions, and every new aiosqlite :memory: connection would get its
    # own empty db (and a shared StaticPool connection trips MissingGreenlet).
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/image_jobs.db",
        execution_options={"schema_translate_map": {"assistants": None}},
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda c: Base.metadata.create_all(c, tables=[ImageJob.__table__])
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s


def _api_key(user_id=1, key_id=2):
    return SimpleNamespace(
        user_id=user_id, id=key_id, model_policy=None, budgets=None
    )


def _image_service(allowed=True, reason=None):
    svc = SimpleNamespace()
    svc._billing = SimpleNamespace(
        can_make_request_cached=AsyncMock(return_value=(allowed, reason, True))
    )
    svc.redis = None
    svc._asset_storage = None
    svc._sqs_client = None
    return svc


def _request(**overrides):
    body = {"prompt": "a lighthouse at dusk", "model": "gpt-image-2", "wait": False}
    body.update(overrides)
    return ImageGenerationRequest(**body)


# --- submit ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_submit_creates_queued_job_and_spawns_runner(session, monkeypatch):
    spawned = {}
    monkeypatch.setattr(jobs_mod, "_spawn_runner", lambda **kw: spawned.update(kw))

    job, created = await submit_job(
        session, api_key=_api_key(), request=_request(), image_service=_image_service()
    )

    assert created is True
    assert job.id.startswith("imgjob_")
    assert job.status == "queued"
    # job-control fields never reach the stored render request
    assert "wait" not in job.request and "idempotency_key" not in job.request
    assert spawned["job_id"] == job.id and spawned["user_id"] == 1


@pytest.mark.asyncio
async def test_submit_idempotent_replay_returns_existing(session, monkeypatch):
    monkeypatch.setattr(jobs_mod, "_spawn_runner", lambda **kw: None)
    key = _api_key()
    req = _request(idempotency_key="idem-1")

    first, created1 = await submit_job(
        session, api_key=key, request=req, image_service=_image_service()
    )
    second, created2 = await submit_job(
        session, api_key=key, request=req, image_service=_image_service()
    )

    assert created1 is True and created2 is False
    assert second.id == first.id


@pytest.mark.asyncio
async def test_submit_idempotency_conflict_on_different_body(session, monkeypatch):
    monkeypatch.setattr(jobs_mod, "_spawn_runner", lambda **kw: None)
    key = _api_key()

    await submit_job(
        session,
        api_key=key,
        request=_request(idempotency_key="idem-2"),
        image_service=_image_service(),
    )
    with pytest.raises(IdempotencyConflictError):
        await submit_job(
            session,
            api_key=key,
            request=_request(idempotency_key="idem-2", prompt="a DIFFERENT prompt"),
            image_service=_image_service(),
        )


@pytest.mark.asyncio
async def test_submit_blocked_wallet_never_queues(session, monkeypatch):
    monkeypatch.setattr(jobs_mod, "_spawn_runner", lambda **kw: None)
    from mlpal_assistants_service.repositories.billing_repository import (
        WALLET_EMPTY_MESSAGE,
    )

    with pytest.raises(WalletEmptyError):
        await submit_job(
            session,
            api_key=_api_key(),
            request=_request(),
            image_service=_image_service(allowed=False, reason=WALLET_EMPTY_MESSAGE),
        )
    count = len((await session.execute(ImageJob.__table__.select())).all())
    assert count == 0


# --- runner ------------------------------------------------------------------
async def _submitted_job(session, monkeypatch) -> ImageJob:
    monkeypatch.setattr(jobs_mod, "_spawn_runner", lambda **kw: None)
    job, _ = await submit_job(
        session, api_key=_api_key(), request=_request(), image_service=_image_service()
    )
    return job


def _patch_runner_env(monkeypatch, session_factory, generate):
    import mlpal_assistants_service.db.session as db_session
    import mlpal_assistants_service.services.image as image_mod

    monkeypatch.setattr(db_session, "async_session_factory", session_factory)
    _gen = generate  # class bodies can't resolve a name they also assign

    class _FakeImageService:
        def __init__(self, *a, **kw):
            pass

        generate = staticmethod(_gen)

    monkeypatch.setattr(image_mod, "ImageService", _FakeImageService)


@pytest.mark.asyncio
async def test_runner_success_stores_sync_payload(session, session_factory, monkeypatch):
    job = await _submitted_job(session, monkeypatch)
    fake_response = SimpleNamespace(
        model_dump=lambda mode: {"data": [{"url": "https://s3/img.png"}], "model": "gpt-image-2"}
    )

    async def generate(**kw):
        return fake_response

    _patch_runner_env(monkeypatch, session_factory, generate)
    await jobs_mod._run_job(
        job_id=job.id, user_id=1, api_key_id=2, model_policy=None, budgets=None,
        redis_client=None, asset_storage=None, sqs_client=None,
    )

    async with session_factory() as fresh:
        got = await get_job(fresh, user_id=1, job_id=job.id)
    assert got.status == "succeeded"
    assert got.result["data"][0]["url"] == "https://s3/img.png"
    assert got.finished_at is not None and got.error is None


@pytest.mark.asyncio
async def test_runner_failure_classification(session, session_factory, monkeypatch):
    job = await _submitted_job(session, monkeypatch)

    async def generate(**kw):
        raise ModelNotFoundError("nope-model")

    _patch_runner_env(monkeypatch, session_factory, generate)
    await jobs_mod._run_job(
        job_id=job.id, user_id=1, api_key_id=2, model_policy=None, budgets=None,
        redis_client=None, asset_storage=None, sqs_client=None,
    )

    async with session_factory() as fresh:
        got = await get_job(fresh, user_id=1, job_id=job.id)
    assert got.status == "failed"
    assert got.error["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_runner_provider_error(session, session_factory, monkeypatch):
    job = await _submitted_job(session, monkeypatch)

    async def generate(**kw):
        raise ProviderError(message="upstream 500", provider="openai")

    _patch_runner_env(monkeypatch, session_factory, generate)
    await jobs_mod._run_job(
        job_id=job.id, user_id=1, api_key_id=2, model_policy=None, budgets=None,
        redis_client=None, asset_storage=None, sqs_client=None,
    )

    async with session_factory() as fresh:
        got = await get_job(fresh, user_id=1, job_id=job.id)
    assert got.status == "failed"
    assert got.error["code"] == "provider_error"


# --- poll / worker_lost ------------------------------------------------------
@pytest.mark.asyncio
async def test_get_job_scoped_to_owner(session, monkeypatch):
    job = await _submitted_job(session, monkeypatch)
    assert await get_job(session, user_id=999, job_id=job.id) is None
    assert (await get_job(session, user_id=1, job_id=job.id)).id == job.id


@pytest.mark.asyncio
async def test_stale_running_job_becomes_worker_lost(session, monkeypatch):
    job = await _submitted_job(session, monkeypatch)
    job.status = "running"
    job.started_at = datetime.now(UTC) - timedelta(minutes=10)
    job.heartbeat_at = datetime.now(UTC) - timedelta(minutes=9)
    await session.commit()

    got = await get_job(session, user_id=1, job_id=job.id)
    assert got.status == "failed"
    assert got.error["code"] == "worker_lost"


@pytest.mark.asyncio
async def test_fresh_running_job_stays_running(session, monkeypatch):
    job = await _submitted_job(session, monkeypatch)
    job.status = "running"
    job.started_at = datetime.now(UTC)
    job.heartbeat_at = datetime.now(UTC)
    await session.commit()

    got = await get_job(session, user_id=1, job_id=job.id)
    assert got.status == "running"


@pytest.mark.asyncio
async def test_heartbeat_loop_touches_row(session, session_factory, monkeypatch):
    job = await _submitted_job(session, monkeypatch)
    monkeypatch.setattr(jobs_mod, "HEARTBEAT_INTERVAL_S", 0.02)
    import mlpal_assistants_service.db.session as db_session

    monkeypatch.setattr(db_session, "async_session_factory", session_factory)

    task = asyncio.create_task(jobs_mod._heartbeat_loop(job.id))
    await asyncio.sleep(0.08)
    task.cancel()

    async with session_factory() as fresh:
        got = await get_job(fresh, user_id=1, job_id=job.id)
    assert got.heartbeat_at is not None
