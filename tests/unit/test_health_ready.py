"""`/health/ready` when a core dependency is down: a 503 with a JSON body, never a 500.

Found by the first OSS deployment on a fresh box: the not-ready branch rendered
`HealthResponse.model_dump()` (a datetime inside) through `JSONResponse`, which raised and
turned every not-ready answer into a 500, so the compose healthcheck never saw readiness.
"""

import json
from types import SimpleNamespace

import pytest

from mlpal_assistants_service import main


class _Request:
    def __init__(self, state: SimpleNamespace) -> None:
        self.app = SimpleNamespace(state=state)


@pytest.mark.asyncio
async def test_not_ready_is_a_503_with_a_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    async def db_down() -> bool:
        return False

    monkeypatch.setattr(main, "check_database_connection", db_down)
    request = _Request(SimpleNamespace(redis=None, asset_storage=None, adapters={}))

    response = await main.ready(request)  # type: ignore[arg-type]

    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["status"] == "not_ready"
    assert body["checks"] == {"database": False, "redis": False, "asset_storage": False}
    assert isinstance(body["timestamp"], str)


@pytest.mark.asyncio
async def test_ready_when_core_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def db_up() -> bool:
        return True

    class _Redis:
        async def ping(self) -> bool:
            return True

    monkeypatch.setattr(main, "check_database_connection", db_up)
    request = _Request(SimpleNamespace(redis=_Redis(), asset_storage=None, adapters={}))

    body = await main.ready(request)  # type: ignore[arg-type]

    assert body.status == "ready"
    assert body.checks == {"database": True, "redis": True, "asset_storage": False}
