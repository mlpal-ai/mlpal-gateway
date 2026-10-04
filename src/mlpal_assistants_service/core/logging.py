"""Stdlib logging → stdout for the service's own modules.

structlog (`structlog.get_logger()`) prints to stdout on its own; most of the
service logs through `logging.getLogger(__name__)` instead, and until
2026-10-01 those records went nowhere: Alembic's `fileConfig` disabled the
loggers at startup and the root logger's only handler was OpenTelemetry's
(which exports, never prints). Every `logger.error(...)` in the request and
background paths was invisible in `kubectl logs`.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
# Chatty libraries stay at WARNING whatever the service level is; SQL echo is
# governed separately by `settings.debug` (SQLAlchemy's own handler).
_QUIET = ("sqlalchemy.engine", "httpcore", "httpx", "botocore", "aiobotocore", "urllib3", "asyncio")


def configure_stdlib_logging(level: str) -> None:
    """Attach one stdout handler to the root logger (idempotent) and apply
    the service log level. Uvicorn's loggers keep their own handlers."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not any(getattr(h, "stream", None) in (sys.stdout, sys.stderr) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    for name in _QUIET:
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
