"""Service modules log via logging.getLogger(__name__); those records must
reach stdout (2026-10-01: they were silently dropped in prod)."""

import logging
import sys

from mlpal_assistants_service.core.logging import configure_stdlib_logging


def test_stdlib_records_reach_stdout_once(capsys):
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    try:
        configure_stdlib_logging("INFO")
        configure_stdlib_logging("INFO")  # idempotent: one stdout handler
        stdout_handlers = [h for h in root.handlers if getattr(h, "stream", None) is sys.stdout]
        assert len(stdout_handlers) == 1
        logging.getLogger("mlpal_assistants_service.services.chat").error("background task failed: x")
        logging.getLogger("httpx").info("chatty")  # quieted to WARNING
        out = capsys.readouterr().out
        assert "background task failed: x" in out and "chatty" not in out
    finally:
        root.handlers[:] = before
        root.setLevel(level)
