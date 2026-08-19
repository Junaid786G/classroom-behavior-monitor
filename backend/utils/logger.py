from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        obj: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            obj["stack"] = self.formatStack(record.stack_info)
        # Attach any extra keys passed via logger.info("x", extra={...})
        _skip = logging.LogRecord.__dict__.keys() | {"message", "asctime"}
        for key, val in record.__dict__.items():
            if key not in _skip and not key.startswith("_"):
                obj[key] = val
        return json.dumps(obj, default=str)


def setup_logging(level: str = "INFO", log_dir: Optional[str] = None) -> None:
    """
    Configure root logger with JSON console + rotating file handlers.
    Call once from main.py lifespan before the app starts serving.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on hot-reload
    if root.handlers:
        root.handlers.clear()

    formatter = _JSONFormatter()

    # ── Console ───────────────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── Rotating file ─────────────────────────────────────────────────────────
    if log_dir:
        from pathlib import Path
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            Path(log_dir) / "app.log",
            maxBytes=20 * 1024 * 1024,   # 20 MB per file
            backupCount=7,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # ── Quieten noisy third-party loggers ─────────────────────────────────────
    for noisy in (
        "uvicorn.access",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "httpx",
        "httpcore",
        "multipart",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
