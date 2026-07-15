"""Audit log writer. Emits one JSONL event per pipeline step.

Every event is schema-validated against schemas/audit_log.schema.json before write.
The audit log must be sufficient to reconstruct every major curation, model,
simulation, and interpretation decision (Build Spec §18).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "audit_log.schema.json"
DEFAULT_LOG_PATH = REPO_ROOT / "build_artifacts" / "audit_log.jsonl"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "no-git-repository"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    with path.open("rb") as fh:
        return sha256_bytes(fh.read())


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


class AuditWriter:
    """Append-only JSONL writer with schema validation on every event."""

    def __init__(self, log_path: Path | None = None, schema_path: Path | None = None):
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self.schema_path = Path(schema_path) if schema_path else SCHEMA_PATH
        with self.schema_path.open("r") as fh:
            self._schema = json.load(fh)
        self._validator = Draft202012Validator(self._schema)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "timestamp": _now_iso(),
            "event_type": event_type,
            "git_commit": _git_commit(),
        }
        event.update(fields)
        errors = sorted(self._validator.iter_errors(event), key=lambda e: e.path)
        if errors:
            raise ValueError(
                "Audit event failed schema validation: "
                + "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
            )
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def read_events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        with self.log_path.open("r") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def event_types_present(self) -> set[str]:
        return {event["event_type"] for event in self.read_events()}

    def reset(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()


def hash_audit_log(log_path: Path | None = None) -> str:
    path = Path(log_path) if log_path else DEFAULT_LOG_PATH
    if not path.exists():
        return sha256_bytes(b"")
    return sha256_path(path)
