"""Small, shared helpers for atomically writing runtime artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON and replace the destination only after serialization succeeds."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write text and replace the destination only after the write succeeds."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
