"""Runtime progress event dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


ProgressCallback = Callable[[dict[str, Any]], None]
_PROGRESS_CALLBACK: ContextVar[ProgressCallback | None] = ContextVar(
    "sjt_progress_callback",
    default=None,
)


def emit_progress(event: Mapping[str, Any]) -> None:
    """Publish a transient runtime event without storing it in workflow State."""

    callback = _PROGRESS_CALLBACK.get()
    if callback is not None:
        callback(dict(event))


@contextmanager
def progress_callback(
    callback: ProgressCallback | None,
) -> Iterator[None]:
    """Make a progress callback available to nested async tasks."""

    token = _PROGRESS_CALLBACK.set(callback)
    try:
        yield
    finally:
        _PROGRESS_CALLBACK.reset(token)
