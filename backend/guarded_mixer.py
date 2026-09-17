"""Write permit proxy for guarded operation; ordinary reads remain available."""
from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Callable, Iterator


class GuardedMixerClient:
    """Engine-owned writes require an explicit, thread-local safety permit.

    Unknown callables are denied too. This is a control boundary, not a sandbox
    against hostile Python code accessing private attributes. The underlying
    console transport may still issue read requests and keepalives.
    """
    _READ_PREFIXES = ("get_", "read_", "find_", "is_", "has_")
    _LIFECYCLE = {"connect", "disconnect", "subscribe", "unsubscribe",
                  "register_callback", "unregister_callback", "add_callback", "remove_callback"}

    def __init__(self, base_client: Any, on_blocked: Callable[[str], None] | None = None):
        self._base_client = base_client
        self._on_blocked = on_blocked
        self._local = threading.local()

    @contextmanager
    def approved_action(self) -> Iterator[None]:
        previous = getattr(self._local, "permitted", False)
        self._local.permitted = True
        try:
            yield
        finally:
            self._local.permitted = previous

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._base_client, name)
        if not callable(attr) or name in self._LIFECYCLE or name.startswith(self._READ_PREFIXES):
            return attr

        def gated(*args: Any, **kwargs: Any) -> Any:
            read_request = name == "send" and len(args) == 1 and not kwargs
            if read_request or getattr(self._local, "permitted", False):
                return attr(*args, **kwargs)
            if self._on_blocked is not None:
                self._on_blocked(name)
            return False
        return gated
