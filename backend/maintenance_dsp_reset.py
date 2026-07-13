"""Fixed M110 maintenance-reset contract; execution is implemented in CP03."""

from __future__ import annotations


M110_RESET_CHANNELS = tuple(range(1, 17))
M110_RESET_SUFFIX_TARGETS = (
    ("eq/on", 0),
    ("flt/lc", 0),
    ("gate/on", 0),
    ("dyn/on", 0),
)
M110_RESET_ADDRESSES = tuple(
    f"/ch/{channel}/{suffix}"
    for channel in M110_RESET_CHANNELS
    for suffix, _target in M110_RESET_SUFFIX_TARGETS
)
M110_RESET_TARGETS = {
    f"/ch/{channel}/{suffix}": target
    for channel in M110_RESET_CHANNELS
    for suffix, target in M110_RESET_SUFFIX_TARGETS
}


def fixed_m110_reset_targets() -> dict[str, int]:
    return dict(M110_RESET_TARGETS)
