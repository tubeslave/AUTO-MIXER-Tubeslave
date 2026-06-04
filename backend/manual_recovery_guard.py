"""Guard rails for blocked manual recovery scripts.

These scripts can rewrite broad WING channel state. They are intentionally not
part of Automixer automation, Product Layer apply, or no-WING release checks.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping


ACK_ENV_VAR = "AUTOMIXER_MANUAL_RECOVERY_ACK"
ACK_VALUE = "manual-recovery-writes-to-wing"


def has_manual_recovery_ack(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get(ACK_ENV_VAR) == ACK_VALUE


def require_manual_recovery_ack(script_name: str, env: Mapping[str, str] | None = None) -> None:
    if has_manual_recovery_ack(env):
        return

    print(
        "\n".join(
            (
                f"{script_name} is blocked from ordinary CLI/autoloop use.",
                "It can rewrite broad WING channel state and is not an Automixer live executor.",
                f"Set {ACK_ENV_VAR}={ACK_VALUE} only during a human-approved manual recovery procedure.",
                "The script will still require its interactive confirmation after this acknowledgement.",
            )
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)
