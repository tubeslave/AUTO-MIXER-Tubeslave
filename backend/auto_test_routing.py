#!/usr/bin/env python3
"""Generate WING routing-address candidates without touching a console.

Routing mutation requires the canonical supervised transaction path with
operator authority, readback, and rollback.  This legacy helper therefore
remains proposal-only and never opens a transport or sends OSC commands.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROUTING_LIVE_APPLY_BLOCK_REASON = "routing_live_apply_requires_supervised_authority"


def build_routing_test_plan(channel: int = 1, dante_output: int = 1):
    """Return legacy address candidates as inert review data."""

    return [
        {
            "name": "Метод 1: /routing/outputs/ch/{ch}/srcgrp и /routing/outputs/ch/{ch}/outgrp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/srcgrp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 2: /routing/outputs/ch/{ch}/src/grp и /routing/outputs/ch/{ch}/out/grp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/src/grp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/out/grp", dante_output),
            ]
        },
        {
            "name": "Метод 3: /routing/ch/{ch}/outputs/srcgrp и /routing/ch/{ch}/outputs/outgrp",
            "addresses": [
                (f"/routing/ch/{channel}/outputs/srcgrp", "WLIVE PLAY"),
                (f"/routing/ch/{channel}/outputs/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 4: /outputs/ch/{ch}/srcgrp и /outputs/ch/{ch}/outgrp",
            "addresses": [
                (f"/outputs/ch/{channel}/srcgrp", "WLIVE PLAY"),
                (f"/outputs/ch/{channel}/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 5: /ch/{ch}/routing/outputs/srcgrp и /ch/{ch}/routing/outputs/outgrp",
            "addresses": [
                (f"/ch/{channel}/routing/outputs/srcgrp", "WLIVE PLAY"),
                (f"/ch/{channel}/routing/outputs/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 6: Альтернативные названия - sourcegrp и outputgrp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/sourcegrp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/outputgrp", dante_output),
            ]
        },
    ]


def auto_test_routing(client=None, channel: int = 1, dante_output: int = 1):
    """Return a blocked receipt and inert candidates; never use ``client``."""

    del client
    plan = build_routing_test_plan(channel=channel, dante_output=dante_output)
    logger.warning(
        "Live routing test blocked: %s; generated %s review-only candidates",
        ROUTING_LIVE_APPLY_BLOCK_REASON,
        len(plan),
    )
    for index, method in enumerate(plan, 1):
        logger.info("Кандидат %s: %s", index, method["name"])
        for address, value in method["addresses"]:
            logger.info("  proposal_only: %s = %s", address, value)
    return {
        "status": "blocked",
        "reason": ROUTING_LIVE_APPLY_BLOCK_REASON,
        "live_authority_granted": False,
        "live_write": False,
        "attempted_writes": 0,
        "proposals": plan,
    }


def main():
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    logger.warning(
        "Соединение с %s не открыто: live routing доступен только через "
        "supervised authority/readback/rollback workflow",
        ip,
    )
    auto_test_routing(channel=1, dante_output=1)
    return 2


if __name__ == "__main__":
    sys.exit(main())
