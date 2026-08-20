#!/usr/bin/env python3
"""Build a review-only WING-to-Dante routing proposal.

This legacy CLI intentionally contains no console transport or mutation sink.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROUTING_LIVE_APPLY_BLOCK_REASON = "routing_live_apply_requires_supervised_authority"


def check_current_routing(client, channel: int):
    """Return candidate read addresses without querying a console."""

    del client
    return [
        f"/ch/{channel}/out/conn/grp",
        f"/ch/{channel}/out/conn/out",
        f"/ch/{channel}/out/grp",
        f"/ch/{channel}/out/out",
    ]


def route_channels_to_dante(client, start_channel: int = 1, num_channels: int = 24,
                            dante_start_output: int = 1, card_name: str = "WLIVE"):
    """Return sequential MOD <- CRD proposals without using ``client``."""

    read_candidates = check_current_routing(client, start_channel)
    proposals = [
        {
            "output_group": "MOD",
            "output_number": dante_start_output + offset,
            "source_group": "CRD",
            "source_channel": start_channel + offset,
            "card_name": card_name,
        }
        for offset in range(max(0, num_channels))
    ]
    logger.warning(
        "Live Dante routing blocked: %s; generated %s proposal rows",
        ROUTING_LIVE_APPLY_BLOCK_REASON,
        len(proposals),
    )
    return {
        "status": "blocked",
        "reason": ROUTING_LIVE_APPLY_BLOCK_REASON,
        "live_authority_granted": False,
        "live_write": False,
        "attempted_writes": 0,
        "read_candidates": read_candidates,
        "proposals": proposals,
    }


def main():
    """Основная функция"""
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    if len(sys.argv) > 2:
        start_channel = int(sys.argv[2])
    else:
        start_channel = 1
    
    if len(sys.argv) > 3:
        num_channels = int(sys.argv[3])
    else:
        num_channels = 24
    
    if len(sys.argv) > 4:
        dante_start_output = int(sys.argv[4])
    else:
        dante_start_output = 1
    
    if len(sys.argv) > 5:
        card_name = sys.argv[5].upper()
    else:
        card_name = "WLIVE"
    
    logger.info("=" * 60)
    logger.info("Маршрутизация каналов на Dante выходы")
    logger.info("=" * 60)
    logger.info(f"IP пульта: {ip}")
    logger.info(f"Начальный канал: {start_channel}")
    logger.info(f"Количество каналов: {num_channels}")
    logger.info(f"Начальный Dante выход: {dante_start_output}")
    logger.info(f"Карта: {card_name}")
    logger.info("=" * 60)
    
    logger.warning("Соединение с %s не открыто", ip)
    route_channels_to_dante(
        client=None,
        start_channel=start_channel,
        num_channels=num_channels,
        dante_start_output=dante_start_output,
        card_name=card_name,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
