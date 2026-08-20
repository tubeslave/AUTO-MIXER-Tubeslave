#!/usr/bin/env python3
"""Build review-only WING output-routing proposals.

Live routing requires the canonical supervised authority, readback, and
rollback workflow. This legacy CLI never opens a transport or mutates a mixer.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROUTING_LIVE_APPLY_BLOCK_REASON = "routing_live_apply_requires_supervised_authority"


def route_channels(client, output_group: str, start_output: int, num_outputs: int,
                   source_group: str, start_source_channel: int):
    """Return sequential output-route proposals without using ``client``."""

    del client
    proposals = [
        {
            "output_group": output_group,
            "output_number": start_output + offset,
            "source_group": source_group,
            "source_channel": start_source_channel + offset,
        }
        for offset in range(max(0, num_outputs))
    ]
    logger.warning(
        "Live bulk routing blocked: %s; generated %s proposal rows",
        ROUTING_LIVE_APPLY_BLOCK_REASON,
        len(proposals),
    )
    return {
        "status": "blocked",
        "reason": ROUTING_LIVE_APPLY_BLOCK_REASON,
        "live_authority_granted": False,
        "live_write": False,
        "attempted_writes": 0,
        "proposals": proposals,
    }


def main():
    """Основная функция"""
    if len(sys.argv) < 7:
        print("Использование:")
        print("  python3 route_channels.py <IP> <OUTPUT_GROUP> <START_OUTPUT> <NUM_OUTPUTS> <SOURCE_GROUP> <START_SOURCE_CHANNEL>")
        print("")
        print("Параметры:")
        print("  IP                  - IP адрес пульта (по умолчанию: 192.168.1.102)")
        print("  OUTPUT_GROUP        - OUTPUT GROUP (куда): MOD, CRD, LCL, AUX, A, B, C, SC, USB, AES, REC")
        print("  START_OUTPUT        - Начальный номер выхода")
        print("  NUM_OUTPUTS         - Количество выходов")
        print("  SOURCE_GROUP        - SOURCE GROUP (откуда): CRD, PLAY, CH, AUX, BUS, MAIN, MTX, SEND, MON, USR, OSC")
        print("  START_SOURCE_CHANNEL - Начальный номер канала источника")
        print("")
        print("Примеры:")
        print("  # DANTE выходы (MOD 1-24) <- WLIVE PLAY каналы (CRD 1-24):")
        print("  python3 route_channels.py 192.168.1.102 MOD 1 24 CRD 1")
        print("")
        print("  # Локальные выходы (LCL 1-8) <- каналы пульта (CH 1-8):")
        print("  python3 route_channels.py 192.168.1.102 LCL 1 8 CH 1")
        print("")
        print("  # USB выходы (USB 1-4) <- BUS шины (BUS 1-4):")
        print("  python3 route_channels.py 192.168.1.102 USB 1 4 BUS 1")
        return 1
    
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
    output_group = sys.argv[2].upper()
    start_output = int(sys.argv[3])
    num_outputs = int(sys.argv[4])
    source_group = sys.argv[5].upper()
    start_source_channel = int(sys.argv[6])
    
    logger.warning("Соединение с %s не открыто", ip)
    route_channels(
        client=None,
        output_group=output_group,
        start_output=start_output,
        num_outputs=num_outputs,
        source_group=source_group,
        start_source_channel=start_source_channel,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
