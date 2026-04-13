#!/usr/bin/env python3
"""
AUTO-MIXER Tubeslave — Quick Start Script.

Launches the automatic soundcheck engine. By default it scans the network
for WING (OSC broadcast) and dLive (MIDI/TCP probe) mixers, connects
to the first one found, detects the audio device, and starts auto-mixing.

    python3 start_soundcheck.py                 # full auto-discover
    python3 start_soundcheck.py --scan-only     # just list found mixers
    python3 start_soundcheck.py --ip 192.168.3.70  # direct connect
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from auto_soundcheck_engine import main

if __name__ == "__main__":
    main()
