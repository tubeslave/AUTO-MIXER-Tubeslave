#!/usr/bin/env python3
"""
AUTO-MIXER Tubeslave — Quick Start Script.

Launches the automatic soundcheck engine with dLive + SoundGrid defaults.
Run from the project root:

    python3 start_soundcheck.py

Or with custom options:

    python3 start_soundcheck.py --ip 192.168.3.70 --audio-device soundgrid
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from auto_soundcheck_engine import main

if __name__ == "__main__":
    main()
