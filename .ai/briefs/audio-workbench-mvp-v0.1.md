# Audio Workbench MVP hardening

## Goal
Turn PR #99 from a collection of analysis components into a fail-closed DAWless MVP that can render aligned multitracks, run reversible experiments, preserve overload evidence, refuse incomplete verification, expose runtime health, and hand accepted bounded actions to the existing Automixer safety bridge.

## Required outcomes
- CI green on Python 3.10/3.11/3.12.
- Unknown renders and empty evidence cannot finalize.
- A/B rejects length/channel mismatches.
- Source audio cannot be overwritten.
- Candidate/offline renders do not silently clip.
- Plugin execution is denied without an exact enabled calibration record.
- DAWless aligned multitrack renderer exists.
- Runtime doctor exists.
- Accepted actions can be translated into existing MixAgentBackendBridge structures without applying them.
- Learned observers remain optional and non-authoritative until calibrated.

## Non-goals
Ableton automation, live mixer writes, and learned-judge authority are not required for v0.1.
