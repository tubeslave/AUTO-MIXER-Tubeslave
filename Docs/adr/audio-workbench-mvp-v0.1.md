# ADR: Audio Workbench MVP hardening and Automixer integration

## Context
Audio Workbench v0.1 had useful analysis/experiment building blocks but its readiness gate could be bypassed, A/B comparison silently truncated mismatched renders, candidate rendering clipped overloads, plugin calibration was advisory only, and the project lacked a minimal multitrack renderer and explicit bridge to existing Automixer safety structures.

## Options considered
1. Keep Audio Workbench standalone and advisory.
2. Build a parallel live-control path.
3. Harden offline Workbench and bridge accepted actions into the existing MixAgentBackendBridge / AutoFOHSafetyController.

## Decision
Choose option 3. Audio Workbench is the offline laboratory and evidence ledger. Existing Automixer safety remains authoritative for live/control writes.

## Why this won
It reuses established mixer safety, prevents two competing control stacks, keeps DAWless chat mixing possible, and makes offline experiments reversible and auditable.

## Rejected alternatives
A parallel live writer was rejected because it would bypass existing safety policy. Metric-only auto-finalization was rejected because measurements are evidence, not artistic verdicts.

## Implementation plan
- fail closed on unknown renders and empty evidence;
- preserve float overload evidence and forbid source overwrite;
- reject A/B length/channel mismatches;
- enforce plugin calibration records;
- add aligned DAWless multitrack renderer;
- add runtime doctor;
- add non-applying Automixer bridge;
- keep learned observers optional and capability-gated.

## Test plan
Run repository pytest on Python 3.10/3.11/3.12 plus Audio Workbench safety regressions. Keep PR draft until CI is green.

## Risks and rollback
Offline renderer currently supports aligned mono/stereo sources, static gain/pan/polarity only. It intentionally rejects ambiguous stereo panning. Live application remains disabled through this bridge.
