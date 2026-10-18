# ADR: Compression Director v1.1 measured-GR calibration

## Problem
Compression Director v1 derives threshold from a static gain-computer formula. Real p95 gain reduction also depends on RMS integration, attack/release and source timing, so requested and rendered GR diverge. The v1 event analyzer also reports attack/recovery values without saying when the scan ended at a source, next-event or analysis-window boundary.

## Decision
Keep v1 intact and add a wrapper module. v1.1 calibrates only threshold by repeatedly running the accepted `LinkedCompressor` against a fixed RAW-active sample mask. Bisection targets actual active p95 GR. Ratio, attack, release, knee, RMS integration and max-GR are immutable during calibration. Unreachable targets are reported rather than weakening safety limits.

The new timing evidence repeats the v1 macro-envelope peak detector and reports attack/recovery censor counts and reasons. It does not claim transcription, note-level onset accuracy or subjective quality.

## Acceptance boundary
Machine acceptance requires deterministic tests, full exact-branch CI and real-source evidence. Musical preference remains pending human listening. No candidate can become a baseline automatically.

## Rollback
The implementation is additive (`compression_director_v11.py`). Reverting it restores v1 without touching the compressor core, live code, mastering or delivered audio.
