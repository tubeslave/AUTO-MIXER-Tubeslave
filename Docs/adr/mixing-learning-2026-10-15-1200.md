# ADR: Parallel drum send test requires render-path latency manifest

- **Date:** 2026-10-15
- **Update:** `ML-2026-10-15-1200`
- **Status:** accepted for research queue; no automatic application

## Decision

Revise `RULE-ML-20261015-0900-01` and `EXP-ML-20261015-0900-01` so the snare-top send comparison is valid only with a frozen, documented offline render path. Record DAW/version, sample rate, buffer, PDC state, monitoring/low-latency state, routing graph, processor modes and reported latency. Run a separate latency preflight before creative listening.

## Rationale

Official Ableton and Logic documentation shows that modern DAWs normally compensate reported path latency, while routing and monitoring modes can create exceptions. Equal processor counts do not imply equal latency. A/B variants must therefore share the same actual export path, and low-latency monitoring must not silently disable sends or bypass processors.

## Consequences

- The existing `0/−3 dB` send test remains `not_run`, `auto_apply:false`.
- No matching dummy plug-in is added merely to equalize stage count.
- Latency preflight and musical preference testing remain distinct evidence stages.
- Any undocumented PDC/routing state cancels the experiment.

