# Mix Agent Offline and Backend Facade

## Context

The project already contains live AutoFOH analysis, safety controllers and
offline mixing scripts, but the research-agent recommendation calls for a
separate explainable agent that can analyze stems, compare references, produce
reports and bridge safely to a real console.

## Options considered

1. Rewrite the existing live and offline pipelines as a new `mix_agent`.
2. Keep the current pipelines and add a thin facade with shared models,
   reporting, rules and a backend safety bridge.
3. Only extend `tools/offline_agent_mix.py`.

## Decision

Add a thin `mix_agent` package that wraps and complements the existing project
without replacing the live AutoFOH stack.

## Why this won

The facade gives offline users a clean CLI and report format while preserving
the live safety guarantees already encoded in `AutoFOHSafetyController`.

## Rejected alternatives

Rewriting the live stack is too risky for a real-console project. Extending only
the offline script would not address backend integration or reusable reports.

## Implementation plan

- Add typed models for analysis, issues, actions and dashboards.
- Add audio loading, metrics, masking, reference comparison and stereo analysis.
- Add explainable rules and Markdown/JSON reports.
- Add conservative offline apply.
- Add backend bridge that delegates mixer writes to the existing safety layer.

## Test plan

Run targeted tests for metrics, rules, CLI/report generation and backend bridge
translation, then run the standard suite before merge.

## Risks and rollback

The facade is additive. Rollback is removing `mix_agent/`,
`backend/mix_agent_bridge.py`, new tests and this ADR. Live console writes
remain guarded by the existing safety controller.
