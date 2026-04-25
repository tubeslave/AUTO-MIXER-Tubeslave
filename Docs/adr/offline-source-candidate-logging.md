# Offline Source Candidate Logging

## Context

The source-grounded rule store exists, but offline mixes did not yet emit
training rows for the actual EQ, compressor, pan, and FX candidates used by the
offline renderer.

## Options Considered

1. Log candidates from the live agent path.
2. Log candidates only from the offline mix renderer.
3. Replace the offline decision path with source-rule decisions.

## Decision

Log source-grounded candidate traces from `tools/offline_agent_mix.py` only,
behind `source_knowledge.enabled` or `--source-knowledge-enable`.

## Why This Won

It creates usable training/evaluation data without changing live OSC behavior,
mixer safety, or the current offline render decisions.

## Rejected Alternatives

Live-path logging was deferred because it raises latency and safety questions.
Replacing decisions was rejected because the layer is not yet a trusted policy.

## Implementation Plan

- Replay the channel strip for logging and capture EQ/comp/pan before/after
  metrics.
- Log FX return and send candidates while rendering shared FX buses.
- Store selected rule IDs, source IDs, action payloads, metrics, and Codex
  listening-proxy feedback in JSONL.

## Test Plan

- Unit test direct source candidate logging.
- Unit test channel EQ/comp/pan tracing.
- Unit test FX candidate logging.
- Run the standard pytest command before merge.

## Risks And Rollback

The risk is extra offline CPU cost when enabled. Rollback is to disable
`source_knowledge.enabled` or remove the CLI flag; default behavior remains off.
