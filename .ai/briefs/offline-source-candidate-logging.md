# Offline Source Candidate Logging

## Task

Connect the source-grounded learning layer to `tools/offline_agent_mix.py` so
offline EQ, compressor, pan, and FX candidates are logged with provenance and
before/after metrics.

## Safety Constraints

- Shadow/offline only.
- Do not change live OSC behavior.
- Keep `source_knowledge.enabled` false by default.
- If the logging layer fails, the offline mix render must continue.

## Acceptance Notes

- JSONL decision rows include `selected_rule_ids`, `source_ids`,
  `before_metrics`, `after_metrics`, and Codex listening-proxy feedback.
- FX buses and active sends are logged as candidates.
- Tests cover direct logging, channel EQ/comp/pan tracing, and FX candidate
  tracing.
