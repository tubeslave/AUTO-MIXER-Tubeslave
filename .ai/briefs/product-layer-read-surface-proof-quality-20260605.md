# product-layer-read-surface-proof-quality-20260605

## Context

The no-WING release `summary.json` now exposes read-surface proof-quality facts:
declared fixture path count, declared pytest function count, physical gap surface
count, and enforcement flags. The Product Layer supervised proof evidence
snapshot currently reads `supervised_proof_artifact_report.json` but does not
surface sibling no-WING summary proof quality to the operator UI.

## Goal

Expose sanitized no-WING read-surface proof quality through
`supervised_proof_evidence.summary` and the existing Settings module status row.

## Non-goals

- Do not generate artifacts from Product Layer.
- Do not query WING or mixer hardware.
- Do not treat summary proof quality as promotion readiness.
- Do not widen live write authority.

## Safety constraints

- Source summary values must be normalized and fail closed.
- `full_autonomous_ready` remains false.
- Physical gap counts stay visible.
- Missing summary files must not block supervised proof artifact display.

## Expected evidence

- Product Layer snapshot reads sibling `summary.json` as read-only metadata.
- Summary fields show enforcement flags and physical gap count.
- Frontend module status row exposes the proof-quality facts.
- Tests prove no live-write fields are introduced.
