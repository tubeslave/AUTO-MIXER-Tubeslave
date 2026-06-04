# ADR: Legacy Proof Artifact Classification

Date: 2026-06-04

## 1. Context

The supervised proof artifact report now consumes local proof-like summaries
from historical WING validation runs. Older `wing_operator_proposal_*`
summaries can contain `execute=true`, physical readback fields, rollback
evidence, and emergency-stop checks, but lack explicit `proof_mode` and
`physical_wing` schema fields.

## 2. Options considered

- Treat `execute=true` as physical proof.
- Treat missing mode as dry-run.
- Classify old executed WING summaries without explicit schema as blocked
  legacy evidence.

## 3. Decision

Classify old executed WING operator proposal summaries without explicit proof
schema as `legacy_ambiguous` and add the blocker
`legacy_physical_schema_missing`.

## 4. Why this won

It preserves the audit value of historical artifacts while preventing them from
inflating dry-run readiness or supervised promotion evidence. Future physical
proof runs remain accepted when they include explicit schema and physical
readback sources.

## 5. Rejected alternatives

Inferring physical proof from `execute=true` was rejected because old artifacts
were not written under the current promotion contract. Keeping the dry-run
fallback was rejected because it made executed physical-looking rows appear as
ready dry-run rehearsal. Ignoring all historical WING summaries was rejected
because blocked rows still help show what evidence is missing.

## 6. Implementation plan

1. Add a `legacy_ambiguous` proof mode for old executed
   `wing_operator_proposal` payloads without explicit mode fields.
2. Add `legacy_physical_schema_missing` as a common blocker.
3. Preserve explicit future `proof_mode=physical` artifacts when physical
   sources are present.
4. Test evaluator and report aggregation behavior.

## 7. Test plan

- Compile touched Python files.
- Run focused evaluator and report tests.
- Run broader supervised proof/readiness/no-WING tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: historical physical proof rows no longer appear dry-run-ready. That is a
safe regression because current promotion should be based on explicit proof
schema. Rollback is to remove the legacy classifier and blocker; no live write
path is introduced either way.
