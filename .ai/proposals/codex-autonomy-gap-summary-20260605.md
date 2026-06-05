# Proposal: autonomy-gap-summary-20260605

## Short thesis

Add a deterministic, read-only `promotion_gap_summary` to
`autonomy_promotion_readiness.summary` so the Product Layer can explain the
distance from current supervised/proposal-only state to broader autonomous
mixing.

## What I understood

The project is moving toward a full autonomous AI mixing engineer, but current
safe implementation must keep live console writes narrow. The readiness rows
already preserve that boundary, yet downstream surfaces need a compact summary
of why each action family is still blocked.

## Proposed solution

In `backend/autonomy_promotion_readiness.py`:

- categorize row blockers into stable buckets: capability, safety, proof,
  read surface, write surface, autonomy mode, and other;
- include evidence requirement IDs as their own summary bucket;
- expose `blocked_target_ids_by_bucket`;
- expose `evidence_requirement_ids_by_action_family`;
- expose `next_required_evidence_by_action_family`;
- keep the summary read-only and derived from rows only.

In `frontend/src/services/operatorAdapters.ts`, map nonzero gap buckets to
read-only `ModuleStatusRecord` rows with `domain = "promotion_gap"` and no live
write kinds.

## Likely files to touch

- `backend/autonomy_promotion_readiness.py`
- `tests/test_autonomy_promotion_readiness.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `.ai/briefs/autonomy-gap-summary-20260605.md`
- `Docs/adr/autonomy-gap-summary-20260605.md`

## Alternatives considered

Add a separate module or websocket route. That is more surface area than this
needs because the data is already in the readiness snapshot.

Push this only into the frontend adapter. That would hide the canonical Product
Layer evidence model from backend agents and tests.

## Risks

The main risk is implying promotion readiness from aggregated counts. The
summary must keep `full_autonomous_ready=false` and must not change any
row-level authority fields.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py tests/test_processing_registry.py -q`
- `npm --prefix frontend test -- --runInBand operatorAdaptersContract.test.mjs`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may prefer a separate remediation-plan module. I would keep this in the
existing snapshot first because the rows already contain the source evidence and
the immediate need is operator/agent scanability.
