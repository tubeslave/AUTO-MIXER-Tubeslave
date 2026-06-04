# Supervised Proof Artifact Evaluator Brief

Task id: supervised-proof-artifact-evaluator-20260604
Date: 2026-06-04

## Context

The autonomy promotion readiness snapshot exposes that fader is narrow
supervised-only, while gain and broader action families are still blocked by
missing proof. The next safe step is not a live write expansion; it is a pure
artifact evaluator that can review dry-run and physical proof summaries and
fail closed on missing pre-read, readback, rollback, emergency-stop, operator
approval, or idempotency evidence.

## Objective

Add a read-only evaluator for supervised fader/gain proof artifacts. It should
distinguish dry-run readiness from physical supervised promotion candidates and
never mark Full Autonomous ready.

## Safety Boundary

- No live console writes.
- No mixer transport imports.
- No proposal queue insertion.
- No new executor methods.
- Only fader/gain proof artifacts may be considered; all other console action
  families block.

## Expected Files

- `backend/supervised_proof_artifact_evaluator.py`
- `backend/processing_registry.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `tests/test_processing_registry.py`
- `Docs/adr/supervised-proof-artifact-evaluator-20260604.md`

## Test Plan

- Compile touched Python modules.
- Run focused evaluator and registry tests.
- Run broader promotion/read/write/safety boundary tests.
- Scan the evaluator for live transport and direct write tokens.
- Smoke-check dry-run versus physical proof outcomes.
