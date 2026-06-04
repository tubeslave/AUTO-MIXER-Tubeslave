# Supervised Promotion Evidence Bridge Brief

Task id: supervised-promotion-evidence-bridge-20260604
Date: 2026-06-04

## Context

The proof artifact report now summarizes dry-run and physical fader/gain proof
artifacts, but the report still needs to be tied to the safety coverage rows
that govern live readiness. Without this bridge, a future safety-matrix update
could be based on manual interpretation instead of a tested evidence contract.

## Objective

Add a read-only bridge from proof artifact reports to manual safety-row review
candidates for `fader_supervised` and `gain_supervised`.

## Safety Boundary

- Do not edit safety coverage rows automatically.
- Do not widen live write policy.
- Do not mark Full Autonomous ready.
- Do not import mixer transports or create console writes.
- Require manual safety review even when physical proof candidates exist.

## Expected Files

- `backend/supervised_promotion_evidence.py`
- `tests/test_supervised_promotion_evidence.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `backend/processing_registry.py`
- `Docs/adr/supervised-promotion-evidence-bridge-20260604.md`

## Test Plan

- Compile touched Python files.
- Run proof report, promotion evidence, registry, and no-WING focused tests.
- Run broader readiness/write-surface tests.
- Run the standard full pytest command.
