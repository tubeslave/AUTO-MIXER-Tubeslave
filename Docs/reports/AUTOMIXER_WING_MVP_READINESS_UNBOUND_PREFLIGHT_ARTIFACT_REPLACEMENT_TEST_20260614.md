# WING MVP readiness unbound preflight artifact replacement test

Date: 2026-06-14

## Summary

The autonomy readiness regression now treats an unbound WING read-only preflight
artifact as `readonly_preflight_artifact_replacement_required`, matching the
current evidence builder contract.

## Safety impact

- Grants no live authority.
- Keeps readiness updates blocked for incomplete WING proof artifacts.
- Leaves the artifact-bound positive candidate path covered by a separate test.

## Validation target

The targeted test now expects zero valid candidates, one invalid candidate, no
readiness update candidate, and `artifact_root_missing` in the readiness
violations.
