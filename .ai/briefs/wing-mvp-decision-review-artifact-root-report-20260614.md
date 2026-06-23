# Brief: WING MVP Decision Review Artifact Root Report

Date: 2026-06-14
Task ID: wing-mvp-decision-review-artifact-root-report-20260614

## Context

Review decisions now validate `artifact_root` against the accepted read-only
preflight candidate. The next downstream surface is the decision-review report
and no-WING summary, where operators and release gates inspect whether the
accepted candidate can unlock supervised proof planning.

## Goal

Expose candidate `artifact_root` in the WING read-only decision-review report
and no-WING summary contract so the manual root binding remains visible beyond
the internal ledger.

## Safety Boundary

- No WING hardware contact.
- No backend/WING WebSocket opening.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Expected Validation

- Focused decision-review tool tests.
- Full no-WING release-check module.
- Project Director/WING rails.
- Compile, memory lint, and whitespace diff.
