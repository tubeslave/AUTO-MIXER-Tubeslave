# Brief: WING MVP no-WING Plan Prestart SHA Format Gate

Date: 2026-06-14
Task ID: wing-mvp-nowing-plan-prestart-sha-format-gate-20260614

## Context

The WING read-only offline plan carries the no-WING prestart `summary.json`
path and SHA256 that later handoff packet checks compare against. The
accepted-candidate contract now rejects malformed prestart digests, but the
plan artifact validator must also fail closed before a malformed plan can feed
handoff validation.

## Goal

Reject malformed `backend_port_prestart_summary_sha256` values in the
no-WING WING read-only plan artifact and prove a handoff packet cannot pass by
matching the same malformed plan digest.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Expected Validation

- Focused no-WING plan/handoff SHA tests.
- Full no-WING release-check module.
- Project Director/WING rails.
- Compile, memory lint, and whitespace diff.
