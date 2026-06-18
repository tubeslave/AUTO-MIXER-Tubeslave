# Evaluation Framework

Status: baseline framework, thresholds pending approval

## Evidence classes

- Static/doc consistency checks.
- Unit tests.
- Integration tests.
- Offline audio analysis and render checks.
- Replay evaluation.
- Simulator validation.
- Fault-injection checks.
- Real hardware read validation.
- Supervised live validation.
- Listening and expert assessment.

## Required metrics

Future completion evidence must cover:

- LUFS and level-plane behavior.
- True peak and clipping/overload statistics.
- Masking and intelligibility metrics.
- Reference-mix comparison.
- Repeatability and stability.
- Control latency.
- Parameter oscillation rate.
- Recovery, rollback and fail-closed behavior.
- Blind listening tests and expert assessment.

## Threshold policy

No final quality thresholds are approved in this setup. Candidate thresholds may
be measured and proposed, but every candidate threshold remains
`PENDING_APPROVAL` until explicitly approved.

## Current baseline commands

- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`
- `python3 tools/memory/memory_lint.py .`
- Milestone-specific commands from `Docs/plans/`

## Hardware evidence rule

Only real WING/Rack artifacts captured under an approved runbook and reviewed
by the operator can move a capability to `HARDWARE_READ_VERIFIED` or
`SUPERVISED_LIVE_VERIFIED`.

