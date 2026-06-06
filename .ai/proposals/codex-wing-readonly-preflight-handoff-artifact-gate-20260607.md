# Proposal: WING Read-Only Preflight Handoff Artifact Gate

## Short thesis

Treat the operator handoff packet like a release artifact, not just a rendered
Markdown note. The no-WING gate should read it back and prove it remains a
read-only, operator-required hardware-proof handoff.

## What I understood

The project is moving toward supervised and autonomous console control, but the
current safe boundary requires real WING read-only proof before any supervised
write proof. The existing handoff packet helps operators run that proof, so
drift in its command sequence is a safety issue.

## Proposed solution

Generate the no-WING handoff packet as JSON inside the run directory, then add a
schema validator that checks:

- read-only safety flags stay true/false in the safe direction;
- preview, capture, review, and decision-review commands match the configured
  WING read-only proof flow;
- the preview command uses `--plan-only`;
- the capture command requires physical readback;
- the review command writes a pending decision template;
- the packet keeps hardware/readiness blockers open.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-readonly-preflight-handoff-artifact-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-handoff-artifact-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_HANDOFF_ARTIFACT_GATE_20260607.md`
- `Docs/INDEX.md`

## Alternatives considered

- Keep Markdown-only output: too weak for release gating.
- Add another standalone handoff validator tool: more surface area than needed.
- Validate the default `Docs/reports` handoff path: less precise than validating
  the exact artifact generated inside each no-WING run.

## Risks

The main risk is a false sense of hardware readiness. The validator should
explicitly keep `readiness_matrix_update_allowed_by_packet=false`,
`readiness_closure_allowed_by_packet=false`, and
`live_authority_granted=false`.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py`
- Focused pytest for handoff/no-WING validators.
- no-WING quick run with `/tmp` artifact root.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue that the next proof work should focus on real WING operation
instead of another offline gate. That is fair, but without hardware access this
keeps the operator path precise and prevents the release gate from accepting a
misleading handoff.
