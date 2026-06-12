# ADR: WING Read-Only Runbook Final Prestart Gate

## Context

The WING MVP handoff now requires the supervised proof transition to appear in
the final operator-sequence step and to include a valid accepted-candidate
prestart trace. The read-only preflight runbook still used older wording that
could be read as allowing a transition to supervised fader/gain proof after only
a clean read-only artifact and accepted decision JSON.

## Options considered

1. Keep the runbook unchanged and rely on no-WING release-check tooling.
2. Clarify the runbook with the same final-step prestart gate fields required by
   the release-check tooling.
3. Rewrite the read-only and supervised runbooks together.

## Decision

Use option 2. The read-only runbook must state that the next supervised runbook
is reachable only after the no-WING release handoff proves a final handoff
operator-sequence step with:

- `operator_sequence_has_supervised_prestart_gate: true`
- `operator_sequence_supervised_prestart_gate_is_final: true`
- `accepted_decision_ready`
- `valid accepted-candidate prestart trace`
- `WING_SUPERVISED_FADER_GAIN_PROOF`

## Why this won

It matches the current tooling contract without changing runtime behavior,
hardware contact, or write authority. The operator-facing runbook and automated
release-check summary now describe the same gate.

## Rejected alternatives

Keeping the old runbook wording was rejected because it preserved a weaker
manual interpretation than the automated gate. A broader rewrite was rejected as
unnecessary for this narrow safety alignment.

## Implementation plan

- Patch `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Extend `tests/test_wing_mvp_rails.py` with read-only runbook assertions.
- Add a report capturing the no-hardware scope and verification.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks_keep_readonly`
- Affected WING MVP and no-WING release-check tests.
- `python3 tools/memory/memory_lint.py`
- Full test suite.

## Risks and rollback

The main risk is stale field names if release-check summaries change. Roll back
by updating the runbook assertions and ADR to the new summary fields in the same
commit that changes the release-check schema.
