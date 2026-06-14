# Codex Proposal: WING MVP Proof Plan Canonical Prestart Helper

## Short thesis

Use the canonical prestart-trace helper in the no-WING `next_wing_proof_plan`
gate so stale or partial accepted-candidate rows cannot unlock supervised
proof planning.

## What I understood

The project is still blocked on real WING read-only connection proof. Offline
work is useful only when it strengthens the path to trustworthy real artifacts
without claiming live readiness.

## Proposed solution

- Replace the local partial accepted-candidate prestart validation in
  `validate_supervised_proof_report_next_wing_plan` with
  `wing_readonly_preflight_accepted_candidate_prestart_source_ids`.
- Compare the ledger-reported violations map against
  `wing_readonly_preflight_accepted_candidate_prestart_violations_by_source`.
- Update the valid test fixture so it includes the full expected-artifact and
  prestart-summary contract.
- Add a regression test proving expected-artifact binding failures are caught.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

- Keep the local partial validation: rejected because it can diverge from the
  shared contract and under-check the trace.
- Duplicate the full contract inline: rejected because it repeats logic already
  maintained in `backend/wing_readonly_preflight_contract.py`.

## Risks

The gate becomes stricter and stale fixtures/reports without full trace fields
will fail. That is desired for the WING proof rail.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py`
- Focused WING rail pytest.
- Full standard pytest.

## Where the other agent may disagree

Kimi may ask to move more validation callers to the same helper in this commit.
That may be useful, but this patch keeps the change focused on the release gate
that decides whether proof-plan output is trustworthy.
