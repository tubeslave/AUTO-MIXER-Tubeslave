# Codex Proposal: M117 Operator Diagnostic Truth Contract

## 1. Short thesis

Make the truth surfaces distinguish read-only operator-attended diagnostic
contact from unattended/live authority, and pin that distinction in one narrow
regression test.

## 2. What I understood

M116 legitimately contacted the real WING, backend and CoreAudio with the
operator present, but sent no console writes and granted no standing authority.
The ledger and tests model this correctly. A historical paragraph in
`Docs/CURRENT_STATE.md` still collapses `hardware_contacted` and `live_writes`
into one always-false rule, while three current summaries omit the provenance
qualifier on the zero-signal observations.

## 3. Proposed solution

- Correct only the current M116/current-contract prose in the canonical truth
  surfaces.
- State that unattended/control-plane rows remain no-contact, while the exact
  operator-approved diagnostic branch may record read-only contact.
- Require all M116 zero-signal capture wording to say `operator-observed` and
  `not independently machine-verified`.
- Extend the existing M116 truth rail rather than create a new generalized
  authority mechanism.

## 4. Likely files to touch

- `Docs/CURRENT_STATE.md`
- `Docs/ACTIVE_MILESTONE.md`
- `Docs/ROADMAP.md`
- `ops/autopilot/BLOCKERS.md`
- `tests/test_wing_mvp_rails.py`
- Council records and the final checkpoint truth surfaces

## 5. Alternatives considered

- Leave prose historical: rejected because it sits under unresolved/current
  truth and directly contradicts the latest authority row.
- Change the ledger back to no-contact: rejected because that would falsify the
  real operator-attended diagnostic.
- Generalize tests for arbitrary hardware diagnostics: rejected as broader than
  the one proven M116 evidence class.

## 6. Risks

- Accidentally implying that read-only contact grants live authority.
- Overstating operator observations as machine-verified evidence.
- Invalidating the dynamic CP05 source binding after tracked edits.

## 7. Test plan

- Add focused assertions for typed diagnostic contact and provenance wording.
- Run focused M116 rails, all WING MVP rails and the standard full suite.
- Rebind CP05 in a marker-only commit after the non-marker checkpoint commit.

## 8. Where the other agent may disagree

The other reviewer may prefer touching only `CURRENT_STATE.md`. I expect the
omitted provenance qualifier across the other current handoff surfaces to be a
real evidence-class inconsistency worth fixing in the same narrow checkpoint.
