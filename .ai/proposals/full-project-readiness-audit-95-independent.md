# Proposal: full project readiness audit to 95, independent review

## 1. Short thesis

The current checkout is not safe to score from its green tests. First restore a
single evidence-bound authority model, then make test and proof artifacts
immutable enough to support release claims.

## 2. What I understood

The M006 implementation added a capable transaction coordinator and real-WING
family proofs, but runtime permission was promoted in static code while the
owner packet remains only `ready_for_owner_review`. The proof runner can execute
without an explicit live-write acknowledgement, and tests write synthetic rows
into the same ignored append-only log cited by the release report.

## 3. Proposed solution

1. Treat family proof results as `hardware_proven`, never `live_allowed`.
2. Require an independently validated owner/readiness decision before normal
   supervised execution.
3. Make the family proof tool dry-run by default and require typed confirmation
   of WING IP, family, target scope, and approval id before connecting.
4. Redirect test audit roots to per-test temporary directories.
5. Export release evidence into immutable run-scoped bundles with hashes; do
   not cite the mutable global runtime log.
6. Reconcile `STATE.json`, milestone, blockers, roadmap, and reports only after
   the executable gate matches the evidence state.

## 4. Likely files to touch

- Family registry, coordinator, proof runner, tests/conftest
- M006 report and autopilot truth surfaces
- A verifier/exporter for run-scoped transaction evidence if no existing tool
  can provide it

## 5. Alternatives considered

- Accept current proof summaries as owner promotion. Rejected because every
  summary says `live_authority_granted=false`.
- Rely on the M006 report text. Rejected because text is not a gate and cites a
  mutable log containing unit events.

## 6. Risks

- Existing UI and tests that expect green `live_allowed` rows will fail and
  must be corrected.
- Historical proof files remain valuable evidence but need clearer semantics;
  rewriting them would damage provenance.

## 7. Test plan

- Test that importing/running the proof tool without execute confirmation never
  constructs `WingClient`.
- Test that all unpromoted families block normal execution but remain eligible
  for explicit local proof runs.
- Compare canonical audit-log size/hash before and after focused and full tests.
- Validate cross-surface state fields and authority denials.

## 8. Where the other agent may disagree

Codex may want to retain narrow fader/gain normal authority from earlier proof.
The independent recommendation is stricter: current safety matrices and owner
decision artifacts still deny all standing live kinds, so the executable gate
should do the same until those surfaces are deliberately reconciled.
