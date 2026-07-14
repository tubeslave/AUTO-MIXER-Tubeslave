# ADR M109 — isolate static no-WING contract tests from live artifact history

Later bounded supersession: ADR M110 permits only the seven exact run IDs and
twelve exact payload hashes recorded in
`artifacts/wing_readonly_preflight_invalid/relocation-manifest-20260714.json`
to move byte-for-byte into immutable negative provenance. All other M109
production intake and fail-closed rules remain in force.

## Evidence-driven amendment — 2026-07-13

The initial three repaired tests passed, then the complete module exposed one
additional positive aggregate test:
`test_no_wing_release_check_runs_gap_rollup_schema_after_report`. Its synthetic
release run mocks command execution and an accepted review ledger, expects an
all-passed result, but still discovered the mutable canonical preflight root.

The decision now permits the same opt-in empty-evidence helper in exactly four
positive tests. This is the smallest change that makes all synthetic positive
paths artifact-history-hermetic. Production discovery, deterministic negative
tests and canonical physical evidence remain unchanged.

## 1. Context

HEAD `abf6b5d5e494bfa5343fab40424d456ea4339ec9` fails the mandatory test suite
because three positive runner-contract tests and one positive aggregate
orchestration test call the no-WING validator against the repository's
canonical real-WING evidence root. That root truthfully contains seven failed
preflight attempts, so the production validator correctly returns
`wing_readonly_preflight_invalid_artifacts_present`.

The three static tests assert constant agreement for proof-runner identity,
run-summary status/reason, and supported value types. The aggregate test checks
a fully synthetic release-run schema. Their dependency on mutable hardware
history is accidental.

Kimi CLI was requested as the independent Council architect but returned HTTP
402 while verifying membership. A fresh read-only Council agent supplied the
independent proposal and critique instead.

## 2. Options considered

1. Delete or relocate failed hardware artifacts.
2. Change production intake to ignore invalid artifacts or select the latest
   valid artifact.
3. Add a production dependency-injection argument for tests.
4. Add an autouse empty-evidence fixture.
5. Add one explicitly called test-local helper for exactly three positive
   constant-contract tests and one positive aggregate orchestration test.

## 3. Decision

Choose option 5.

The helper first builds an evidence object with
`artifact_payloads=(), artifact_roots=()`, then monkeypatches
`release_check.build_wing_readonly_preflight_evidence` to return that prebuilt
object. It is called only from the four named positive tests.

No production file, validator default, evidence policy, or canonical artifact
changes.

## 4. Why this won

- It restores test hermeticity at the narrowest correct boundary.
- It preserves failed real-WING attempts as immutable provenance.
- It preserves the deliberate fail-closed production aggregate.
- Prebuilding the evidence avoids recursive monkeypatch calls.
- Opt-in use avoids hiding artifact-sensitive tests.
- Existing deterministic invalid-evidence tests continue to prove the safety
  rail without freezing mutable repository history.

## 5. Rejected alternatives

- Deleting/moving artifacts loses or changes evidence provenance.
- Latest-valid-only production selection weakens a deliberate release gate.
- Production dependency injection expands a safety-critical API only for test
  convenience.
- An autouse fixture has excessive blast radius.
- A new integration test that expects today's exact `13/6/7` artifact counts
  would be brittle and would recreate the mutable-history dependency.

## 6. Implementation plan

1. Add `_inject_empty_wing_readonly_preflight_evidence(monkeypatch)` in
   `tests/test_no_wing_release_check.py` near the existing empty-evidence test.
2. Add `monkeypatch` and call the helper in:
   - `test_no_wing_release_check_tracks_wing_preflight_proof_runner_contract`;
   - `test_no_wing_release_check_tracks_wing_preflight_run_summary_contract`;
   - `test_no_wing_release_check_tracks_wing_preflight_evidence_allowed_value_types`;
   - `test_no_wing_release_check_runs_gap_rollup_schema_after_report`.
3. Do not touch adjacent negative mismatch tests.

## 7. Test plan

1. Run the four repaired positive tests.
2. Run explicit invalid-artifact and authority-claim fail-closed tests.
3. Run all of `tests/test_no_wing_release_check.py`.
4. Run `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.
5. Run `git diff --check` and inspect status/diff.
6. Confirm default canonical validation still reports the real invalid-artifact
   violation; treat its current counts as evidence, not a frozen assertion.

## 8. Risks and rollback

Risk: the helper could later be reused in artifact-intake tests and hide a
regression. Mitigation: its name is explicit, it is not a fixture, and this ADR
limits use to four named tests.

Risk: a wrong monkeypatch namespace would leave the tests coupled to canonical
evidence. The targeted tests and dirty canonical root make that mistake fail
immediately.

Rollback: revert only the helper and four call sites. Production behavior and
real artifacts are never modified, so no console or evidence rollback is
needed.
