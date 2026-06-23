# ADR: WING MVP no-WING Artifact Manifest Verifier

## 1. Context

The no-WING release check writes `summary.json`, operator-readable `summary.md`,
and detached `artifact-manifest.json`. The detached manifest is intended to
prove the artifact bundle that will be reviewed before the real WING read-only
session, but the runbook currently relies on manual inspection of many fields.

## 2. Options considered

- Keep manual inspection only.
- Extend `tools/no_wing_release_check.py` with a verification mode.
- Add a separate read-only verifier CLI.

## 3. Decision

Add `tools/verify_no_wing_artifact_manifest.py` as a separate read-only
verifier, and make `tools/wing_readonly_preflight_runner.py` invoke that
verifier against `--prestart-summary` before opening any backend/WING
WebSocket.

## 4. Why this won

The operator needs to verify an existing bundle without rewriting it. A separate
CLI keeps generation and verification distinct while reusing the existing
manifest contract builders.

## 5. Rejected alternatives

Manual inspection is too easy to misread when many manifest fields are present.
Adding a mode to the writer makes the operational boundary less clear.

## 6. Implementation plan

- Load `summary.json`, `summary.md`, and `artifact-manifest.json` from the run
  directory.
- Recompute required artifact SHA256 values.
- Recompute manifest cross-surface, self-consistency, and summary-surface
  checks using existing no-WING helpers.
- Emit a JSON report and exit nonzero when any check fails.
- Surface the verifier command and no-authority expectations in no-WING
  `summary.json` and `summary.md`.
- Verify that those summary/Markdown verifier fields still match the expected
  command and no-authority contract.
- Mirror the same verifier/no-authority fields in detached
  `artifact-manifest.json` and fail verification when the sidecar command
  drifts.
- Emit top-level verifier report safety fields: `read_only: true`,
  `live_write: false`, no WING/backend contact, no console mutation/query, no
  test execution, no readiness closure, and no live authority.
- Publish `artifact_manifest_verifier_report_required_fields` in
  `summary.json`, `summary.md`, and `artifact-manifest.json`, then fail
  verification if any surface drifts from that required report-field list.
- Add a verifier-report self-check that fails verification when the final
  verifier JSON omits any field from that required report-field contract.
- Keep early verifier failures, including missing files and load errors, on the
  same no-authority report shape with false artifact-match defaults and the
  required-field self-check surface.
- Require `failed_checks_missing_check_entries: []` so every verifier
  `failed_checks` item is backed by a machine-checkable boolean in `checks`.
- Include `checks` and the verifier report self-check fields in the required
  verifier-report contract, making the diagnostic surface itself mandatory.
- Mirror the verifier report contract in `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
  so the operator preflight path requires those fields before opening a
  backend/WING WebSocket.
- Enforce the same detached-manifest verifier in the read-only preflight runner
  before any WebSocket is opened.
- Add operator runbook and Project Director marker coverage.

## 7. Test plan

- Valid no-WING bundle passes verifier.
- Stale `summary.md` fails verifier with hash mismatch checks.
- no-WING summaries expose the verifier command, required passed status, empty
  `failed_checks`, and no-authority flags.
- Stale verifier command fields in `summary.json` or missing verifier lines in
  `summary.md` fail the verifier.
- Stale verifier command fields in `artifact-manifest.json` fail the verifier.
- Stale verifier-report required-field lists in `artifact-manifest.json` fail
  the verifier.
- A future required verifier-report field that is not emitted by the final
  report fails the verifier with `verifier_report_required_fields_present`.
- Missing bundles and load errors still expose the required verifier report
  field surface while denying readiness closure and live authority.
- Missing bundles and load errors expose explicit `checks` entries for their
  `failed_checks`, and valid bundles require
  `failed_checks_missing_check_entries: []`.
- The verifier required-field list includes `checks`,
  `verifier_report_required_fields`,
  `verifier_report_required_fields_missing`, and
  `verifier_report_required_fields_match_contract`.
- The read-only runbook requires `failed_checks_have_check_entries: true`,
  `failed_checks_missing_check_entries: []`, and the required verifier-report
  self-check fields before any WING session.
- The verifier CLI `main(argv)` path is tested directly: valid bundles return
  exit code `0`, failing bundles return exit code `1`, and stdout remains the
  same JSON report surface used by the runbook.
- The read-only preflight runner blocks before WebSocket when the prestart
  no-WING artifact manifest verifier fails.
- Missing bundles still return a no-authority verifier report.
- no-WING quick artifact directory passes verifier CLI.
- Focused WING/no-WING/Project Director tests and full standard pytest.

## 8. Risks and rollback

Risk: operator flow adds one more command before the WING session. Rollback:
remove the verifier, tests, and runbook marker. This does not touch WING,
mixer state, readiness closure, or live authority.
