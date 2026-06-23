# Automixer WING MVP: no-WING Artifact Manifest Verifier

## Summary

Added a read-only verifier for generated no-WING release-check artifact
directories:

```bash
PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py \
  path/to/no_wing_artifact_dir
```

The verifier recomputes `summary.json` and `summary.md` SHA256 values, checks
them against `artifact-manifest.json`, replays the existing manifest
cross-surface and self-consistency verdicts, and fails closed with
`status: failed` plus `failed_checks` when the bundle drifts.

The no-WING `summary.json` and operator-readable `summary.md` also expose the
verifier command and required no-authority outcomes:
`artifact_manifest_verifier_required: true`,
`artifact_manifest_verifier_command`,
`artifact_manifest_verifier_status_required: passed`,
`artifact_manifest_verifier_failed_checks_required: []`,
`artifact_manifest_verifier_contacts_wing: false`,
`artifact_manifest_verifier_mutates_console_state: false`,
`artifact_manifest_verifier_queries_console: false`,
`artifact_manifest_verifier_readiness_closure_allowed: false`, and
`artifact_manifest_verifier_live_authority_granted: false`.
The detached `artifact-manifest.json` sidecar now mirrors the same
verifier/no-authority fields. The verifier validates these fields in
`summary.json`, `summary.md`, and `artifact-manifest.json`; stale commands or
missing operator-visible verifier lines fail the artifact bundle verification.
The verifier JSON report itself also exposes the safety boundary with
`read_only: true`, `live_write: false`, `contacts_wing: false`,
`contacts_backend: false`, `mutates_console_state: false`,
`queries_console: false`, `executes_tests: false`,
`readiness_closure_allowed: false`, and `live_authority_granted: false`.
The expected verifier report fields are now published as
`artifact_manifest_verifier_report_required_fields` in `summary.json`,
`summary.md`, and `artifact-manifest.json`; the verifier fails closed if any
surface changes that required-field list. The verifier report now also checks
its own final JSON against that list via
`verifier_report_required_fields_present`,
`verifier_report_required_fields_missing`, and
`verifier_report_required_fields_match_contract`. Missing bundles and JSON load
errors also keep the same no-authority report shape: artifact match verdicts
default to `false`, safety fields deny live authority, and the required-field
self-check surface remains present. `failed_checks_missing_check_entries: []`
is now part of the required report contract, so every `failed_checks` item must
also have a machine-checkable boolean entry in `checks`. The required report
contract now also includes `checks`,
`verifier_report_required_fields`,
`verifier_report_required_fields_missing`, and
`verifier_report_required_fields_match_contract`, so the diagnostic surface is
itself mandatory. `Docs/runbooks/WING_READONLY_PREFLIGHT.md` now requires these
same verifier-report markers before opening a backend/WING WebSocket.
`tools/wing_readonly_preflight_runner.py` also enforces the same verifier
against the `--prestart-summary` artifact directory and blocks before any
WebSocket request when the detached manifest verifier fails.

## Safety Boundary

- Does not contact WING.
- Does not query or mutate console state.
- Does not approve live writes.
- Does not close readiness.
- Requires `readiness_closure_allowed: false` and `live_authority_granted:
  false` in the verified surfaces.

## Operator Impact

Before opening the backend/WING session, run the verifier on the same passed
no-WING artifact directory that contains the prestart `summary.json`. Continue
only when the JSON report returns `status: passed` and `failed_checks: []`.
The read-only preflight runner repeats this check from `--prestart-summary` and
fails closed before WebSocket if the sidecar bundle is stale.

## Validation

- Focused verifier tests cover a valid bundle, stale `summary.md`, stale
  `summary.json` verifier command, and stale `artifact-manifest.json` verifier
  command.
- Focused verifier tests cover stale `artifact-manifest.json`
  `artifact_manifest_verifier_report_required_fields`.
- Focused verifier tests cover a future required verifier-report field that is
  not emitted by the final report.
- Missing bundle and load-error verification still return full-shape
  no-authority safety reports.
- Focused verifier tests cover explicit `checks` entries for early
  `failed_checks`.
- Focused verifier tests cover `checks` and verifier self-check fields as
  required report fields.
- WING MVP rail tests cover the read-only runbook's verifier-report marker
  checklist.
- Read-only preflight runner tests cover stale prestart sidecar manifests
  blocking before WebSocket.
- Focused verifier tests cover the CLI `main(argv)` path, including exit code
  `0` for a valid bundle, exit code `1` for a missing bundle, and JSON stdout.
- no-WING quick plus verifier CLI is part of the validation plan.
- Project Director baseline markers include the verifier command and no-authority
  verdicts.
