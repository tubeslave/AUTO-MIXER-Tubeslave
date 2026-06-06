# ADR: Read-Surface Acceptance Template Artifact Gate

## Context

The physical-readback review path includes an acceptance-template builder that
creates pending operator records for valid read-surface candidates. The no-WING
release check generated this JSON artifact, but did not validate the generated
file before continuing to later gates.

## Options considered

1. Keep relying on unit tests for the template builder.
2. Let the acceptance ledger be the first consumer to reject bad records.
3. Validate the generated acceptance-template artifact inside no-WING release
   checks.

## Decision

Use option 3. The release check now validates
`read_surface_physical_readback_acceptance_template.json` immediately after the
template builder runs.

## Why this won

The generated artifact is an operator handoff at a safety boundary. Validating
it in the active release run catches drift in safety flags, pending-placeholders,
record counts, operator-required edits, and fixture/live authority claims before
the summary can pass.

## Rejected alternatives

- Builder-only tests do not prove the release artifact path.
- Ledger-only rejection happens too late for release summary clarity.
- Automatic fixture closure from an accepted-looking template is unsafe and out
  of scope.

## Implementation plan

1. Add a no-WING schema validator for the generated template JSON.
2. Validate top-level safety flags and record-level false authority flags.
3. Require pending operator placeholders in generated template records.
4. Add the validator result to no-WING `summary.json` and `summary.md`.
5. Cover valid, missing, and unsafe artifacts in tests.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py tools/build_read_surface_physical_readback_acceptance_template.py`
- Focused no-WING pytest for template artifact validation.
- no-WING quick run.
- Memory lint and diff check.

## Risks and rollback

The validator must not imply that a template is physical proof. It keeps
fixture closure, execution permission, live widening, live authority, and full
autonomous readiness false. Rollback is to remove the validator, tests, and
summary/report additions; no mixer transport path is affected.
