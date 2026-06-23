# ADR: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface Value Gate

## 1. Context

The no-WING handoff summary renders manifest self-consistency fields from the
detached artifact manifest. A presence-only check proves the lines exist but not
that the values still match the fail-closed contract.

## 2. Options considered

- Keep only required-line checks.
- Add a dedicated value check for the self-consistency summary surface.
- Require operators to inspect `artifact-manifest.json` manually every time.

## 3. Decision

Add a dedicated value check for manifest self-consistency fields on both
`summary.json` and `summary.md`.

## 4. Why this won

The summary pair is the handoff entrypoint. Checking expected values there
prevents stale or edited summary surfaces from implying readiness closure or live
authority.

## 5. Rejected alternatives

Required-line checks miss wrong values. Manual sidecar inspection stays useful,
but it should not be the only way to catch an unsafe summary surface.

## 6. Implementation plan

- Define diagnostic summary-surface value-gate fields.
- Compare `summary.json` and `summary.md` lines against expected
  self-consistency values.
- Fail closed with explicit mismatch lists.
- Add regression coverage for stale live-authority values.

## 7. Test plan

- Targeted no-WING summary value-gate tests.
- no-WING quick artifact inspection.
- Focused WING/no-WING/Director tests.
- Full standard pytest.

## 8. Risks and rollback

Risk: generated summary consumers may need to ignore additional diagnostic
fields. Rollback: remove the value-gate helper, fields, and tests. The rollback
does not touch WING, mixer state, readiness closure, or live authority.
