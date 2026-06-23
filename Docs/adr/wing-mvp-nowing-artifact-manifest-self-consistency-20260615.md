# ADR: WING MVP no-WING Artifact Manifest Self-Consistency

## 1. Context

The no-WING release check writes `summary.json`, `summary.md`, and
`artifact-manifest.json`. The manifest sidecar is the detached record tying the
operator-readable and machine-readable summaries together. It should prove its
own declarations match the actual artifact rows it contains.

## 2. Options considered

- Keep only individual required-artifact fields.
- Add manifest self-consistency verdicts inside `artifact-manifest.json`.
- Add a separate post-generation validation artifact.

## 3. Decision

Add self-consistency verdicts directly to `artifact-manifest.json`.

## 4. Why this won

The manifest already contains the required IDs, paths, SHA fields, SHA values,
and safety-denial fields. Computing the verdict in the same writer keeps the
diagnostic local to the artifact an operator or automation will inspect.

## 5. Rejected alternatives

Individual fields without an aggregate verdict force readers to reconstruct the
invariant manually. A separate validator would duplicate the same state and
could drift from the writer.

## 6. Implementation plan

- Add a manifest self-consistency helper.
- Write the required artifact SHA field list into the detached manifest.
- Add self-consistency fields to the manifest payload.
- Add regression coverage for stale artifact rows.

## 7. Test plan

- Targeted no-WING manifest tests.
- Focused WING/no-WING/Director tests.
- Full standard pytest.

## 8. Risks and rollback

Risk: generated manifest consumers may need to ignore additional fields.
Rollback: remove the self-consistency helper, fields, and tests. The rollback
does not touch live mixer behavior.
