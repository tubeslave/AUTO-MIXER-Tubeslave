# Codex Proposal: WING MVP no-WING Artifact Manifest Self-Consistency

## 1. Short thesis

Add a detached manifest self-consistency verdict so the no-WING artifact bundle
can prove its sidecar declarations match its actual artifact rows.

## 2. What I understood

The no-WING bundle is useful only if an operator can trust what it says about
the generated handoff artifacts. It must also preserve the boundary that offline
evidence does not close readiness or authorize live control.

## 3. Proposed solution

Add `build_no_wing_artifact_manifest_self_consistency_check()` and include its
verdict fields in `artifact-manifest.json`. The check compares declared and
actual artifact IDs, paths, SHA fields, SHA availability, top-level SHA values,
and readiness/live-authority denial flags.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- Council docs and WING MVP report/index entries

## 5. Alternatives considered

Only asserting this in tests was rejected because operators and later tooling
need the verdict in the generated artifact itself. Adding another standalone
manifest validator was rejected because the manifest writer already has all the
required state.

## 6. Risks

The main risk is adding fields that drift from the generated artifact layout.
The targeted regression test covers stale artifact rows.

## 7. Test plan

- Targeted no-WING manifest tests.
- Focused no-WING, Project Director, and WING MVP rails tests.
- Full standard pytest.

## 8. Where the other agent may disagree

Kimi may prefer not to add more manifest fields. I prefer explicit fields here
because they make the generated handoff bundle self-auditing without requiring a
reader to reconstruct the invariant by hand.
