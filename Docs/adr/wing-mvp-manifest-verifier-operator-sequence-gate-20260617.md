# WING MVP Manifest Verifier Operator Sequence Gate

## Context

The WING read-only preflight handoff packet already declares an operator step
that requires `tools/verify_no_wing_artifact_manifest.py` to pass before the
operator previews or opens the backend/WING preflight path. The no-WING release
validator checked the broader operator sequence contract, but it did not expose
or fail closed on the manifest verifier step as its own declared gate.

## Options considered

1. Rely on the full operator sequence SHA and contract match.
2. Add a dedicated manifest-verifier operator-step gate inside the no-WING
   handoff packet validator.

## Decision

Add a dedicated validator gate for the manifest-verifier operator step. The
gate recomputes the step index and text from `operator_sequence`, compares them
with the packet's declared fields, requires the step to appear before
`preview_readonly_request_plan`, and verifies the required marker list.

## Why this won

The manifest verifier is the last detached no-WING artifact-bundle integrity
check before any backend/WING WebSocket path. Treating it as a named gate gives
operators and Project Director output a precise failure reason instead of a
generic operator sequence mismatch.

## Rejected alternatives

Keeping only the sequence SHA was rejected because a sequence mismatch does not
tell the operator whether the manifest verifier was skipped, moved after
preview, or merely renamed.

## Implementation plan

- Extend `tools/no_wing_release_check.py` handoff packet validation with
  manifest-verifier operator-step detail fields and fail-closed violations.
- Update the no-WING handoff fixture to include the declared manifest verifier
  fields.
- Add a regression that moves the verifier step after preview and requires the
  dedicated violations.

## Test plan

- Focused handoff/no-WING tests.
- Fresh no-WING quick artifact plus detached manifest verifier.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline validation. Rollback is to remove the
dedicated manifest-verifier step checks; no WING contact or live authority is
introduced by this ADR.
