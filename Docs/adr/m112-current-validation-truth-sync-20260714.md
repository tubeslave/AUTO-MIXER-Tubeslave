# ADR — M112 Current Validation Truth Sync

## 1. Context

The committed M111/M005 migration baseline and its validation advanced beyond
the canonical control-plane reports. CP05 binds all committed non-marker
content, so a normal truth edit invalidates the marker pair.

## 2. Options considered

One mixed commit, self-referential commit hashes, leaving the latest capsule
ephemeral, or a non-marker commit followed by a marker-only rebind.

## 3. Decision

Use the two-commit sequence. Record `451d0fdf3` only as the validated
product/runtime baseline and use dynamic CP05 verification for exact current
tree identity. Track exactly three canonical capsule files.

## 4. Why this won

It is finite, auditable, preserves launch provenance, avoids recursive hashes
and keeps generated evidence outside the canonical source tree.

## 5. Rejected alternatives

The other options either make CP05 stale, create an impossible self-reference,
or leave canonical truth dependent on untracked local evidence.

## 6. Implementation plan

Sync all non-marker truth and parser surfaces, validate and commit; then update
only the two CP05 markers and commit them separately.

## 7. Test plan

Parse all JSON/JSONL, verify the capsule manifest, run focused rail tests, the
150 WING/static rails, the standard full suite, diff/memory checks and CP05 on
the final clean committed tree.

## 8. Risks and rollback

Any non-marker edit after rebind invalidates CP05. Roll back by reverting the
M112 non-marker commit and then recomputing/recommitting the marker pair. This
ADR grants no WING contact or live authority.
