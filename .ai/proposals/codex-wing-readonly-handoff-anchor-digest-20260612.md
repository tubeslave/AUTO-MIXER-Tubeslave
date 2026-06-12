# Proposal: WING Read-Only Handoff Anchor Digest

Date: 2026-06-12

## Short thesis

Hash the handoff `mvp_blocker_anchor` with a canonical JSON SHA256 and validate
the digest against both the packet anchor and the current blocker table.

## What I understood

The current proof target is still WING read-only connection proof. Offline
handoff tooling can harden proof discipline, but it cannot contact WING, grant
live authority, or close readiness.

## Proposed solution

- Add `wing_mvp_blocker_anchor_sha256(anchor)` to no-WING tooling.
- Add `mvp_blocker_anchor_sha256` to handoff packets.
- Render the digest in the handoff Markdown and no-WING summary.
- Fail closed when the digest is missing, mismatches the packet anchor, or
  mismatches the current first-open blocker anchor.
- Add regression coverage for digest mismatch.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## Alternatives considered

- Compare only the full anchor object. This already catches stale packets, but
  does not provide a compact reviewer-visible fingerprint.
- Hash the raw Markdown row. That would be brittle to whitespace and table
  formatting changes.

## Risks

- Canonical JSON field changes intentionally change the digest and require a new
  handoff packet.
- Reviewers may treat the digest as readiness proof; documentation must keep the
  no-live-authority boundary explicit.

## Test plan

- Focused handoff/no-WING/rails tests.
- Project Director and direction guard checks.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer no hash because object comparison already exists. I would keep
the digest because operator summaries and release artifacts benefit from a
compact, deterministic fingerprint.
