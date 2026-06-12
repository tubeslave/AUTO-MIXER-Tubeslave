# Automixer WING Read-Only Handoff MVP Blocker Anchor Digest

Date: 2026-06-12

## Summary

WING read-only handoff packets now include `mvp_blocker_anchor_sha256`, a
canonical JSON SHA256 digest of the current first-open MVP blocker anchor. The
no-WING validator fails closed if the digest is missing or does not match the
packet anchor and current first-open blocker row.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session opened.
- No OSC/WING writes.
- No listener was stopped or reconfigured.
- No live authority granted.
- No MVP blocker was closed from documentation or offline evidence.

## Change

- Added `wing_mvp_blocker_anchor_sha256()` to no-WING tooling.
- Added `mvp_blocker_anchor_sha256` to WING read-only handoff packets.
- Rendered the digest in handoff Markdown and no-WING summary Markdown.
- Added no-WING schema violations for missing or mismatched anchor digest.
- Added regression coverage for digest mismatch.
- Updated the read-only runbook to require
  `mvp_blocker_anchor_sha256_matches_current_first_open: true`.

## Why It Matters

The handoff anchor proves which blocker row the offline packet was generated
for. The digest gives reviewers and release summaries a compact fingerprint of
that exact row context while preserving the no-live-authority boundary.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py`
  - `220 passed`
- `python3 tools/build_wing_readonly_preflight_handoff_packet.py --format json --output /tmp/wing_handoff_anchor_digest.json --quiet`
  - `selected_blocker: WING read-only connection proof`
  - `mvp_blocker_anchor_sha256: 2ee68397df8475ca179f6801e1737a06220ade3462fc9cc3efecb3fc33d7eb2a`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - `selected_blocker: WING read-only connection proof`
  - `guard_status: on_target`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3204 passed, 1 skipped, 1 warning`
