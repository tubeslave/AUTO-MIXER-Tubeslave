# Codex Proposal — M116 M110 HW01 No-Signal Truth Sync

## 1. Short thesis

Record HW01 as `attempted_blocked_master_signal_missing` and preserve the
diagnostic/accepted-proof distinction; change no product or live-write code.

## 2. What I understood

The WING/CoreAudio prerequisite is now physically satisfied. The route attempt
entered the proof-owned backend and capture path but could not issue an HW01
artifact because every captured channel was silent. Existing truth surfaces
still say no device existed and no stream or OSC path opened.

## 3. Proposed solution

Add one structured `m116_hw01_no_signal_attempt` object to `STATE.json`, update
the active-work prerequisite/contact/checkpoint fields, and synchronize the
current blocker, next operator action, authoritative Markdown surfaces and one
append-only ledger row. Keep all authority flags false and make the evidence
class explicitly ephemeral and non-promoting.

## 4. Likely files to touch

- `ops/autopilot/STATE.json`, `RUN_LEDGER.jsonl`, `BLOCKERS.md`,
  `HEARTBEAT.md`, `LAST_REPORT.md`
- `Docs/CURRENT_STATE.md`, `ACTIVE_MILESTONE.md`, `ROADMAP.md`, active M001 and
  M110 plans, `Docs/reports/MVP_PROGRESS_LATEST.md`
- `.ai/project-layer/wing_mvp_blockers.md`, `.ai/project-layer/current_tasks.md`
- `tests/test_wing_mvp_rails.py` and M116 Council/ADR/plan documents

## 5. Alternatives considered

Promoting HW01, importing the local runtime summary as canonical hardware
evidence, or leaving the old missing-device state were rejected. The first two
overclaim an unissued route proof; the third leaves operator guidance false.

## 6. Risks

The largest risk is conflating physical contact with accepted evidence. A
second risk is making old append-only ledger rows appear wrong; only the new
tail row may define current truth.

## 7. Test plan

Validate JSON/JSONL, run focused truth rails, the complete WING rail file,
memory lint, full standard tests, no-WING and independent diff review. Commit
non-marker truth, then recompute and commit only the two CP05 markers.

## 8. Where another agent may disagree

An alternate reviewer may prefer a new canonical diagnostic artifact. That
would add provenance machinery without closing HW01; the local summary path and
digest are sufficient when clearly classified as non-accepted evidence.
