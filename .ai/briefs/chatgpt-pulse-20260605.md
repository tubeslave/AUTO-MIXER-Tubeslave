# Brief: chatgpt-pulse-20260605

## Short thesis

Accept one small offline evaluation improvement from the 2026-06-05 ChatGPT
Pulse snapshot: let the existing ABX/loudness gate consume safety audit evidence
from adjacent tools instead of duplicating AutoEQ or dynamics validators.

## What I understood

The Pulse snapshot repeated several useful audio-evaluation ideas:

- Diff-DEQ / NDMP model output should pass through a canonical JSON shape before
  any DSP preset or ABX claim.
- CamillaDSP / ABX pipeline checks should include loudness, peaks, ABX
  significance, and parameter safety.
- Adaptive SLO / Rego / CoreML / Demucs / driver-watch ideas may matter later,
  but they either need separate architecture, hardware, external tooling, or
  bigger dependency choices.

Current project state already has:

- `tools/offline_abx_gate.py` for offline ABX/loudness preference evidence.
- `backend/autoeq_audit.py` for AutoEQ PEQ/filter safety checks.
- Proposal-only/fail-closed/read-only work in progress for many 20260605
  surfaces.

## Proposed solution

Extend the existing offline ABX gate with a small generic safety-audit bridge:

- ABX artifacts may include `safety_audits`, `audit_reports`, or `audits`.
- Each audit row must expose a pass/fail status such as `passed`,
  `overall_pass`, or `technical_pass`.
- If an attached audit fails or lacks a trustworthy status, the ABX artifact
  fails the technical gate.
- A CLI flag can require at least one safety audit for stricter CI runs.

This keeps AutoEQ-specific clamp logic in `backend/autoeq_audit.py` and avoids a
new live apply path.

## Likely files to touch

- `tools/offline_abx_gate.py`
- `tests/test_offline_abx_gate.py`
- `.paperclip/reports/chatgpt_pulse_review_*.md`

## Alternatives considered

- Duplicate PEQ frequency/gain/Q clamps inside the ABX gate. Rejected because
  AutoEQ audit already owns that logic.
- Add CamillaDSP / Rego / CoreML / Demucs tooling. Deferred because that is
  larger, dependency-heavy, or hardware/lab dependent.
- Update the canonical roadmap. Rejected because this is an incremental offline
  evaluation bridge within existing scope.

## Risks

- A generic audit bridge only trusts attached audit output; it does not prove
  the audit was run correctly.
- Existing dirty worktree remains a review risk, so the slice must stay narrow.
- Offline ABX evidence still does not grant live WING readiness.

## Test plan

- `python3 -m py_compile tools/offline_abx_gate.py tests/test_offline_abx_gate.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_offline_abx_gate.py -q`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue that parameter clamps should be evaluated directly inside the ABX
artifact. My recommendation is to keep domain-specific validators separate and
make the ABX gate require their pass/fail evidence when stricter CI needs it.
