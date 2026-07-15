# Independent Proposal: M117 Operator Diagnostic Truth Contract

Codex read-only reviewer substitute; this is not a Kimi opinion. Kimi CLI did
not produce a proposal because its membership verification returned HTTP 402.

## 1. Short thesis

Create a narrow M117 truth/provenance checkpoint that corrects the
operator-facing contract without changing product code, readiness score or live
authority.

## 2. What I understood

M116 legitimately contacted the real WING, backend, OSC and CoreAudio under
operator attendance. It produced no route artifact, reset, correction or live
write. The route gate machine-recorded `master_channel_signal_missing`;
additional all-channel captures and console checks were operator-observed only.

## 3. Proposed solution

- Preserve the M116 ledger row unchanged because the ledger is append-only.
- Append one M117 offline truth-correction row that explicitly preserves the
  M116 contact facts and corrected provenance.
- Qualify `STATE.external_validation_scope` and `STATE.current_blocker`.
- Replace universal no-hardware-contact wording with an event-scoped authority
  contract.
- Synchronize all exact blocker mirrors.
- Add durable historical-M116 and prose-provenance tests.
- Keep strict readiness `88/100`, allowlist `0`, HW01 incomplete and authority
  false.
- Rebind CP05 in the required marker-only follow-up commit after the source
  commit.

## 4. Likely files to touch

- `ops/autopilot/STATE.json`
- `ops/autopilot/RUN_LEDGER.jsonl`, append only
- `Docs/CURRENT_STATE.md`
- `Docs/ACTIVE_MILESTONE.md`
- `Docs/ROADMAP.md`
- `Docs/plans/M001-wing-readonly-preflight-evidence-gate.md`
- `ops/autopilot/BLOCKERS.md`, `HEARTBEAT.md` and `LAST_REPORT.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_wing_mvp_rails.py`
- M117 council, ADR and plan records
- CP05 evidence/marker pair in the separate marker-only rebind commit

## 5. Alternatives considered

- Edit the M116 ledger row: rejected because history is append-only.
- Remove the diagnostic detail: rejected because it is useful blocker evidence.
- Treat operator observations as machine evidence: rejected.
- Leave prose stale because structured fields are correct: rejected because
  operator-facing truth is itself a safety surface.

## 6. Risks

- A non-marker change invalidates CP05 until rebinding.
- Updating only some blocker mirrors will fail the synchronization rail.
- An M117 offline latest row could hide M116-specific test coverage unless
  ledger history is searched explicitly.
- No-WING `contacts_* = false` must remain separate from M116's real contact.

## 7. Test plan

- JSON and JSONL parsing; verify the ledger gained exactly one row.
- New historical M116 ledger test.
- New provenance wording/current-surface test.
- Entire `tests/test_wing_mvp_rails.py` and the standard full suite.
- No-WING capsule/manifest verification, memory lint and `git diff --check`.
- Final dynamic CP05 verification after marker-only rebind.
- Confirm no WING/backend/OSC/audio contact during M117.

## 8. Where the other agent may disagree

A reviewer may call the stale validation strings historical and leave them
untouched. They should be corrected where they use present-tense latest-row
language and directly disagree with current ledger semantics.
