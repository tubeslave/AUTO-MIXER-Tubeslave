# Current Tasks

## Purpose

This file is a lightweight task memory for agents. It does not replace
Paperclip, ADRs, tests, or git history.

## Active Tasks

- M110 attended WING reset and staged soundcheck is the current
  operator-directed task. CP05 and clean detached launch-only are verified.
  The real WING and 48x48/48 kHz CoreAudio stream are now reached, but HW01 is
  blocked before route-artifact issuance on `master_channel_signal_missing`.
  CP05 is dynamically checked against the current non-marker tree; repeat HW01
  in one proof-owned attended session only after a confirmed non-silent Main
  47/48 source, then continue HW02-HW05 in order.
  Do not claim 95/100 before the repeated final audit and SHA-bound listening
  prove `show_mix_ready`; do not convert the one-run M006 permission into
  standing authority.

- Use `Docs/MVP_WING_SUPERVISED.md` and
  `.ai/project-layer/wing_mvp_blockers.md` as the current execution rail.
- Use `.ai/project-layer/autonomous_mvp_and_model_routing.md` and
  `.ai/agent-core/model_budget_policy.md` for Goal-mode autonomy, virtual
  subagent routing, and model-budget protection.
- Consult `tools/codex_autoloop/project_director_agent.py` before selecting
  autonomous work, but take the active task from `ops/autopilot/STATE.json`,
  `Docs/ACTIVE_MILESTONE.md` and the current external handoff. The first
  non-closed row in the frozen legacy WING MVP table is dependency evidence,
  not a task selector and not a reason to loop on one historical blocker.
- Keep `Docs/CORRECTION_METHODS_ROADMAP.md` and
  `.ai/project-layer/correction_methods_backlog.md` visible for Gain/Trim, Pan,
  EQ, Comp, FX, Balance, routing, scene, and output methods.
- Maintain the memory pipeline and keep safety policy explicit.
- Continue improving `input_trim_preamp_character` as the canonical input-trim
  method when the task is specifically about gain staging.
- Keep Paperclip/Codex automation in dry-run or audit-only mode unless the
  operator explicitly approves a supervised action.
- Preserve Council artifacts in `.ai/briefs/`, `.ai/proposals/`, `.ai/reviews/`,
  and `Docs/adr/`.

## Watch List

- Do not let report automation become a WING/OSC write path.
- Do not let Project Director Agent become a code-writing or live-WING actor;
  it is an audit/planning agent only.
- Do not spend GPT-5.5 high reasoning on routine cycles; use the local budget
  gate and keep high reasoning for rare architecture/safety review windows.
- Do not treat shadow-only or offline-only evidence as live readiness.
- Do not widen correction-method live writes before the supervised WING MVP
  hardware proof and a separate ADR.
- Keep memory append-only unless a human explicitly asks for cleanup.
- If full tests fail for old reasons, run the focused test and record the wider
  failure honestly.

## Update Rule

Append task updates with date, owner, status, evidence, and next action. Do not
delete old tasks unless a human explicitly requests pruning.

## 2026-07-14 — M110 pre-hardware source checkpoint

- Owner: Codex + attended operator.
- Status: CP00 complete; CP01-CP04 implemented/offline-validated; CP05 and
  HW01-HW05 pending.
- Evidence: exact 64-address processor bypass; fixed M006 restore stages
  23/14/14 with 77 unique changed addresses; durable parent/child journals;
  split/replan lanes; two-pass HW05 audit; session-latched readiness
  invalidation; no WING/USB contact or live write in this cycle.
- Next action: finish full source tests and independent review, commit source,
  bind/commit CP05 after lease release, then run the real attended workflow and
  record actual readiness without widening the normal allowlist.

## 2026-07-14 — M111 migration route; M110 WING USB prerequisite missing

- Owner: Codex + attended operator.
- Status: CP00-CP05 complete; clean detached launch-only verified; HW01 blocked
  before start because exact CoreAudio WING candidate count is `0`; HW02-HW05
  pending.
- Evidence: commit `5c054eb9e28d29601d974850b8874858b3fe0881`; launch summary
  `/Users/dmitrijvolkov/.automixer/runtime/m110_attended_final_20260714T173441Z/plan-summary.json`;
  local CoreAudio/macOS USB enumeration classified as
  `ephemeral_local_enumeration`; no audio stream, WING, OSC/UDP or live write.
- M111 status: offline fail-closed migration route implemented; activation
  requires a fresh source/run challenge, exact same-console serial,
  clean committed replay and explicit migration-bound M006 decision.
- M112 validation-source baseline:
  `451d0fdf398fe7f1f4ffd1c37f2e6f2d370506df` is committed and verified with
  `7160 passed, 3 skipped`, 150 rails and no-WING 66/66.
- Next action: require physical WING USB 48x48/48 kHz before starting the
  attended HW01-HW05 workflow. No control-plane sync grants live authority.

## 2026-07-15 — M116 real HW01 attempt blocked on digital silence

- Owner: Codex + attended operator.
- Status: CP00-CP05 complete; HW01 attempted but incomplete on
  `master_channel_signal_missing`; HW02-HW05 pending.
- Evidence: WING Rack serial `0100Q1V0604AAE`, firmware
  `3.0.5-4-ga4ea833d:release`; CoreAudio WING index 1 at 48x48/48 kHz; local
  runner summary SHA-256
  `3a300561f3a6b31099cf2586f0047d674ff1342abb767f761b856f384ef91273`;
  operator-observed short and 120-second diagnostics showed all 48 channels at
  digital zero; those extra observations are not independently
  machine-verified. The diagnostic is not accepted HW01 evidence and
  serial-bound USB identity was not reached.
- Safety: mutation attempts 0, writes 0, reset false, corrections 0, authority
  false, strict score 88/100 and allowlist 0.
- Next action: physically contain outputs, start and keep a confirmed non-silent
  Ableton/WING source running, then repeat HW01 from a clean CP05-bound checkout.
