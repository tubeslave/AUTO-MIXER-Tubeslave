# Current Tasks

## Purpose

This file is a lightweight task memory for agents. It does not replace
Paperclip, ADRs, tests, or git history.

## Active Tasks

- M110 attended WING reset and staged soundcheck is the current
  operator-directed task. Finish CP05 from committed source, remove the empty
  writer-lease directory, then execute HW01-HW05 in one proof-owned session.
  Do not claim 95/100 before the repeated final audit and SHA-bound listening
  prove `show_mix_ready`; do not convert the one-run M006 permission into
  standing authority.

- Use `Docs/MVP_WING_SUPERVISED.md` and
  `.ai/project-layer/wing_mvp_blockers.md` as the current execution rail.
- Use `.ai/project-layer/autonomous_mvp_and_model_routing.md` and
  `.ai/agent-core/model_budget_policy.md` for Goal-mode autonomy, virtual
  subagent routing, and model-budget protection.
- Follow `tools/codex_autoloop/project_director_agent.py` before selecting
  Codex work; the selected task must come from the first non-closed WING MVP
  blocker.
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
