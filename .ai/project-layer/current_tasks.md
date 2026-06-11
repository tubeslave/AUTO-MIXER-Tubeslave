# Current Tasks

## Purpose

This file is a lightweight task memory for agents. It does not replace
Paperclip, ADRs, tests, or git history.

## Active Tasks

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
