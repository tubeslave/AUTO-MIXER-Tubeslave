# Model Budget Policy

Status: active autonomy guard

## Purpose

Keep Codex moving toward the Automixer goal without spending the most expensive
GPT-5.5 high-reasoning quota early in the day, week, or month.

This policy is repo-local. It does not grant live mixer authority and does not
override `AGENTS.md`, `CLAUDE.md`, or the WING supervised MVP safety gates.

## Operating Rule

Use the smallest model and reasoning effort that can safely complete the current
step. Escalate only when a blocker truly needs architectural judgment, safety
review, or cross-file implementation.

## Virtual Subagents

Codex cycles may use these virtual subagent roles inside one execution context.
They are roles and routing rules, not parallel writers.

| Role | Default model profile | Reasoning | Allowed work | Forbidden work |
| --- | --- | --- | --- | --- |
| Director | budget_balanced | medium | Choose the next blocker from `.ai/project-layer/wing_mvp_blockers.md`, reject broad drift, write the cycle plan | Production edits |
| Scout | budget_light | low | Read files, collect repo evidence, summarize reports, update task facts | Architecture decisions, code edits |
| Implementer | budget_balanced | medium | Small scoped patches, tests, docs tied to the selected blocker | Live WING writes, broad rewrites |
| Safety Reviewer | budget_strict | medium | Check live-sound safety, rollback, fail-closed behavior, operator approval | Granting live readiness without real proof |
| Heavy Architect | premium_high | high | Rare monthly/weekly architecture review only when lower profiles cannot resolve the blocker | Routine daily cycles |

Only one agent workflow writes code at a time. These roles must not be turned
into multiple simultaneous Codex writers in the same worktree.

## Escalation Rules

- Start every automated cycle with `budget_light` or `budget_balanced`.
- Use `premium_high` only for explicit architecture/safety review windows and
  only when the local budget gate allows it.
- If a high-budget window is exhausted, continue with lower-cost blocker work or
  skip the cycle cleanly; do not spend high-reasoning quota just to produce a
  report.
- Prefer offline tests, static checks, and deterministic tools before asking a
  larger model to reason about the same evidence.
- Never use model escalation to bypass safety gates, hardware proof, operator
  approval, or the fader/gain-only live boundary.

## Current MVP Filter

The long-term goal remains a full AI Mixing Engineer. The active MVP remains
the supervised WING Rack proof. Automated cycles must choose work that closes or
measurably reduces one of these blockers:

1. WING read-only connection proof.
2. Physical console readback proof.
3. Supervised fader write proof with operator approval.
4. Supervised gain-down / trim proof with operator approval.
5. Rollback proof.
6. Emergency-stop proof.
7. Connection-loss / timeout fail-closed proof.
8. Operator-visible evidence report.
9. Readiness matrix update based on real proof.

## Local Budget Gate

`tools/codex_autoloop/check_model_budget.py` is the local guard for automated
cycles. It tracks reservations in `logs/codex_autoloop/model_budget_state.json`
and emits the model profile, model name, and reasoning effort for the next
cycle.

The local gate is conservative. It cannot read the provider-side account quota,
so it protects the tariff by limiting how often automated runs request expensive
profiles.
