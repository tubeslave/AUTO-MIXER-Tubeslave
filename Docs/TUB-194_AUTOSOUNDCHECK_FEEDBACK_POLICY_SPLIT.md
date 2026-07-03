# TUB-194 Policy Split: AutoSoundcheck Analysis vs Live-Apply vs Feedback Emergency Handling

## Purpose

This document makes the policy boundary explicit between:

- `backend/auto_soundcheck_engine.py` as an analysis-capable soundcheck path that also contains broad correction behavior; and
- `backend/feedback_detector.py` plus `backend/auto_soundcheck_engine.py::_handle_feedback_event` as a narrow emergency feedback response candidate.

It exists to support the policy work referenced by `TUB-180` and to prevent these two surfaces from being described as one shared live-apply scope.

This document does not approve any new live console mutation.

## Decision Summary

The repository currently contains three different policy states that must remain distinct:

1. `AutoSoundcheck analysis-only`
2. `AutoSoundcheck live-apply`
3. `Feedback emergency live reaction`

They are not the same risk class, not the same operator expectation, and not the same approval problem.

## Status Matrix

| Path | Current status | Approved for real live console mutation | Notes |
| --- | --- | --- | --- |
| `AutoSoundcheck` analysis-only | allowed as analysis / recommendation behavior | `no write scope involved` | May inspect audio, compute recommendations, and report intended actions while remaining dry-run |
| `AutoSoundcheck` live-apply | blocked / not approved | `no` | Broad correction surface; requires separate future gate and approval work |
| Feedback emergency reaction | emergency-exception candidate only | `no` | Separate future review item; must not inherit approval from analysis-only or from AutoSoundcheck live-apply |
| Manual `LiveApplyService` path | approved narrow gate | `yes`, but only narrow documented scope | Current approved real-write scope is manual `set_fader` / `set_gain` only |

## Split 0: Canonical Approved Gate

### What Is Currently Approved

The current approved live-write gate is:

- `backend/live_apply.py`
- manual UI `set_fader`
- manual UI `set_gain`

This gate is narrow by design and is the comparison point for the rest of this document.

### What This Means For TUB-194

- `AutoSoundcheck` does not become approved merely because it can compute safe-looking recommendations.
- feedback reaction does not become approved merely because it is safety-motivated.
- any future widening must define how it relates to `LiveApplyService` rather than bypassing that question.

## Split A: AutoSoundcheck Analysis-Only

### What It Is

- Entrypoint: `start_soundcheck.py`
- Main module: `backend/auto_soundcheck_engine.py`
- Safe scope: listen, analyze, classify, compute recommendations, and report intended corrections without mutating the console

### Current Policy Status

- Allowed to exist as analysis and recommendation behavior
- Must remain dry-run by default
- Must not be described as approved live mutation
- Must not be used to imply that the later live-apply question is already solved

### Implementation Boundary

In this state, `AutoSoundcheck` may:

- read audio and meter state
- derive EQ, compressor, pan, fader, and trim recommendations
- expose recommendation metadata and operator-visible dry-run status

In this state, `AutoSoundcheck` may not claim approval to:

- write EQ directly
- write compressor settings directly
- write fader or trim changes as approved live console operations
- bypass the current approved gate with direct client calls

## Split B: AutoSoundcheck Live-Apply

### What It Is

- Entrypoint: `start_soundcheck.py`
- Main module: `backend/auto_soundcheck_engine.py`
- Scope: channel analysis plus possible EQ, compressor, pan, fader, and gain-related correction behavior

### Why It Is A Broad Mutation Surface

`AutoSoundcheckEngine` can reach several write-capable methods directly through the mixer client, including:

- EQ writes via `self.mixer_client.set_eq_band(...)`
- compressor writes via `self.mixer_client.set_compressor(...)`
- fader and other correction application in the soundcheck path

It is therefore a general correction engine, not a narrow safety interlock.

### Current Policy Status

- Not approved for live console mutation
- Must remain dry-run by default
- Explicit `--live-apply` opt-in does not equal policy approval
- Current confirmation semantics are local to this path and do not replace a unified gate
- Readback-confirmed mutation is not established for the full correction surface
- This path is a current bypass path relative to the approved `LiveApplyService` scope

### Policy Interpretation

`AutoSoundcheck live-apply` must be treated as a product-expansion question:

- wider parameter surface
- wider blast radius
- multiple correction domains
- greater need for operator review and audit consistency

Because of that, it belongs in the same policy family as other broad automation expansion work, not in the feedback exception lane.

### Current Bypass Boundary

For this issue, the important implementation boundary is explicit:

- approved gate today: `backend/live_apply.py` for manual `set_fader` / `set_gain`
- current bypass path: `backend/auto_soundcheck_engine.py` direct mutation behavior

Future work may migrate this path, gate it, or replace it, but this document does not approve keeping the current bypass as a live product surface.

## Split C: Feedback Emergency Live Reaction

### What It Is

- Detector: `backend/feedback_detector.py`
- Current reaction hook: `backend/auto_soundcheck_engine.py::_handle_feedback_event`
- Scope: react to detected feedback by applying a narrow EQ notch or reducing a fader

### Why It Is A Different Risk Class

This path is safety-critical and time-sensitive.

Its policy question is not "should broad automation be allowed to mix live?" The real question is narrower:

- should the system ever be allowed to execute an emergency anti-feedback action before ordinary operator confirmation completes;
- if yes, what is the smallest allowed write scope;
- what hard bounds, rate limits, floors, and audit semantics are required for that exception.

### Current Policy Status

- Not approved
- Document only as an emergency exception candidate
- Must not be presented as covered by current `LiveApplyService` approval
- Must not inherit approval from AutoSoundcheck
- Requires separate review even if AutoSoundcheck remains blocked
- Remains blocked by default unless and until a separate exception policy is approved

### Candidate Future Exception Envelope

If this path is ever reviewed for approval, it should be reviewed under a narrower envelope than AutoSoundcheck:

- only emergency feedback mitigation
- only bounded notch or bounded fader reduction operations
- explicit floors and maximum repeated actions
- dedicated audit semantics
- dedicated rollback or recovery expectations where feasible
- dedicated operator-visible status distinct from ordinary live apply

## Why These Must Stay Separate

If the two paths are merged conceptually, the repo risks making unsafe reasoning sound legitimate:

- "feedback is urgent, therefore AutoSoundcheck live apply can also be urgent"
- "AutoSoundcheck already has a live-apply flag, therefore feedback actions are implicitly approved"
- "a future feedback exception means general soundcheck correction can also bypass the normal gate"

Those inferences are unsafe and should be blocked by documentation.

The same applies to analysis-only language:

- "analysis-only AutoSoundcheck is allowed, therefore live-apply is almost approved"
- "feedback is safety-related, therefore it can share the same approval story as soundcheck analysis"

Those inferences are also unsafe and should be blocked.

## Repo Evidence

### AutoSoundcheck Broad Correction Surface

- `backend/auto_soundcheck_engine.py:1077-1090` writes EQ bands directly through the mixer client
- `backend/auto_soundcheck_engine.py:1514-1524` writes compressor settings directly through the mixer client
- `backend/auto_soundcheck_engine.py:1934-1973` computes gain-trim recommendations with `auto_apply` status carried as live/dry metadata
- `backend/auto_soundcheck_engine.py:2015-2026` exposes explicit CLI flags for `--live-apply` and `--no-apply`

### Current Approved Gate

- `backend/live_apply.py` is the current approved gated write service for narrow manual scope
- `Docs/console_write_safety_policy.md` defines that approved scope as manual `set_fader` / `set_gain` only

### Current Bypass Paths Relevant To This Split

- `backend/auto_soundcheck_engine.py` broad direct-write behavior
- `backend/arrangement_automation/controller.py` arrangement automation direct-apply behavior
- `backend/auto_fader.py` direct fader automation behavior

### Feedback Emergency Reaction Surface

- `backend/feedback_detector.py:1-8` describes real-time feedback detection and suppression intent
- `backend/auto_soundcheck_engine.py:1686-1708` applies a notch EQ or reduces a fader in direct response to a feedback event

## Required Documentation Rules Going Forward

- Any doc that mentions `AutoSoundcheck` analysis-only must keep that state distinct from live-apply approval.
- Any doc that mentions `AutoSoundcheck` live apply must describe it as a broad blocked automation surface unless policy changes explicitly.
- Any doc that mentions feedback live reaction must describe it as a separate emergency exception candidate unless policy changes explicitly.
- No doc should describe "AutoSoundcheck and feedback" as one combined live-apply approval item.
- No doc should collapse analysis-only, live-apply, and emergency feedback handling into one approval state.
- No UI or API wording should imply that the `--live-apply` flag means policy approval.

## Recommended Next Tasks

1. Add tests that prove `AutoSoundcheck` analysis-only mode cannot silently escalate into live mutation.
2. Create a dedicated emergency-exception design doc for feedback only, including floors, repeat limits, and audit requirements.
3. Keep any future AutoSoundcheck approval work scoped to unified gating, operator confirmation, readback, and explicit bounded write semantics.
4. Add tests that prove feedback and AutoSoundcheck paths cannot inherit each other's approval state accidentally.
