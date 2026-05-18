# AUTO-MIXER Project Implementation Plan

Date: 2026-05-18
Audience: Codex, Kimi, Paperclip, GPT Director, engineering agents, project owner
Status: canonical shared plan

## Agent Visibility Contract

This document is the canonical implementation plan for AUTO-MIXER-Tubeslave.
Agents must read it before creating project tasks, changing scope, claiming live
readiness, or proposing work that touches WING/OSC/runtime behavior.

Companion Paperclip-readable pointer:

`/.paperclip/reports/automixer_project_implementation_plan_20260518.md`

Current health-baseline pointer:

`/Docs/HEALTH_BASELINE_20260518.md`

Hard boundary: this plan does not grant live write approval. WING/OSC writes
remain blocked unless an operator explicitly approves a supervised run with
readback, rollback, cooldown, and emergency-stop proof.

## Current Execution Status

Last updated: 2026-05-18

- Phase 0 local health baseline is complete. Backend tests and frontend build
  are green after cleanup.
- Frontend dependency audit is red and must be handled as a separate
  modernization task, not with blind `npm audit fix --force`.
- Product Layer proposal workflow is implemented and covered by focused tests
  plus server integration tests.
- Generated dependency/build folders are ignored and can be recreated locally.
- Branch synchronization is still unresolved: local `main` was
  `ahead 66, behind 26` before the health-baseline report.
- Live-console readiness is unchanged. No physical WING proof was performed as
  part of the cleanup/baseline work.

## 1. Goals And Objectives

AUTO-MIXER-Tubeslave solves the problem of slow, manual, and risky soundcheck
and live-mix preparation on Behringer WING Rack. The project must give the
operator a safe assistant that observes the console, analyzes audio/mix state,
proposes actions, and applies only explicitly approved changes through a
supervised gate.

Business outcomes:

- Reduce soundcheck and preparation time by 30-50%.
- Reduce live-operation risk by enforcing dry-run, approval, cooldown, readback,
  rollback, and emergency-stop behavior.
- Deliver a demonstrable MVP: backend, frontend, WING OSC, replay/test harness,
  and operator UI.
- Create a reliable base for packaging, Paperclip/GPT coordination, and later
  commercial productization.

MVP success criteria:

- The system can run in read-only observation mode without touching the console.
- The system can generate operator proposals from analysis/replay state.
- The UI can show safety state, proposals, readiness, and blocked conditions.
- A limited real-WING proof can apply one approved change, read it back, roll it
  back, respect cooldown, and prove emergency stop.

## 2. Scope

### In Scope

- Backend coordinator and websocket API stabilization.
- WING Rack OSC integration through the existing `WingClient` path.
- Central supervised live-write gate for approved live commands.
- Read-only analysis and Product/Operator Layer modes:
  Observe, Assist, Supervised.
- Operator proposal queue:
  create, accept, dismiss, and apply through supervised gate only.
- MVP automation modules in proposal/dry-run first mode:
  Auto-Gain, Auto-EQ, Auto-Fader, Auto-Soundcheck.
- Frontend Control Center:
  system state, WING state, safety state, proposals, readiness, telemetry.
- Replay and QA:
  no-console tests, write-intent capture, regression tests, frontend build.
- Documentation:
  operator runbook, live readiness checklist, deployment notes.

### Out Of Scope For MVP

- Fully autonomous live mixing without the operator.
- Live EQ, compressor, routing, snapshot, scene, mute, or FX writes unless a
  separate approved safety gate and test path exists.
- Training new ML models from scratch.
- Broad mixer support beyond WING Rack as the primary target.
- Cloud SaaS, billing, multi-user product auth, or remote fleet management.
- Rewriting legacy pipelines when a narrow adapter or status-only surface is
  enough.
- Paperclip dispatch, GPT bridge mutation, or live runtime activation unless
  separately approved.

## 3. Work Breakdown Structure

### 0. Baseline And Governance

0.1 Record current branch, dirty diff, and divergence from upstream.

0.2 Run current health checks:

- backend tests;
- frontend build;
- websocket smoke check where available;
- conflict-marker and secret hygiene checks.

0.3 Produce risk register:

- direct live-write bypasses;
- failing tests;
- readback/rollback gaps;
- frontend status-only tabs;
- packaging/runtime risks;
- Paperclip/GPT visibility risks.

Deliverable: baseline report and explicit go/no-go list for implementation.

### 1. Safety And Live-Write Hardening

1.1 Inventory all WING/OSC write paths.

1.2 Ensure all state-changing writes route through supervised gate or fail
closed.

1.3 Fix wrappers that hide blocked write results from callers.

1.4 Require structured result propagation:

- requested value;
- old value;
- applied value;
- readback value;
- rollback status;
- blocked reason;
- telemetry artifact path.

1.5 Validate negative controls:

- unapproved write is blocked;
- disarmed gate is blocked;
- unsupported write kind is blocked;
- cooldown is enforced;
- emergency stop blocks further writes.

Deliverable: supervised write gate is the only accepted live-write path for MVP.

### 2. Backend Product Layer

2.1 Stabilize `backend/server.py` as the primary websocket coordinator.

2.2 Keep handler status explicit:

- available;
- unavailable;
- read-only;
- status-only;
- supervised.

2.3 Complete Operator Product State:

- current mode;
- safety capabilities;
- WING connection state;
- latest analysis;
- proposal counts;
- readiness blockers.

2.4 Complete proposal queue:

- create proposal;
- import analysis recommendations;
- accept;
- dismiss;
- apply through supervised gate;
- persist event evidence.

2.5 Connect Auto-Gain, Auto-EQ, Auto-Fader, and Soundcheck outputs as proposals
first, not autonomous live actions.

Deliverable: backend can produce and manage operator decisions end to end.

### 3. Frontend Operator Experience

3.1 Build Control Center as the first working surface, not a marketing screen.

3.2 Show operator state:

- Observe / Assist / Supervised mode;
- safety gate state;
- WING connection;
- telemetry availability;
- readiness blockers.

3.3 Build proposal workflow:

- pending suggestions;
- action type and target;
- confidence/risk;
- accept/dismiss;
- apply only when supervised mode allows it.

3.4 Keep unsupported modules fail-closed:

- no fake live buttons;
- no dead websocket commands;
- disabled/status-only tabs must explain state through status fields, not
  pretend control exists.

3.5 Verify responsive layout for desktop and practical laptop widths.

Deliverable: operator can understand what the system sees, what it recommends,
and whether it is allowed to touch the console.

### 4. Replay, QA, And Evidence

4.1 Expand replay tests for analysis and proposal generation.

4.2 Add write-intent tests for all MVP automation modules.

4.3 Run regression checks:

- backend focused tests;
- backend suite where practical;
- frontend build;
- import checks for core runtime modules.

4.4 Produce safety evidence artifacts:

- blocked write proof;
- dry-run proposal proof;
- rollback/readback proof where hardware is involved;
- readiness report.

Deliverable: agents can prove current readiness from files and tests, not from
claims in chat.

### 5. Limited WING Pilot

5.1 Preconditions:

- operator present;
- correct WING Rack IP confirmed;
- no autonomous write-capable flows running;
- supervised gate disarmed at start;
- telemetry/artifact path selected;
- rollback target known.

5.2 Pilot flow:

- connect to WING;
- verify read-only state;
- arm supervised gate;
- send one negative control;
- apply one approved fader or gain change on one channel;
- verify readback;
- wait cooldown;
- rollback to original value through the same supervised path;
- verify rollback readback;
- trigger emergency stop;
- prove next write is blocked.

5.3 Preserve all artifacts even if the run fails.

Deliverable: limited physical-console proof. This proves supervised readiness,
not autonomous live readiness.

### 6. Packaging, Documentation, And Handoff

6.1 Update operator runbook.

6.2 Update developer setup and known commands.

6.3 Update live readiness checklist.

6.4 Publish MVP status report:

- ready;
- supervised only;
- blocked;
- out of scope;
- next task.

6.5 Add the final report to Paperclip/GPT-readable surfaces.

Deliverable: release-candidate handoff package.

## 4. Calendar Schedule

Project start: 2026-05-18

| Phase | Dates | Milestone |
| --- | --- | --- |
| 0. Baseline audit | 2026-05-18 to 2026-05-20 | M0: current status, tests, risks recorded |
| 1. Safety gate hardening | 2026-05-21 to 2026-05-29 | M1: live writes fail closed unless supervised |
| 2. Backend Product Layer | 2026-06-01 to 2026-06-12 | M2: proposals and operator modes work end to end |
| 3. Frontend Control Center | 2026-06-08 to 2026-06-19 | M3: operator sees status, risks, and decisions |
| 4. Replay and QA | 2026-06-15 to 2026-06-26 | M4: replay and regression evidence exists |
| 5. Limited WING pilot | 2026-06-29 to 2026-07-03 | M5: one supervised hardware proof completed |
| 6. Packaging and docs | 2026-07-06 to 2026-07-10 | M6: MVP release-candidate handoff |

Milestone gate rule:

- M0-M4 may be completed without live WING writes.
- M5 requires explicit operator approval for the hardware test.
- No milestone may claim fully autonomous live readiness.

## 5. Resources And Team

### Role Matrix

| Role | Owner | Responsibility |
| --- | --- | --- |
| Product Owner / Audio Owner | Dmitrij Volkov | Product direction, audio acceptance, live-console approval |
| Technical Lead | Backend engineer / Codex | Architecture, scope control, safety boundaries |
| Backend Engineer | Backend engineer / Codex | Server, handlers, WING client, proposal API |
| Frontend Engineer | Frontend engineer / Codex | Control Center, operator workflow, UI states |
| Safety / Live QA | Dmitrij Volkov + QA engineer | WING pilot, rollback proof, emergency-stop proof |
| Audio/DSP Expert | Dmitrij Volkov / FOH consultant | Mix logic, module validation, audio criteria |
| Release Engineer | Engineer / Codex | Tests, packaging, docs, release report |
| Paperclip/GPT Coordinator | Paperclip Director / Codex | Agent handoff, issue hygiene, report visibility |

### RACI

| Workstream | Product Owner | Tech Lead | Backend | Frontend | Safety QA | DSP Expert | Release | Paperclip/GPT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scope and priorities | A | R | C | C | C | C | I | I |
| Safety gate | A | A | R | I | R | C | C | I |
| Backend Product Layer | C | A | R | C | C | C | I | I |
| Frontend Control Center | C | A | C | R | C | C | I | I |
| Replay tests | I | A | R | C | C | C | R | I |
| Limited WING pilot | A | C | C | I | R | C | I | I |
| Packaging and docs | C | A | C | C | C | I | R | C |
| Agent handoff/reporting | C | A | C | C | I | I | C | R |

Legend:

- R = Responsible
- A = Accountable
- C = Consulted
- I = Informed

## 6. Agent Operating Rules

All agents must follow these rules when using this plan:

- Treat the plan as shared context, not as approval to mutate live systems.
- Preserve the separation between live runtime, replay/evaluation, Product
  Layer, offline DSP, and Paperclip/GPT coordination.
- Default to read-only analysis unless the task explicitly asks for code edits.
- Default to dry-run/proposal behavior unless the operator explicitly approves a
  supervised live run.
- Report readiness in three separate states:
  dry-run ready, supervised ready, fully live ready.
- Do not call partial physical-console proof fully live-ready.
- Keep old pipelines and default behavior intact unless a change is explicitly
  scoped and tested.
