# TUB-283 Replay-Safe Emergency Failure Containment and Safe Fallback Framework

## Scope

This framework is replay-only.

- It covers the Analyzer -> Critic -> Decision -> Executor -> Rewind intelligence graph in offline and replay-safe modes.
- It does not send OSC.
- It does not authorize live mixer writes.
- It does not modify runtime orchestration in `backend/server.py`.
- It does not broaden panic-stop semantics beyond the currently documented manual `LiveApplyService` scope.

Primary attachment surfaces for future implementation work:

- `ai_mixing_pipeline/decision_layer/replay_checkpoint_manifest.py`
- `ai_mixing_pipeline/decision_layer/replay_restore_loader.py`
- `ai_mixing_pipeline/decision_layer/replay_policy_validator.py`
- `ai_mixing_pipeline/decision_layer/replay_executor.py`
- `ai_mixing_pipeline/decision_layer/fallback_virtual_mixer.py`
- `docs/console_write_safety_policy.md`
- `backend/live_apply.py`

## Why This Is Safe To Add Now

The repository already has replay-safe primitives that can anchor containment without touching live runtime control:

- replay manifests already preserve `dry_run_only` and `live_mixer_writes`;
- replay rewind restore already rejects non-dry-run or live-write manifests;
- replay policy validation already fails closed on policy drift, signature drift, correlation mismatches, and blocked executor events without reasons;
- replay executor traces already preserve deterministic state transitions, rollback state, and blocked reasons;
- the fallback virtual mixer already provides an offline-only rendering surface;
- manual live panic-stop semantics already exist, but only for the narrow `LiveApplyService` scope documented in `docs/console_write_safety_policy.md`.

TUB-283 should build on those primitives instead of creating a parallel emergency system inside live transport code.

## Design Goals

1. Contain unstable replay behavior before it can influence future live-validation decisions.
2. Preserve deterministic evidence explaining why containment activated.
3. Fail closed on corruption, ambiguity, or confidence collapse.
4. Freeze adaptation safely without deleting replay history.
5. Restore a known-good replay checkpoint without widening live write scope.
6. Give the operator a clear takeover path that remains dry-run-only at this stage.

## Containment State Model

The framework should use five replay-governed states.

### `healthy`

- replay manifest is dry-run-only;
- correlation inventory is coherent;
- validator has no replay-safety errors;
- executor trace is deterministic and explainable.

### `watch`

- non-fatal warnings exist;
- critic disagreement is rising;
- confidence is decaying faster than expected;
- drift against a baseline branch increases but remains explainable.

### `contained`

- unsafe graph execution is frozen;
- no new adaptive promotion is allowed;
- only evidence capture, replay validation, rollback planning, and operator review continue.

### `quarantined`

- current replay branch is isolated from approved replay history;
- corrupted manifests, mismatched event sequences, or unexplained executor divergence cannot be promoted;
- quarantine is append-only and reviewable, never silently rewritten.

### `operator_recovery`

- automated replay promotion stays suspended;
- operator or reviewer selects rollback, confidence reset, branch discard, or controlled re-entry;
- no autonomous exit from this state.

## Trigger Classes

### Analyzer instability

Trigger examples:

- missing analyzer events;
- analyzer signature mismatch;
- correlation mismatch between proposals and analyzer events;
- unexpected proposal family or malformed requested state.

Containment action:

- freeze downstream promotion for the affected branch;
- mark branch `contained`;
- require validator pass before critic or decision reuse.

### Critic disagreement escalation

Trigger examples:

- critic inventory mismatch;
- critic signature mismatch;
- repeated large ranking shifts from similar proposals;
- confidence-weighted scores collapse across multiple critics.

Containment action:

- stop proposal promotion for the current branch;
- retain critic evidence but do not admit it into approved replay state;
- require operator review or baseline comparison.

### Unsafe decision amplification

Trigger examples:

- ranking-selected proposal does not match highest-ranked proposal;
- decision rank mismatch;
- repeated selection drift without corresponding evidence change;
- confidence collapse combined with large policy deltas.

Containment action:

- block decision branch promotion;
- switch to `minimal_safe_automation` fallback;
- force deterministic branch comparison against the last known-good checkpoint.

### Executor divergence

Trigger examples:

- executor selected proposal mismatch;
- executor event not dry-run;
- blocked executor event missing `blocked_reason`;
- state rollback cannot reproduce the expected restored state.

Containment action:

- mark current branch `quarantined`;
- keep the last valid rollback state immutable;
- stop executor reuse until the validator is clean.

### Replay memory corruption

Trigger examples:

- manifest, trim snapshot, and graph checkpoint event sequence mismatch;
- replay correlation inventory mismatch;
- missing checkpoint references;
- manifest reports live writes or non-dry-run state.

Containment action:

- reject restore;
- quarantine the corrupt branch and its derived artifacts;
- prevent baseline comparison from treating corrupted output as a candidate head.

### Adaptive-policy instability

Trigger examples:

- repeated drift-selected-proposal changes with decaying confidence;
- repeated blocked reasons after policy updates;
- policy evolution repeatedly pushes proposals into safety-limit blocks;
- rollback frequency increases after the same adaptive change.

Containment action:

- suspend adaptive promotion;
- reset confidence for the affected policy slice;
- require branch-level review before any further consolidation.

## Safe Fallback Modes

The framework should standardize five fallback modes.

### 1. Frozen-Safe DSP State

Purpose:

- hold the last known-good replay state without admitting new control changes.

Implementation concept:

- preserve the latest validator-clean manifest id;
- preserve the last validator-clean executor rollback state;
- preserve the last validator-clean trim snapshot and graph checkpoint.

Rules:

- no new proposal is treated as promoted;
- no fallback mode may imply live transport activation;
- fallback state must remain reconstructible from replay artifacts alone.

### 2. Operator-Only Control Mode

Purpose:

- suspend autonomous adaptation and reduce the system to operator-visible review artifacts.

Current safe meaning in this repository:

- replay reports, validator findings, branch diffs, rollback choices, and offline renders stay available;
- real console mutation remains limited to separately approved manual paths only;
- this issue does not expand `LiveApplyService` scope or panic-stop scope.

### 3. Minimal Safe Automation Mode

Purpose:

- allow only bounded, explainable, replay-safe processing while the unstable branch is isolated.

Allowed behaviors:

- manifest generation;
- validator execution;
- executor simulation;
- fallback virtual mixer renders;
- operator-assistance proposal generation;
- deterministic rollback rehearsal.

Blocked behaviors:

- adaptive promotion;
- branch auto-selection for future live use;
- any new real sender attachment.

### 4. Rollback-Safe Fallback Transition

Purpose:

- move from a suspect branch to a known-good replay checkpoint predictably.

Required sequence:

1. stop branch promotion;
2. record containment reason and affected correlation ids;
3. select last known-good manifest id;
4. restore through `load_replay_rewind_context()`;
5. rebuild replay executor state from the restored context;
6. validate the restored branch before clearing containment.

### 5. Replay-State Quarantine

Purpose:

- isolate corrupted or unstable replay branches without deleting evidence.

Quarantine rules:

- quarantined artifacts are append-only;
- quarantined artifacts cannot become the approved baseline;
- quarantine must preserve manifest id, checkpoint id, replay correlation ids, validator findings, and recovery outcome;
- quarantine release requires explicit review and a fresh validator-clean replay path.

## Unstable Adaptation Isolation Workflow

### Detection

- validator error appears;
- drift changes accelerate;
- executor blocked reasons spike;
- repeated rollback restores fail to stabilize selection.

### Isolation

- freeze adaptation for the affected branch;
- fork the branch into quarantine with immutable artifact references;
- preserve the last validator-clean baseline.

### Confidence Reset

- reset confidence only for the affected policy slice, not the full project;
- preserve pre-reset evidence;
- require the reset to be logged as an explicit containment action.

### Recovery Decision

Allowed outcomes:

- rollback to last known-good branch;
- discard the unstable branch;
- keep the branch quarantined for analysis;
- re-run replay with narrower proposal families or stricter safety limits.

## Replay-Safe Panic Recovery

Panic recovery in this issue is replay panic recovery, not a new live console stop mechanism.

Required recovery flow:

1. capture the containment trigger;
2. freeze new replay promotion;
3. persist current manifest id, checkpoint id, rollback state, and correlation inventory;
4. restore last known-good replay context;
5. re-run validator on the restored context;
6. enter operator review mode if recovery remains ambiguous.

Recovery must fail closed if:

- the manifest is not dry-run-only;
- the restore loader rejects the manifest;
- the restored branch still shows validator errors;
- the executor trace cannot explain the blocked transition.

## Emergency Operator Takeover Workflow

The operator takeover model for TUB-283 is review-first.

Required operator-visible actions:

- inspect containment trigger;
- inspect affected proposal ids and replay correlation ids;
- inspect validator findings and severity;
- inspect rollback candidate manifest ids;
- approve rollback, quarantine, discard, or controlled replay rerun.

Not allowed in this issue:

- silent autonomous resumption;
- automatic approval of a new adaptive branch;
- live console write escalation.

## Containment Audit Trail

The framework should add a new replay-governed artifact family:

`replay_emergency_containment_event/v1`

Minimum fields:

- `containment_event_id`
- `trigger_kind`
- `trigger_stage`
- `severity`
- `issue_identifier`
- `manifest_id`
- `checkpoint_id`
- `selected_proposal_id`
- `affected_proposal_ids`
- `replay_correlation_ids`
- `validator_finding_codes`
- `rollback_candidate_manifest_id`
- `fallback_mode`
- `confidence_reset_applied`
- `operator_action`
- `operator_action_at`
- `recovery_outcome`
- `dry_run_only=true`
- `live_mixer_writes=false`
- `created_at`

The audit trail must answer:

- why containment activated;
- what triggered fallback;
- which rollback path was selected;
- when operator intervention happened;
- whether recovery ended in rollback, quarantine, discard, or re-entry.

## Relationship To Existing Safety Boundaries

- `docs/console_write_safety_policy.md` remains the authority for real console writes.
- `backend/live_apply.py` remains the narrow manual real-write gate and is not broadened here.
- `ai_mixing_pipeline/decision_layer/fallback_virtual_mixer.py` remains the safe offline render surface.
- `ai_mixing_pipeline/decision_layer/replay_policy_validator.py` remains the first fail-closed containment trigger surface.
- `ai_mixing_pipeline/decision_layer/replay_restore_loader.py` remains the restore admission gate.

## Governance Roadmap

### Phase 1: Documentation and replay guard coverage

- define containment states and fallback modes;
- add replay-only tests for corruption detection and fail-closed blocked traces;
- document audit artifact requirements.

### Phase 2: Replay artifact enrichment

- add containment event artifact generation in offline replay tooling;
- add validator summaries for containment readiness;
- add branch quarantine metadata and recovery outcome recording.

### Phase 3: Operator review surface

- expose containment event summaries in offline reports;
- expose rollback candidate comparison and quarantine inventory;
- expose confidence reset decisions explicitly.

### Phase 4: Future live-policy review

- only after replay containment stays stable;
- only with separate approval;
- only through the documented console-write gate model.

## Safe Next Tasks

1. Add a replay-only containment event writer beside the existing report writers.
2. Add validator summaries that classify findings into `watch`, `contained`, and `quarantined`.
3. Add replay branch quarantine metadata to checkpoint and report outputs.
4. Add regression tests for event-sequence corruption, blocked executor events without reasons, and rollback recovery paths.
5. Add offline report rendering for rollback candidates and recovery outcomes.
6. Add confidence-reset bookkeeping for adaptive-policy branches.
7. Add issue-linked replay containment history for repeated branch failures.
8. Add branch comparison thresholds that recommend quarantine before operator review.
