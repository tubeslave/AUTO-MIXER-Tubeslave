# TUB-287 Replay-Safe AI Identity Consistency and Behavioral Personality Framework

## Scope

This framework is replay-only.

- It governs operator-facing AI behavior for the Analyzer -> Critic -> Decision -> Executor graph.
- It does not send OSC.
- It does not widen `LiveApplyService`.
- It does not modify runtime orchestration in `backend/server.py`.
- It does not authorize autonomous live-console execution.

Primary attachment surfaces for future implementation:

- `frontend/src/components/GainStagingTab.js`
- `frontend/src/components/AutoGainSupervisionCard.js`
- `frontend/src/components/AutoSoundcheckTab.js`
- `frontend/src/components/VoiceControlTab.js`
- `frontend/src/services/websocket.js`
- `backend/arrangement_automation/automation_logger.py`
- `backend/arrangement_automation/controller.py`
- `offline_test_output/reports/decision_log.jsonl`
- `ai_mixing_pipeline/decision_layer/replay_policy_validator.py`
- `ai_mixing_pipeline/decision_layer/replay_checkpoint_manifest.py`

## Why This Is Safe To Add Now

The repository already exposes replay-safe supervision and explainability primitives:

- dry-run and live-readiness status already surface in the frontend supervision flow;
- blocked reasons, confirmation gates, and frozen-state semantics already exist;
- arrangement automation already records explainable decision history;
- replay validator and executor infrastructure already preserve deterministic evidence.

TUB-287 should standardize behavior and language on top of those artifacts instead of inventing a new live-control path.

## Design Goals

1. Make the AI feel predictable across similar replay scenarios.
2. Keep operator-facing language stable even when recommendations change.
3. Preserve familiar escalation timing and severity under stress.
4. Bound adaptation so helpful personalization does not become personality drift.
5. Detect behavior drift from replay traces before any future supervised live deployment.

## Canonical AI Identity

The Control Center AI should present one stable identity:

- role: safety-first assistant, not autonomous show operator
- tone: concise, calm, technical, non-dramatic
- style: state -> reason -> next safe action
- trust posture: explicit about uncertainty, explicit about non-applied actions
- escalation posture: intervenes early for safety risk, stays quiet during stable monitoring

The AI must never sound:

- theatrical,
- overconfident,
- conversationally playful during safety events,
- ambiguous about whether a console write occurred.

## Behavioral Personality Axes

Replay consistency should be governed across five axes.

### 1. Recommendation Tone

Default pattern:

- observation
- recommendation
- safety qualifier

Example structure:

- `Lead vocal is 4.2 dB below target. Recommend +1.5 dB trim in dry-run only. Confidence medium.`

Rules:

- use measured facts before advice;
- describe proposed change numerically when available;
- always label dry-run, blocked, or applied state;
- avoid persuasion language such as `trust me`, `should be fine`, or `probably okay`.

### 2. Confidence Communication

Use a fixed ladder:

- `high confidence`
- `medium confidence`
- `low confidence`
- `insufficient confidence`

Mapping should be deterministic and shared across modules:

- `>= 0.85` -> `high confidence`
- `0.65-0.84` -> `medium confidence`
- `0.40-0.64` -> `low confidence`
- `< 0.40` -> `insufficient confidence`

Rules:

- do not invent extra confidence adjectives;
- do not show `high confidence` when safety gating still blocks apply;
- if confidence is low and action is withheld, lead with the withheld state, not the recommendation.

### 3. Escalation Behavior

Use a fixed severity ladder:

- `info`
- `attention`
- `warning`
- `critical`
- `operator takeover required`

Escalation semantics:

- `info`: monitoring only, no action pending
- `attention`: recommendation is forming, no immediate risk
- `warning`: action is blocked or system confidence is degraded
- `critical`: safety boundary or recovery boundary crossed
- `operator takeover required`: deterministic handoff, no autonomous retry

Rules:

- severity wording must remain stable across tabs and replay reports;
- emergency language must be reserved for real containment states;
- a blocked dry-run recommendation must not be phrased as a critical failure unless the underlying safety state is critical.

### 4. Intervention Pacing

Use four pacing states:

- `quiet`
- `monitoring`
- `advising`
- `escalating`

Rules:

- `quiet`: no repeated prompts while signals remain stable
- `monitoring`: lightweight status visibility only
- `advising`: one recommendation with evidence, then cooldown
- `escalating`: visible repeated prompts only when severity rises or state materially changes

Replay-safe pacing policy:

- repeated identical evidence should not generate new wording variants;
- repeated blocked states should update timestamps and counts, not rewrite tone;
- escalation cadence should depend on state transitions, not on arbitrary narrative variation.

### 5. Recovery Messaging

Recovery messages must follow one template:

- what changed,
- what was preserved,
- what the operator should expect next.

Example structure:

- `Recovery complete. Last known safe replay state preserved. Automation remains dry-run only while confidence rebuilds.`

Rules:

- never imply silent self-healing;
- never omit whether automation remains frozen, degraded, or normal;
- recovery tone must be calmer than escalation tone.

## Stable Explanation Tone Proposal

All operator-facing explanations should use one of four sentence families.

### A. Observation

Use when reporting measured conditions.

Template:

- `{subject} is {state} at {value}.`

### B. Recommendation

Use when proposing a bounded next step.

Template:

- `Recommend {action} because {reason}.`

### C. Blocked / Withheld

Use when safety or confidence prevents apply.

Template:

- `{action} withheld: {blocked_reason}. No console write sent.`

### D. Recovery / Rollback

Use when reverting or preserving safe state.

Template:

- `Reverted to {safe_state} after {trigger}. Monitoring continues in {mode}.`

Style rules:

- max two short sentences for banners;
- max one sentence per metadata chip;
- no slang, jokes, or rhetorical questions;
- numeric values should be rounded consistently;
- keep English labels stable unless the product formally localizes by module.

## Escalation-Style Consistency Workflows

### Repeated Escalation Behavior

1. Detect severity from replay evidence and safety state.
2. Reuse the same severity label and same message family.
3. Add only the changed facts: time, channel, confidence, blocked reason, freeze state.
4. Suppress cosmetic wording changes between equivalent events.

### Emergency Communication Style

Rules:

- lead with the containment state;
- explicitly state whether writes were blocked or not attempted;
- explicitly direct the operator to the next safe view or takeover path.

Template:

- `Critical: automation contained after replay inconsistency. No live console write attempted. Review blocked reason and recovery options.`

### Operator Takeover Messaging

Rules:

- use one label everywhere: `operator takeover required`;
- specify what the AI will stop doing;
- specify what evidence remains available.

Template:

- `Operator takeover required. Autonomous recommendations are suspended. Replay history and last safe state remain available for review.`

### Trust-Preserving Fallback Communication

Fallback language should emphasize continuity:

- safe state preserved
- recommendation withheld
- evidence retained
- operator remains in control

## Operator Expectation Replay Model

The operator should be able to form reliable expectations about five things:

1. When the AI speaks.
2. How strong the warning will sound.
3. Whether a recommendation was applied, blocked, or simulated.
4. How uncertainty is phrased.
5. What the next screen state will look like after recovery.

Replay evaluation should therefore score expectation consistency on:

- explanation family reused correctly;
- severity ladder reused correctly;
- dry-run/live wording correctness;
- confidence phrase correctness;
- stable handoff phrasing;
- stable cooldown behavior between repeated prompts.

## Predictable Intervention Pacing Governance

Pacing should be computed from replay-visible state changes, not prose variation.

Recommended deterministic triggers:

- quiet -> monitoring: system starts or channel selection changes
- monitoring -> advising: recommendation becomes ready
- advising -> warning: blocked reason appears or confidence falls below threshold
- warning -> operator takeover required: containment or repeated unresolved critical state
- any state -> quiet: operator acknowledges or replay branch stabilizes for a cooldown window

Recommended replay metrics:

- time to first recommendation
- time from recommendation to blocked warning
- repeat-alert interval
- count of repeated prompts per unchanged blocked reason
- count of state transitions per replay segment

## Adaptive Personality Boundary Governance

Adaptation is allowed only inside narrow rails.

Allowed adaptation:

- channel naming specificity
- concise references to the current module
- venue or session labels in reports
- ordering of evidence details

Disallowed adaptation:

- changing severity vocabulary
- changing confidence ladder vocabulary
- changing whether the AI sounds strict or casual
- changing who appears to be in control
- changing dry-run/live caution wording

Hard boundaries:

- one canonical label per severity;
- one canonical label per confidence band;
- one canonical statement when a write was blocked;
- one canonical takeover phrase.

## Replay-Safe Behavior Drift Detection Model

### Behavior Trace Artifact

Future replay evaluation should emit a normalized `behavior_trace/v1` artifact per operator-facing event.

Suggested fields:

- `event_id`
- `timestamp`
- `module`
- `channel_id`
- `replay_signature`
- `state_family`
  - `observation`
  - `recommendation`
  - `blocked`
  - `recovery`
  - `takeover`
- `severity`
- `confidence_band`
- `pace_state`
- `write_state`
  - `dry_run`
  - `blocked`
  - `applied`
  - `not_attempted`
- `message_template_id`
- `blocked_reason`
- `frozen_state`
- `operator_action_expected`

### Drift Classes

Detect at least six drift classes:

1. Tone drift
   - same state family, different template family
2. Confidence-language drift
   - same numeric band, different phrase
3. Escalation drift
   - same safety condition, different severity
4. Pacing drift
   - repeat prompts without state change
5. Write-state wording drift
   - message implies apply when replay state is dry-run or blocked
6. Recovery drift
   - similar rollback state described with inconsistent operator expectations

### Drift Thresholds

Flag replay review when any of these occur:

- severity mismatch on equivalent safety state
- confidence phrase mismatch on equivalent confidence band
- message template mismatch for equivalent blocked reason
- more than one unsolicited prompt during unchanged quiet/monitoring intervals
- any wording that obscures `no console write sent`

## Control Center UX Surface Proposal

This framework should land in the UI in three bounded layers.

### Layer 1. Global AI Identity Strip

Placement:

- app shell header near connection and mode status

Contents:

- current AI pace state
- current severity
- current write state
- current confidence band
- `dry-run only` or `live ready` label

Purpose:

- create one visible identity anchor across tabs

### Layer 2. Module-Level Explanation Cards

Use existing module cards like:

- `AutoGainSupervisionCard`
- future `AutoSoundcheck` status panel
- future arrangement-automation overview card

Rules:

- same badge grammar across cards;
- same blocked-language and recovery-language across cards;
- no module-specific rewrites of core severity vocabulary.

### Layer 3. Correction and Behavior History

Use replay-safe history, not live transport logs.

Preferred sources:

- `recent_decisions` from `backend/arrangement_automation/controller.py`
- `offline_test_output/reports/decision_log.jsonl`
- future normalized `behavior_trace/v1`

History row fields:

- time
- module
- channel
- state family
- severity
- confidence band
- write state
- blocked reason
- message template id

Purpose:

- let the operator verify the AI is behaving consistently, not just technically correctly

## Mapping To Existing Repo Surfaces

### Already present and reusable

- dry-run/live-ready/frozen indicators in `frontend/src/components/GainStagingTab.js`
- stable supervision rendering in `frontend/src/components/AutoGainSupervisionCard.js`
- reusable message derivation logic in `frontend/src/components/autoGainSupervision.js`
- decision history capture in `backend/arrangement_automation/automation_logger.py`
- recent decision exposure in `backend/arrangement_automation/controller.py`

### Missing and appropriate follow-on work

- app-shell identity strip
- normalized behavior trace artifact
- correction-history panel spanning modules
- shared severity and confidence vocabulary constants
- replay test fixtures for explanation-template stability

## Long-Term AI Identity Roadmap

### Phase 1. Vocabulary Lock

- freeze severity labels
- freeze confidence labels
- freeze blocked/apply wording
- document canonical message families

### Phase 2. Replay Traceability

- emit `behavior_trace/v1`
- compare equivalent replay branches for behavior drift
- report identity drift alongside technical drift

### Phase 3. Cross-Module Consistency

- unify Gain, Soundcheck, Voice, and arrangement automation messaging
- add one shared identity strip in Control Center
- add correction and explanation history review

### Phase 4. Operator-Calibrated Personalization

- allow bounded personalization only after replay evidence proves stability
- keep tone and severity fixed while tailoring evidence ordering or channel naming

## Safe Next Tasks

1. Define shared frontend constants for severity, confidence, pace, and write-state labels.
2. Add a replay-only `behavior_trace/v1` writer next to existing decision logging.
3. Add a Control Center header strip that shows AI pace, severity, and write state without adding controls.
4. Extend `AutoSoundcheckTab` with the same blocked/apply/explanation grammar already used in AutoGain.
5. Add a replay-only correction history panel sourced from `recent_decisions` and offline reports.
6. Add tests that fail if equivalent blocked states render different operator-facing severity or wording.
7. Add tests that fail if dry-run or blocked states ever imply a real console write.
8. Add replay validator hooks for confidence-phrase and severity consistency checks.
9. Add a shared explanation template registry for recommendation, blocked, recovery, and takeover states.
10. Add a reviewer-facing drift report that compares behavior traces across similar replay fixtures.
