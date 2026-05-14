# TUB-280 Replay-Safe Long-Term Adaptive Memory Consolidation Framework

## Scope

This framework is replay-only.

- It consolidates long-term memory from deterministic replay artifacts.
- It does not write to a live mixer.
- It does not bypass `LiveApplyService`.
- It does not attach first to `backend/server.py` or other live transport paths.
- It does not grant autonomous policy mutation in production.

Primary attachment surfaces:

- `ai_mixing_pipeline/decision_layer/replay_checkpoint_manifest.py`
- `ai_mixing_pipeline/decision_layer/replay_executor.py`
- `ai_mixing_pipeline/decision_layer/replay_policy_validator.py`
- `ai_mixing_pipeline/offline_test_runner/runner.py`
- `backend/arrangement_automation/live_input_trim_controller.py`

## Why This Boundary Is Safe

The repo already has a replay-safe substrate:

- replay manifests preserve dry-run and live-write state;
- replay checkpoints preserve analyzer, critic, and decision history;
- replay executor traces preserve deterministic state transitions;
- replay policy validation reports preserve drift and governance findings;
- live input trim snapshots preserve replayable channel-state context without requiring transport writes.

TUB-280 should build on those artifacts instead of creating a parallel memory path inside live runtime orchestration.

## Design Goals

1. Retain useful long-term experience from replayed and offline-evaluated sessions.
2. Keep memory updates deterministic, explainable, and reversible.
3. Fail closed when evidence quality, policy compliance, or schema compatibility is weak.
4. Separate observed history from approved adaptive policy.
5. Make future cross-venue transfer opt-in and governed, never implicit.

## Long-Term Memory Domains

The framework consolidates five bounded memory domains.

1. Venue memory
   - room response tendencies
   - stable compensation patterns
   - feedback hot zones
   - timing and transition characteristics

2. Performer behavior memory
   - channel-specific gain volatility
   - mic technique consistency
   - movement and stage-interaction patterns
   - speech/vocal priority tendencies

3. Audience-response memory
   - audience absorption shifts
   - crowd-noise pressure windows
   - speech intelligibility pressure during announcements
   - recurring density changes by set segment

4. Operator interaction memory
   - repeated overrides
   - rejected recommendations
   - accepted operator-assistance patterns
   - confidence calibration against human intervention

5. Policy evolution memory
   - validator drift history
   - repeated blocked reasons
   - confidence decay outcomes
   - rollback history for adaptive updates

## Canonical Replay Memory Model

The framework should add three replay-governed artifact families.

### 1. Observation Bundle

`replay_memory_observation_bundle/v1`

Purpose:

- immutable input to consolidation;
- derived from one replayed session or one offline evaluation session;
- never rewritten after creation.

Contents:

- `bundle_id`
- `source_kind`
  - `replay_manifest`
  - `offline_candidate_evaluation`
  - `live_trim_snapshot_replay`
- `source_refs`
  - manifest id
  - checkpoint id
  - executor trace signature
  - validator schema version
  - report path or report id when available
- `environment`
  - venue id or venue fingerprint
  - performer set id or session labels
  - audience context labels
  - operator id or operator fingerprint when available
- `observations`
  - venue observations
  - performer observations
  - audience observations
  - operator observations
  - policy observations
- `governance`
  - `dry_run_only=true`
  - `live_mixer_writes=false`
  - validator result summary
  - evidence quality score
  - blocked reasons
- `created_at`

Rules:

- bundles are append-only;
- every bundle must reference replay-safe source artifacts;
- bundles are rejected if validator findings contain replay-safety errors.

### 2. Consolidated Snapshot

`replay_long_term_memory_snapshot/v1`

Purpose:

- durable approved memory state used by future offline replay and recommendation ranking;
- versioned and rollbackable.

Contents:

- `snapshot_id`
- `parent_snapshot_id`
- `policy_version`
- `coverage`
  - venue coverage
  - performer coverage
  - audience coverage
  - operator coverage
- `venue_memory`
- `performer_memory`
- `audience_memory`
- `operator_memory`
- `policy_memory`
- `confidence_summary`
- `decay_summary`
- `rollback_ref`
- `created_from_bundle_ids`
- `created_at`

Rules:

- snapshots are immutable once written;
- only one snapshot becomes the approved head at a time;
- promotion requires validator-compatible consolidation policy and deterministic merge output.

### 3. Consolidation Report

`replay_memory_consolidation_report/v1`

Purpose:

- explain why a new snapshot was or was not promoted.

Contents:

- source bundle inventory
- merge candidates considered
- evidence admitted and rejected
- confidence updates
- decay applied
- conflicts detected
- rollback impact
- governance findings
- promotion decision

## Per-Domain Schema Boundaries

Each domain should preserve the same internal shape:

- `entity_key`
- `evidence_count`
- `last_seen_at`
- `first_seen_at`
- `confidence`
- `stability`
- `decay_state`
- `transfer_scope`
- `features`
- `supporting_replay_refs`
- `conflicts`

### Venue Memory

`venue_memory[*].features` should be limited to replay-safe aggregate traits:

- room classification tendencies
- repeated room compensation shapes
- likely problematic frequency regions
- stage-to-room speech pressure conditions
- transition timing pressure by scene or set segment

It should not store raw audio, host data, live network coordinates, or direct console credentials.

### Performer Memory

`performer_memory[*].features` should focus on behavior patterns:

- trim adjustment volatility
- fader stability
- repeated proximity-effect or bleed tendencies
- speech dominance windows
- stage-movement triggered adaptation needs

This domain must remain performer-scoped, not globally applied by default.

### Audience Memory

`audience_memory[*].features` should capture:

- expected room absorption drift
- crowd-noise bursts
- talkover pressure
- dense-set energy windows
- seating or occupancy class effects when inferred safely

This data should remain aggregate and anonymous.

### Operator Memory

`operator_memory[*].features` should capture:

- repeatedly accepted recommendation classes
- repeatedly rejected recommendation classes
- typical override thresholds
- preferred escalation style
- confidence recalibration signals

Operator memory must never bypass approval requirements for live writes.

### Policy Memory

`policy_memory[*].features` should capture:

- validator drift frequency
- repeated blocked reasons
- replay signature incompatibilities
- rollback-triggering conditions
- confidence inflation or decay errors

This is the governance spine for future adaptive policy tuning.

## Consolidation Flow

The framework should use a five-stage replay-only pipeline.

1. Ingest
   - load replay manifests, executor traces, validator reports, offline evaluation reports, and trim snapshots
   - reject any artifact marked live-write or non-dry-run

2. Normalize
   - convert artifacts into observation bundles
   - derive bounded domain observations with stable identifiers
   - preserve all replay correlation references

3. Merge
   - compare observations against the current approved snapshot
   - apply bounded merge rules per domain
   - compute confidence growth, stability change, and decay

4. Govern
   - run consolidation-specific validation
   - reject low-quality or policy-incompatible updates
   - emit a consolidation report with accepted and rejected evidence

5. Promote
   - write a new immutable snapshot
   - retain rollback linkage to the prior approved snapshot
   - expose the new head only to offline replay consumers first

## Confidence, Aging, and Decay

Long-term memory must age naturally.

Required mechanics:

- confidence grows only from repeated compatible evidence;
- confidence decays when evidence becomes stale;
- conflicts reduce stability before they change approved behavior;
- low-support entities remain advisory only;
- cross-venue transfer requires stricter support thresholds than same-venue reuse.

Recommended bounded fields:

- `confidence`: `0.0..1.0`
- `stability`: `0.0..1.0`
- `freshness_days`
- `evidence_half_life_days`
- `conflict_ratio`
- `promotion_threshold`
- `transfer_threshold`

## Rollback and Replay Safety

Every promoted snapshot must be reversible.

Required rollback surfaces:

- previous approved snapshot id
- consolidation report id
- source bundle ids
- validator summary used at promotion time

Rollback triggers:

- validator drift spikes after promotion
- branch benchmark regression
- operator rejection clusters
- cross-venue transfer mismatch
- deterministic merge instability

Rollback action:

- restore previous approved snapshot;
- mark the reverted snapshot as superseded;
- preserve all source bundles for later re-analysis.

## Governance Invariants

The framework must fail closed on these rules.

1. No consolidation from artifacts with `live_mixer_writes=true`.
2. No consolidation from artifacts with `dry_run_only=false`.
3. No direct mutation of live runtime state.
4. No deletion of prior snapshots or observation bundles.
5. No unbounded schema fields that hide transport or environment secrets.
6. No cross-domain promotion without explicit evidence references.
7. No policy evolution without a consolidation report and rollback reference.

## Suggested Implementation Boundaries

Safe initial module split:

- `ai_mixing_pipeline/decision_layer/replay_memory_observation.py`
  - bundle schema
  - artifact normalization

- `ai_mixing_pipeline/decision_layer/replay_memory_snapshot.py`
  - long-term snapshot schema
  - versioning
  - rollback references

- `ai_mixing_pipeline/decision_layer/replay_memory_consolidator.py`
  - merge rules
  - confidence and decay logic

- `ai_mixing_pipeline/decision_layer/replay_memory_validator.py`
  - consolidation governance checks
  - promotion eligibility

- `ai_mixing_pipeline/decision_layer/replay_memory_report.py`
  - human-readable promotion and rejection reports

This keeps TUB-280 entirely within offline replay governance.

## Integration Order

Phase 1 should not consume live runtime directly.

1. ingest replay checkpoint manifests
2. ingest replay executor results
3. ingest replay policy validation reports
4. ingest offline candidate evaluation summaries
5. ingest live-input-trim replay snapshots

Only after those are deterministic should later work consider optional read-only live telemetry export into replay bundles.

## Verification Plan

Required tests for first implementation pass:

1. bundle creation rejects non-dry-run artifacts
2. bundle creation rejects live-write manifests
3. same inputs produce identical bundle ids
4. same snapshot base plus same bundles produces identical promoted snapshot ids
5. conflicting evidence lowers stability without silently mutating approved head
6. stale evidence decays confidence predictably
7. rollback restores the previous approved snapshot exactly
8. cross-venue transfer remains blocked below threshold
9. consolidation reports preserve evidence references and rejection reasons
10. validator failures block promotion

## Safe Next Tasks

1. Add replay memory observation and snapshot dataclasses with deterministic ids.
2. Add consolidation validator rules that mirror replay-policy fail-closed behavior.
3. Add bundle builders from replay manifest, executor, validator, and offline report inputs.
4. Add confidence and decay primitives with fixed thresholds and tests.
5. Add immutable snapshot promotion plus rollback references.
6. Add report writing for consolidation decisions and rejected evidence.
7. Add fixture-based tests for venue, performer, audience, and operator domains.
8. Add cross-venue transfer gates with stricter thresholds than same-venue merges.
9. Add benchmark scenarios that include consolidation pressure on top of TUB-266 stress suites.
10. Update architecture docs after implementation so live/runtime docs do not imply direct adaptive writes.
