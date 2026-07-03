# TUB-250 Replay Observability Map

## Scope

Cold-start QA analysis for `TUB-250` focused on replay-safe traceability, dry-run verification, and proof that dangerous OSC writes are not sent accidentally from the currently gated paths.

No production code was modified in this heartbeat. This work product is analysis-only and replay-only.

## Current Replay/Observability Surfaces

### 1. Decision proposal trace surface

- `ai_mixing_pipeline/decision_layer/replay_proposal_ranking.py`
- Current artifacts:
  - deterministic `proposal_id`
  - deterministic `replay_signature`
  - per-proposal `current_state`, `requested_state`, `confidence`, `safety`, `metadata`
  - ranking output with `run_signature`, `ranked_proposals`, `ranking_trace`, `selected_proposal_id`
- Value:
  - this is the strongest existing base for Analyzer -> Critic -> Decision replay introspection
- Gap:
  - not yet unified with report writers or live-apply audit IDs

### 2. Offline report trace surface

- `ai_mixing_pipeline/reports/writers.py`
- `ai_mixing_pipeline/reports/decision_report_writer.py`
- Current artifacts:
  - `decision_log.jsonl`
  - `critic_scores.csv`
  - `accepted_actions.json`
  - `rejected_actions.json`
  - summary markdown
- Value:
  - candidate-level replay evidence exists
- Gap:
  - no shared schema for analyzer outputs, critic snapshots, proposal timelines, correlation IDs, or deterministic timing metadata

### 3. Production mix dry-run OSC trace surface

- `automixer/production_mix_v1/osc_queue.py`
- Current artifacts:
  - queued `OSCCommand` payloads
  - `would_send` reconstruction in dry-run mode
  - live flush only when `dry_run=False` and a sender is attached
- Value:
  - supports replay-safe "what would have been sent" artifacts
- Gap:
  - does not stamp queue events with a replay run ID or upstream decision correlation ID

### 4. Tier A live-apply audit surface

- `backend/live_apply.py`
- Current artifacts:
  - `audit_id`
  - `source`, `operation`, `channel`, `requested_value`
  - `dry_run`, `confirm_live_apply`, `accepted`
  - `send_status`, `readback_status`, `blocked_reason`
  - `desired_value`, `confirmed_value`, `verification_id`
  - `failure_reason`, `timeout_ms`, `guard_reasons`
  - `metadata`
- Value:
  - strongest current proof trail for manual UI / voice live-write gating
- Gap:
  - audit events are not yet linked back to replay proposal IDs, critic snapshots, or deterministic timeline artifacts

### 5. Replay regression/safety harness

- `tests/replay_support.py`
- `tests/test_replay_harness.py`
- `tests/test_live_apply_contract.py`
- `tests/test_direct_write_inventory.py`
- Current artifacts:
  - socket/python-osc transport blockers
  - `RecordingSender` incident records
  - `ReplayMixer` mutation/query event logs with source file and source function
  - direct-write inventory coverage across multiple mutation-capable paths
- Value:
  - strongest current QA proof that replay tests stay off real transport
- Gap:
  - replay events are local to test fixtures and not materialized into a reusable artifact/schema for offline debugging

## Safety Claims Supported By Current Source

### Proven by source structure

- Manual live apply is default dry-run in `backend/live_apply.py`.
- Confirmation alone does not disable dry-run.
- Unsupported Tier A operations fail closed.
- Readback failure or timeout does not count as accepted live apply.
- `production_mix_v1` queue can describe intended OSC without sending when dry-run is enabled.
- Replay tests install blockers for `socket.sendto` and python-osc send methods.

### Proven by targeted test run in this heartbeat

Command attempted:

```bash
python3 -m pytest -q tests/test_replay_harness.py tests/test_live_apply_contract.py tests/test_direct_write_inventory.py
```

Observed result:

- 64 passed
- 5 failed

Failures:

1. `tests/test_replay_harness.py`
   - 2 async tests failed because async pytest support plugin is missing in the environment, not because a transport safety assertion failed.
2. `tests/test_direct_write_inventory.py`
   - 3 behavior failures in arrangement automation / auto-fader expectations:
     - no event emitted where one was expected
     - replayed safe offsets differ from recorded offsets
     - min/max fader clipping assertion mismatch

Implication:

- The replay/live-apply gate evidence is strong.
- The arrangement/auto-fader inventory tests currently show unresolved behavior drift and need dedicated follow-up before stronger end-to-end safety claims are made for those paths.

## What Is Missing For TUB-250

### Missing observability primitives

- one canonical replay trace schema
- one canonical correlation ID that survives analyzer -> critic -> proposal -> ranking -> write-plan -> audit
- deterministic event timeline artifact with stable ordering fields
- replay timing fields separated into:
  - logical order
  - wall-clock capture time
  - replay processing duration
- snapshot schema for:
  - analyzer outputs
  - critic outputs
  - proposal set
  - selected decision
  - resulting write-plan

### Missing operator/debug outputs

- trace map document for a single replay run
- diffable replay-vs-replay comparison artifact
- one place that answers:
  - why a proposal existed
  - why it ranked where it ranked
  - whether it was dry-run-only
  - whether a write was blocked
  - whether readback confirmed or failed

## Recommended Implementation Split

### Child task 1

Create a canonical replay trace schema and deterministic timeline artifact for Analyzer -> Critic -> Decision.

### Child task 2

Thread correlation IDs and replay signatures through report writers, live-apply audit events, and queued write plans.

### Child task 3

Turn the current replay test fixture event logs into persisted snapshot/debug artifacts and add comparison-oriented regression checks.

## QA Recommendation

Do not claim end-to-end replay observability is complete yet.

Current state is better described as:

- strong local building blocks exist
- live-apply gating is observable and mostly well-tested
- replay proposal ranking already has deterministic identity primitives
- the repository still lacks a unified observability layer and deterministic timeline product
- arrangement automation and auto-fader replay inventory need separate stabilization work

## Safe Next Steps

1. Land a canonical `ReplayTraceEvent` schema and version it.
2. Add a deterministic timeline writer that emits JSONL with stable sequence numbers.
3. Add `trace_id`, `parent_trace_id`, and `proposal_id` propagation into report writers.
4. Add `trace_id` propagation into `LiveApplyRequest.metadata` and audit events.
5. Add a replay snapshot bundle format for analyzer, critic, proposal, ranking, and write-plan artifacts.
6. Export `ReplayMixer` and `RecordingSender` event logs as reusable artifact payloads instead of test-only structures.
7. Add golden-file comparison tests for deterministic ranking and timeline emission.
8. Fix async test environment support for replay handler tests.
9. Triage the three failing inventory tests before expanding safety claims for arrangement automation and auto-fader.
10. Add a single operator-facing markdown or HTML trace map for one replay run.
