# Codex proposal: staged supervised soundcheck execution

## 1. Short thesis

Keep `SupervisedConsoleTransactionCoordinator` fail-closed and atomic, but stop
using the whole musical soundcheck as its rollback boundary. Add a pure
classification/readiness layer and an ordered batch orchestrator that submits
only small, independently safe semantic child plans. A local transition issue
must remain visible without erasing unrelated work; any global safety issue
must still stop the batch before the next write.

## 2. What I understood

The current planner emits one normal transaction. Queue, server and coordinator
then validate that entire plan before the first send. Against the archived M006
state, 36 of 51 operations pass the existing local transition rules and 15 do
not, but the single blocked operation makes the executable result `0/51`.

This is not permission to call the 36 operations musically complete or to widen
live authority. Twelve of those 36 were already satisfied in the archived
physical state. The current path also lacks a trusted pre-write feedback and
physical Main-route gate for constructive moves. Therefore the immediate fix is
plan decomposition and truthful readiness, not automatic fader/send boosts.

## 3. Proposed solution

### A. Separate three boundaries

1. A soundcheck **run** owns an immutable `run_id`, a `plan_revision`, stage
   order, logical correction coverage and final readiness.
2. A **dependency group** owns semantic coupling. It is the smallest unit that
   must be accepted, rejected or rolled back together.
3. A coordinator **child transaction** remains atomic and uses the existing
   snapshot, durable prepare, apply, readback, audio recheck and CAS rollback
   kernel unchanged.

### B. Add typed classification

Every logical correction receives:

- `scope`: `global`, `dependency_group`, or `operation`;
- `disposition`: `apply`, `already_satisfied`, `split_steps`,
  `operator_review`, or `forbidden`;
- exact `reason` and `required_evidence`;
- `stage`, `required_for`, original baseline and cumulative budget;
- optional `replacement_of` and replacement revision.

The classifier is proposal-only unless it is fed a trusted physical snapshot.
String transition failures from the family adapter map to local dispositions;
disconnect, stale readback, emergency generation, writer/WAL conflict, unknown
rollback, missing fresh Main evidence and insufficient headroom remain global.

### C. Emit staged child plans

Normal operations are partitioned by stage and then into one-operation or
explicit dependency-group child plans. Initial stages are conservative:

- `foundation_level`: trim-down, HPF preparation and dry rough balance;
- `source_control`: gate only with transient/tail evidence;
- `phase_alignment_review`: polarity/delay as an evidence-bound group;
- `tonal_dynamics`: EQ and compressor semantic components;
- `group_balance`: bus/DCA under separate policy;
- `spatial_fx`: pan, send, return and FX;
- destructive routing, scene, mute and master actions remain separate approval.

The first implementation may use one logical operation per child plan. This is
deliberately less efficient but makes local isolation, idempotency and rollback
truth explicit. Later grouping is allowed only when dependency semantics are
proven.

### D. Prevent step-splitting from laundering limits

`split_steps` is descriptive in the first safe implementation. Execution stays
operator-review-only until the orchestrator has all of:

- an original physical baseline;
- per-step and cumulative session caps relative to that baseline;
- a fresh snapshot and analysis between steps;
- absolute targets, bounded step count and no integrator ratchet.

Large pan, trim, threshold or EQ moves cannot become executable merely by
dividing their delta.

### E. Bound recomputation and substitution

The run records an explicit replan request when a required local correction is
removed. A replacement can resolve the original only if a trusted analyzer
produces it with `replacement_of`, no higher risk class, no new write kind
outside effective authority, fresh evidence and at most two substitution hops.
No mechanical gain-to-fader, boost-to-competitor-cut, or makeup-to-fader
conversion is treated as equivalent without source/masking/GR evidence.

### F. Keep constructive authority closed

Represent `fader_constructive_step` as a candidate lane, but do not promote it
or `send` in this change. Its future pre-write proof must include fresh physical
Main 47/48 route binding, windowed headroom, trusted source identity and own
signal, feedback/monitor freshness, positive-move cumulative budget, WAL/CAS,
post-write dwell and verified rollback. Send remains a later high-risk lane;
trim-up remains operator-only/open-loop because the USB tap is pre-trim.

### G. Publish an honest two-axis result

Expose one nested `soundcheck_readiness` object, independent of the engine
runtime `state`:

- `safety_state`: `clear`, `blocked_global`, `manual_review_required`, or
  `emergency_stopped`;
- `mix_readiness`: `soundcheck_incomplete`, `safe_core_ready`, or
  `show_mix_ready`;
- logical counts, physical step counts, stage results, unresolved required IDs,
  source coverage and Main evidence.

Logical corrections, not physical split steps, are counted. Only a terminal
committed and verified result, `already_satisfied` with a trusted snapshot, or a
verified replacement resolves a required correction. `applied_operation_ids`
alone is not success because a later audio failure can roll the transaction
back. Proposal-only derivation is incomplete unless an explicit baseline proof
establishes that no change is required.

### H. Fix run identity and duplicate truth

Transaction fingerprints remain deterministic for content, but queue identity
must include immutable `run_id` and `plan_revision`. Do not infer readiness from
the bounded proposal queue or from import success. Generic per-action proposal
rows may remain for UI review, but only the run aggregator owns soundcheck
readiness.

## 4. Likely files to touch

- `backend/soundcheck_transaction_plan.py`
- new `backend/soundcheck_execution_orchestrator.py`
- `backend/wing_supervised_console_adapters.py` for typed read-only assessment
- `backend/operator_recommendation_bridge.py`
- `backend/operator_proposal_queue.py`
- `backend/auto_soundcheck_engine.py`
- `backend/handlers/soundcheck_handlers.py`
- frontend soundcheck status types/views
- focused backend/frontend tests and truth documents

The coordinator should change only if a narrow typed result hook is required;
its atomic execution semantics are not weakened.

## 5. Alternatives considered

1. **Skip invalid operations inside the coordinator.** Rejected: it destroys
   atomic semantics and hides dependency failures.
2. **One transaction per stage.** Rejected as the first fix: one local blocker
   would still erase an entire stage.
3. **Open fader and send boosts immediately.** Rejected: current headroom check
   is post-write, feedback is not in the transaction gate, and Main route proof
   is static rather than a fresh physical binding.
4. **Apply the current 36 and declare core readiness.** Rejected: passing delta
   guards is not equivalent to audible or musical completion.
5. **Status-only patch.** Useful for truth but insufficient; it would leave the
   0/51 architectural defect intact.

## 6. Risks

- Over-fragmentation can create excessive operator interactions. Mitigation:
  one approval envelope with deterministic child execution and aggregate UI.
- Split steps can bypass cumulative limits. Mitigation: keep them non-executable
  until baseline/cumulative enforcement exists.
- A replacement can move risk elsewhere. Mitigation: provenance, risk ceiling,
  bounded hops and fresh analysis.
- Earlier child stages may commit before a later global failure. Mitigation:
  report partial committed state explicitly; never silently roll back previous
  successful stages as if the run were atomic.
- Dirty checkout can obscure ownership. Mitigation: scoped files only, no broad
  cleanup, staging or commit.

## 7. Test plan

- Exact archived M006-shaped classifier fixture: 51 logical operations,
  36 locally admissible (including 12 `already_satisfied`) and 15 local issues
  with the exact reason histogram.
- Safe child plus blocked sibling: safe child remains present; blocked sibling
  remains unresolved; run is incomplete.
- Parameterized global hazards: zero child applies.
- Coordinator child atomicity and reverse rollback tests remain unchanged.
- Dependency-group rejection, run idempotency/revision and resume tests.
- Cumulative caps prevent pan/trim/EQ/threshold split laundering.
- No replacement without trusted evidence; no risk escalation or new authority.
- EQ/gate/compressor enable semantics cannot be reported as audible success
  when bypassed.
- Constructive fader blocked without every fresh pre-write proof; send and
  trim-up remain separate/operator-only.
- Status tests prevent `safe_core_ready` or `show_mix_ready` with unresolved
  required logical corrections.
- Full backend, frontend, E2E and no-WING gates.

## 8. Where the other agent may disagree

An opposing reviewer may prefer a smaller planner-only change: emit one child
plan per operation and add status counts, deferring runtime orchestration. That
is a reasonable first merge boundary. My recommendation still defines the
full run contract now so a planner-only patch cannot accidentally become a new
live execution path or another misleading completion surface.
