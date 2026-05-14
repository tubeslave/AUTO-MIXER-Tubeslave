# TUB-266 Replay-Safe Intelligence Stress and Scale Framework

## Scope

This framework is replay-only.

- It builds deterministic stress workloads for the Analyzer -> Critic -> Decision -> Executor -> Rewind path.
- It uses only replay-safe proposal builders, ranking, manifest generation, executor simulation, rewind restore, and policy validation.
- It does not send OSC.
- It does not modify runtime orchestration.
- It does not attach to live mixer transport.

Primary implementation surface:

- `ai_mixing_pipeline/decision_layer/replay_stress_scale.py`
- `tests/test_replay_stress_scale.py`

## What The Framework Measures

Per scenario, the framework records:

- analyzer workload construction latency;
- decision ranking latency;
- timeline and checkpoint generation latency;
- manifest build latency;
- executor batch latency;
- rewind restore latency;
- policy validation latency;
- branch comparison latency;
- replay batch repetition latency.

Per scenario, the framework also records:

- proposal count;
- critic event count;
- timeline event count;
- executor event count;
- replay correlation inventory;
- timeline, checkpoint, manifest, trim snapshot, and executor trace byte size;
- branch signature diversity;
- determinism stability across repeated ranking, executor, and rewind passes.

## High-Load Scenario Catalog

The framework supports deterministic scale knobs through `ReplayStressScenarioSpec`:

- `analyzer_events`: proposal volume entering the intelligence graph
- `critic_passes`: number of synthetic critic scoring passes attached to each proposal
- `decision_burst_size`: ranking burst size after proposal generation
- `executor_batches`: repeated executor simulations for the same replay graph
- `checkpoint_copies`: repeated manifest/checkpoint generation count
- `rewind_iterations`: repeated replay restore count
- `branch_variants`: alternate branch rankings for comparison pressure
- `scenario_repetitions`: repeated replay batches that simulate concurrent offline workloads

Recommended baseline scenarios:

1. `baseline_small`
   - `analyzer_events=8`
   - `critic_passes=2`
   - `decision_burst_size=8`
2. `critic_heavy`
   - `analyzer_events=16`
   - `critic_passes=6`
   - `decision_burst_size=16`
3. `executor_batch_heavy`
   - `analyzer_events=18`
   - `executor_batches=6`
   - `rewind_iterations=5`
4. `branch_compare_heavy`
   - `analyzer_events=20`
   - `branch_variants=5`
   - `scenario_repetitions=4`
5. `checkpoint_pressure`
   - `analyzer_events=24`
   - `checkpoint_copies=6`
   - `critic_passes=4`

## Replay-Only Governance

The framework fails closed around replay safety expectations:

- every generated proposal remains `dry_run_only=true`;
- every manifest must remain `dry_run_only=true`;
- every manifest must keep `live_mixer_writes=false`;
- every executor run must remain dry-run-only;
- the framework declares `transport_mutation_allowed=false`;
- the framework declares `runtime_modifications=false`;
- the framework declares `orchestration_changes=false`.

This means the framework is suitable for scale rehearsal of intelligence logic, not for live-console validation.

## Saturation Signals

The current framework exposes these bottleneck signals:

- proposal count growth;
- critic inventory per proposal;
- timeline artifact byte growth;
- graph checkpoint byte growth;
- manifest byte growth;
- trim snapshot byte growth;
- executor trace byte growth;
- branch signature fan-out;
- repeated executor trace stability under batch replay.

These are replay-safe proxies for future real-time pressure points.

## Determinism Checks

Each scenario validates:

- ranking signature stability on repeated ranking;
- selected proposal stability on repeated ranking;
- executor trace stability across repeated executor batches;
- executor trace stability across repeated replay batches restored from rewind context;
- rewind manifest identity stability across repeated restore operations;
- replay policy validator success for the resulting manifest and executor trace.

## Current Limits

This framework does not yet do the following:

- wall-clock real-time scheduling pressure;
- true multithreaded contention;
- transport queue backpressure against a real console;
- UI/operator intervention simulation;
- memory profiling beyond artifact byte-size proxies.

Those are intentionally out of scope until replay-only scale validation remains consistently deterministic.

## Safe Next Steps

1. Add persistent JSON report writing for named stress suites.
2. Add memory-usage sampling around large manifest/timeline builds.
3. Add regression fixtures for selected stress scenarios.
4. Compare replay stress metrics across branches in CI.
5. Add threshold alerts for manifest/timeline growth.
6. Add issue-linked benchmark history so saturation trends are visible over time.
