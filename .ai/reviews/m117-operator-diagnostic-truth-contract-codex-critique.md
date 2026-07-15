# Codex Critique of Independent Proposal: M117

## 1. Strongest part of the other proposal

The independent proposal correctly treats append-only history and future test
coverage as one problem: append an M117 control-plane row, but continue finding
and pinning the unique M116 diagnostic row anywhere in ledger history. This is
stronger than a prose-only patch and preserves the physical facts after the
latest row advances.

## 2. Possible mistakes or blind spots

The proposed file list should explicitly include the stale M116 plan status:
its CP04-CP06 rows still say partial/pending after both M116 commits and final
validation completed. It should also correct `Docs/ROADMAP.md` saying M115
preserves an M113 identity `fail-open`; the implemented gate is fail-closed.
Neither fix changes architecture, but both are current truth defects.

## 3. Where the solution is too complex, fragile, or costly

Do not redesign `STATE.json.last_validation_summary.authority`. That object is
the no-WING validation authority surface and many historical rails correctly
expect its contact flags to remain false. The typed exception belongs in ledger
history and `active_work.m116_hw01_no_signal_attempt`, not in a generalized
state-authority schema migration.

## 4. What should be kept from the other proposal

Keep the append-only M117 row, exact blocker mirror synchronization, durable
historical M116 test, evidence digest pins, explicit operator-observation class,
no-hardware M117 boundary and marker-only CP05 rebind.

## 5. My revised recommendation

Adopt the independent proposal with two bounded additions: mark the completed
M116 validation/source/rebind checkpoints complete, and replace the misleading
M113 `fail-open` roadmap phrase with `fail-closed identity gate`. Do not touch
production code, live policy, standing authority or the canonical no-WING
contact flags.
