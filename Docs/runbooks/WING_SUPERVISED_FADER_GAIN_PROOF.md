# Runbook: WING Supervised Fader/Gain Proof

Status: blocked until a clean read-only preflight candidate is manually reviewed

## Goal

Prove the narrow supervised live-write loop for WING Rack:

1. exact operator-approved proposal;
2. bounded fader or gain-down / trim change;
3. physical readback after apply;
4. rollback through the supervised rollback gate;
5. rollback physical readback;
6. emergency-stop block proof;
7. operator-visible artifact.

## Hard Boundary

Allowed write kinds for this MVP:

- `fader`
- `gain` with a negative delta only

Forbidden live writes:

- EQ
- compressor
- gate
- routing
- mute/unmute
- FX
- scenes
- matrices
- DCA changes
- full autonomous mode

## Preconditions

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` has produced a clean real artifact.
- The read-only artifact was manually reviewed.
- The review decision ledger has exactly one accepted decision for the clean
  preflight candidate and no duplicate or conflicting decision records.
- The next supervised proof plan lists the accepted read-only preflight
  `summary.json` source id that anchors the physical proof.
- `.ai/project-layer/wing_mvp_blockers.md` shows read-only preflight as reviewed
  or ready for supervised proof.
- The operator explicitly approves the exact channel, value type, and delta.
- The command repeats that approval with matching `--operator-confirm-channel`,
  `--operator-confirm-value-type`, and `--operator-confirm-delta-db` flags.
- The selected delta is small and negative.
- The operator is present and can stop the test.

## Fader Proof Command

Replace channel `3` and delta `-0.25` with the operator-approved values:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type fader \
  --delta-db -0.25 \
  --operator-confirm-channel 3 \
  --operator-confirm-value-type fader \
  --operator-confirm-delta-db -0.25 \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop \
  --artifact-root artifacts/wing_operator_proposal_validation \
  --reason "supervised_wing_mvp_fader_proof"
```

## Gain-Down / Trim Proof Command

Use only a negative delta:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type gain \
  --delta-db -0.25 \
  --operator-confirm-channel 3 \
  --operator-confirm-value-type gain \
  --operator-confirm-delta-db -0.25 \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop \
  --artifact-root artifacts/wing_operator_proposal_validation \
  --reason "supervised_wing_mvp_gain_down_proof"
```

## Expected Result

The artifact must show:

- `status: passed`
- `proof_runner: wing_operator_proposal_validation_runner`
- `write_policy: apply_operator_proposal_only`
- apply went through `apply_operator_proposal`
- rollback went through `rollback_last_manual_write`
- emergency stop was applied
- later apply was blocked with `emergency_stop_active`
- fader/gain physical readback was recorded after apply and rollback

## Fail Closed If

- read-only preflight was not reviewed first;
- duplicate or conflicting read-only preflight review decision JSON files are
  still present;
- operator approval is not explicit;
- operator confirmation flags are missing or do not match the requested
  channel, value type, and delta;
- the delta is positive;
- the value type is not `fader` or `gain`;
- rollback readback is missing;
- emergency stop does not block a later apply;
- timeout happens after `apply_operator_proposal` and physical state is not
  manually inspected;
- any direct write path is used instead of `apply_operator_proposal`.

## After The Run

Update:

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- operator-visible proof report in `Docs/reports/`

Do not claim full autonomous readiness.
