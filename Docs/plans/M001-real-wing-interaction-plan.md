# Plan M001: Real WING Rack interaction path

Status: active operator plan, accepted read-only decision recorded; the
supervised CH3 fader and gain-down proofs have passed on the real console with
operator approval, and the next external steps are the M005
disconnect/timeout/stale-state protection proof and the M006 owner readiness
review.

## Purpose

This plan defines the full safe interaction path between AUTO-MIXER and the real
Behringer WING Rack. It turns "test with the real console" into staged evidence:
read-only proof first, then one narrow supervised fader/gain-down write proof,
then broader soundcheck analysis as proposal-only evidence until separate
hardware-backed approvals exist.

The plan does not grant full autonomous live mixing authority.

## Current Gate

The current gate is not transport failure. Read-only WING commands have worked:
connection, channel-name scan, physical readback and observation soundcheck
analysis. The canonical read-only decision ledger is accepted-ready, and the
earlier failed CH3 fader rollback is closed by the passed rerun
`20260704T045741Z_wing_operator_proposal_ch3_fader` plus the passed gain-down
proof `20260704T045820Z_wing_operator_proposal_ch3_gain`, both with apply,
physical readback, rollback and emergency-stop/block evidence.

Current gate state:

- decision review status: `accepted_decision_ready`;
- accepted decision records: `1`;
- rejected decision records: `2`;
- invalid decision records: `0`;
- unreviewed read-only source ids: `0`;
- latest supervised proof status: `passed`;
- latest supervised proof reason: `supervised_wing_mvp_gain_down_proof_ch3`;
- backend WING runtime: observation-only unless explicitly reconnected for the
  supervised proof step;
- allowed current WING live-write proof types: `fader` and `gain` only;
- allowed current write direction: small negative move only.

## Hard Safety Boundary

Allowed after the gate opens:

- one operator-approved `fader` proof, negative dB move only;
- one operator-approved `gain` proof, input-trim/gain-down only;
- physical pre-read, apply readback, rollback readback and emergency-stop proof;
- evidence artifact capture.

Forbidden until a separate ADR and hardware proof expand scope:

- full autonomous soundcheck apply;
- fader boosts or input gain-up;
- EQ, compressor, gate, mute, routing, FX, scene, matrix or DCA writes;
- unattended real-console control;
- treating offline, mock, tmp-root or proposal-only evidence as readiness.

## Stage 0 - Operator Setup

Operator actions:

1. Confirm WING Rack is the intended console.
2. Confirm firmware is the expected WING Rack firmware for the current proof
   gate.
3. Keep the operator physically present with the console reachable.
4. Choose a low-risk channel for proof, default channel `3` unless the operator
   selects another channel.
5. Keep PA/headphones/monitors in a safe state before any write proof.
6. Do not expect AutoSoundcheck to apply broad corrections during this plan.

Codex/backend actions:

1. Keep backend and frontend running.
2. Confirm WebSocket backend is reachable.
3. Confirm dashboard safety status before every proof step.
4. Do not send live writes in unattended automation.

## Stage 1 - Read-Only Connection And Scan

Goal: prove the backend can see the real WING without mutating it.

Allowed actions:

- connect WING with `observation_only=true`;
- scan channel names;
- query selected readback values such as fader, gain/trim and pan;
- run read-only preflight into `artifacts/wing_readonly_preflight`;
- produce `summary.json` and `events.jsonl`;
- verify every sent request is read-only.

Evidence required:

- `summary.json` under `artifacts/wing_readonly_preflight/<run_id>/`;
- matching `events.jsonl`;
- physical readback statuses, not cache-only readbacks;
- `live_authority_granted: false`;
- `write_request_sent: false`;
- `mutates_console_state: false`.

Stop if:

- readback comes only from cache;
- event log contains writes;
- run id or artifact path does not match the canonical root;
- the wrong backend/WING session is connected.

## Stage 2 - Operator Decision Review

Goal: convert read-only evidence into an accepted or rejected operator decision.

Required report:

- `Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md`

Operator actions:

1. Pick exactly one clean read-only candidate to accept.
2. Reject stale, duplicate or unintended candidates.
3. Edit each generated decision JSON:
   - set `decision` to `accepted` or `rejected`;
   - set `reviewed_by`;
   - set `reviewed_at` to a UTC timestamp after `artifact_captured_at`;
   - remove `template_status` or replace the pending marker.

Validation command:

```bash
PYTHONPATH=backend python3 tools/review_wing_readonly_preflight_decisions.py \
  --artifact-root artifacts/wing_readonly_preflight \
  --decision-root artifacts/wing_readonly_preflight_review_decisions \
  --output Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md \
  --quiet
```

Gate opens only when:

- ledger status becomes `accepted_decision_ready`;
- exactly one accepted clean candidate is available;
- no duplicate or conflicting records remain;
- accepted candidate trace gates are valid:
  - prestart trace;
  - verifier WebSocket proof;
  - artifact-root trace;
  - request-sequence trace;
  - contact trace;
  - expected-artifact trace.

## Stage 3 - Supervised Write-Proof Session

Goal: prove the narrow live-write loop on the real WING and immediately roll it
back.

Required operator approval:

- exact channel;
- exact value type: `fader` or `gain`;
- exact delta, negative only;
- accepted read-only source id.

Before execute:

1. Reconnect WING only for the supervised proof session with
   `observation_only=false`.
2. Keep all proof writes inside `apply_operator_proposal`.
3. Let the runner prove apply is blocked outside `supervised` mode.
4. Let the runner switch to `supervised` only for the approved proof.

Fader proof command template:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type fader \
  --delta-db -0.25 \
  --operator-confirm-channel 3 \
  --operator-confirm-value-type fader \
  --operator-confirm-delta-db -0.25 \
  --accepted-readonly-source-id artifacts/wing_readonly_preflight/<run_id>/summary.json \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop \
  --artifact-root artifacts/wing_operator_proposal_validation \
  --reason "supervised_wing_mvp_fader_proof"
```

Gain-down proof command template:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type gain \
  --delta-db -0.25 \
  --operator-confirm-channel 3 \
  --operator-confirm-value-type gain \
  --operator-confirm-delta-db -0.25 \
  --accepted-readonly-source-id artifacts/wing_readonly_preflight/<run_id>/summary.json \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop \
  --artifact-root artifacts/wing_operator_proposal_validation \
  --reason "supervised_wing_mvp_gain_down_proof"
```

Expected evidence:

- pre-read `verified_from_physical_readback`;
- apply result verified through physical readback;
- rollback applied through `rollback_last_manual_write`;
- rollback readback verified physically;
- emergency stop applied;
- later apply blocked with emergency-stop evidence;
- `wing_surface`, `wing_address` and `proof_direction` match the runbook.

Stop if:

- delta is positive;
- value type is not `fader` or `gain`;
- selected WING address is outside `/ch/<channel>/fdr` or
  `/ch/<channel>/in/set/trim`;
- pre-read/apply/rollback readback is cache-only;
- proof proposal bypasses `apply_operator_proposal`;
- emergency stop cannot block a later apply.

## Stage 3.5 - M005 Disconnect/Timeout/Stale-State Proof

Goal: prove that loss of WING connectivity fails closed before any stale data
or disconnected write can be treated as safe.

Required operator approval:

- WING Rack is the intended console and the operator is physically present;
- proof channel, value type and negative delta match the command;
- physical cable-loss confirmation token is entered only after the operator has
  actually pulled the WING network link for this proof;
- accepted read-only source id is the current operator-accepted preflight
  artifact.

The reusable runner is:

```bash
python3 tools/wing_timeout_disconnect_proof_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type fader \
  --delta-db -0.25 \
  --operator-confirm-channel 3 \
  --operator-confirm-value-type fader \
  --operator-confirm-delta-db -0.25 \
  --accepted-readonly-source-id artifacts/wing_readonly_preflight/<run_id>/summary.json \
  --disconnect-mode physical-cable-loss \
  --operator-confirm-physical-cable-loss WING_CABLE_PULLED_BY_OPERATOR \
  --reconnect-wing-ip <WING_IP> \
  --execute \
  --artifact-root artifacts/wing_timeout_disconnect_proof \
  --reason "m005_disconnect_timeout_stale_state_proof"
```

Expected evidence:

- physical baseline readback before loss;
- operator-confirmed physical cable loss;
- stale-state query reports readback unavailable, not cache-fresh;
- `apply_operator_proposal` is blocked while disconnected;
- reconnect recovery confirms the channel value is unchanged from baseline;
- `live_authority_granted=false` and `readiness_closure_allowed=false` remain
  in the artifact.

Stop if:

- the runner observes live or stale physical readback after disconnect;
- any direct write request type appears in the event log;
- blocked-write probe applies unexpectedly without rollback/manual review;
- reconnect readback is unavailable or differs from baseline.

## Stage 4 - Readiness Update

Goal: update readiness only from accepted real proof.

Allowed after successful proof:

- record artifact path in reports/state;
- update readiness matrix rows for the exact proven surfaces;
- keep full autonomous and broad soundcheck apply blocked;
- keep EQ/dynamics/FX/routing/etc. in proposal-only or read-only mode.

Not allowed:

- mark full AutoSoundcheck apply ready;
- widen live write kinds beyond `fader` and `gain`;
- use tmp-root or observation-only runs as readiness closure.

## Stage 5 - Real Soundcheck Analysis

Goal: use the real console/audio session to generate useful corrections without
unsafe broad writes.

Allowed now:

- run AutoSoundcheck in observation mode;
- import/read proposed corrections as operator-visible evidence;
- label source identities conservatively;
- use Ayaic Mix Monolith planes for gain/balance analysis;
- separate "recommended" from "applied".

Allowed only after the supervised proof passes:

- apply a narrow operator-approved fader/gain-down correction through the same
  supervised proposal/apply/rollback/readback path.

Still blocked:

- automatic EQ/compressor/gate/FX/routing/pan/master apply to WING;
- fader boosts or gain-up from short soundcheck material;
- direct OSC write APIs outside the supervised proposal path.

## Stage 6 - Failure Handling

On any unexpected state:

1. Stop sending commands.
2. Query current readback state.
3. Roll back the last manual write if a write occurred.
4. Trigger or preserve emergency stop if rollback/readback is not verified.
5. Save artifacts and report the blocker.
6. Do not retry with broader authority.

## Current Executable Slice

Because the read-only decision gate is accepted-ready and the supervised CH3
fader/gain-down proofs have passed, but M005 and M006 still require
operator/owner action, unattended Codex work may execute only the safe slice:

1. no-WING quick safety gate;
2. decision review status refresh;
3. backend/frontend listener check;
4. dashboard safety snapshot;
5. optional read-only scans/preflights with `observation_only=true`;
6. M005 runner dry-runs or offline tests that do not contact WING or send
   `apply_operator_proposal`;
7. read-only soundcheck runtime observation artifacts that subscribe to meters
   and request dashboard/channel/decision/safety snapshots without creating or
   applying proposals;
8. offline long-run drift audits of existing soundcheck logs, provided the
   audit reads evidence only and cannot create, apply or retry corrections;
9. read-only operator review coverage checks that surface whether pending
   proposals include `reason`, `risk`, old/new value and rollback context,
   without mutating queue state or granting apply authority;
10. dirty-worktree split handoffs that record WING proof WIP, soundcheck
   runtime WIP and frontend UI restyle WIP as separate commit groups and
   generated review patches before the next soundcheck cycle;
11. report-only M006 owner review packets and offline verifiers that inventory
   accepted fader/gain proof artifacts, require a complete M005 summary
   contract, and keep readiness updates blocked until M005 and owner
   acceptance.

Codex must not execute or retry the supervised proof unattended; this now also
includes the M005 physical cable-loss proof, reconnect proof and any supervised
`apply_operator_proposal` probe. The CH3 fader and gain-down proofs are
complete: the operator verified CH3 state by read-only physical readback, then
approved the exact channel, value type and negative delta for each proof before
it ran. The next steps are the M005 supervised disconnect/timeout/stale-state
protection proof with the operator present and the M006 owner readiness review
that updates the readiness matrix from the accepted proof artifacts.
