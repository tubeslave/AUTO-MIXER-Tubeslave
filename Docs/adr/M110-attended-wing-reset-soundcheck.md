# ADR M110 — attended WING reset and staged soundcheck authority

Status: accepted; software implementation complete and hardware validation
pending as of 2026-07-14.

## 1. Context

The project can analyze the real WING and plan staged child transactions, while
normal live authority intentionally remains empty. M110 adds a separate,
source-bound attended path for one reset and one staged soundcheck without
widening that standing authority. The software path is implemented and tested;
it grants no hardware readiness until committed CP05 evidence and HW01-HW05
physical proof exist.

## 2. Options considered

1. Widen the standing family allowlist and run the staged plan.
2. Use the legacy direct-WING family runner and reset functions.
3. Add a dedicated generic proof/write API.
4. Add backend-issued fixed provenance plus exact one-time capabilities behind
   the existing canonical Apply evidence lifecycle.

## 3. Decision

Choose option 4.

### Route provenance

Add authenticated `capture_wing_master_route_proof`, contract v1. It accepts
only bounded request identity, approval, a fresh request nonce, the fixed
literal confirmation and the attended-launch HMAC described below. Unknown
fields and every address/expected-value/artifact field are rejected.

Its complete request object is exactly:

```json
{
  "type": "capture_wing_master_route_proof",
  "contract_version": 1,
  "request_id": "<1..128 safe ASCII>",
  "approved": true,
  "approval_id": "<1..128 safe ASCII>",
  "confirmation": "CAPTURE_WING_MAIN_USB_47_48_ROUTE_PROOF",
  "request_nonce": "<32 lowercase hex>",
  "authority_proof": "hmac-sha256:<64 lowercase hex>"
}
```

`request_id` and `approval_id` must be strings matching
`[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`; booleans, coercion and empty values are
rejected. `contract_version` must be the integer `1` (not boolean), `approved`
must be the boolean `true`, and all other values are strings with the exact
syntax shown.

Inside `_console_write_scope` and `_audio_capture_lock`, the backend performs
two complete passes over exactly:

- `/$syscfg/$serial`
- `/$syscfg/$firmware`
- `/io/out/USB/47/grp == MAIN`
- `/io/out/USB/47/in == 0`
- `/io/out/USB/48/grp == MAIN`
- `/io/out/USB/48/in == 1`

Between passes it requires fresh real WING audio, exact device/index/48
channels/48 kHz identity, sample-count progression, signal on both 47/48 and
conservative true peak `max(sample_peak) + 3 dB <= -1 dBTP`.

The audio identity is hardware-bound, not name-bound. On macOS the backend
requires exactly one CoreAudio row with the selected exact name, manufacturer
`BEHRINGER`, USB transport, 48 inputs, 48 outputs and 48 kHz, plus exactly one
`ioreg` USB device named WING with vendor ID `5015` (`0x1397`), product ID `1291`
(`0x050b`) and a stable location ID. Its USB serial must equal the OSC chassis
serial or equal that serial followed by exactly three decimal digits. The
latter is the observed WING USB-interface convention on this console
(`0100Q1V0604AAE` to `0100Q1V0604AAE214`); arbitrary longer prefixes are
rejected. The canonical binding digest, hardware UID and exact PortAudio stream
object are held in the private authority. The OS identity is recollected after
artifact persistence and before any later audio-gated write; device, stream,
USB or serial drift revokes authority.

The canonical audit artifact schema is
`automixer.wing_master_capture_route_proof.v1`. Artifacts live under a fixed
per-user 0700 root, use 0600 regular non-symlink files, atomic exclusive write,
file and directory fsync. Artifact path/SHA and any ledger are audit evidence,
not live authority: same-user files cannot authenticate their own issuer.

Live route authority is a private non-serializable object held only by the
issuing process and bound to its runtime nonce/source identity, console,
capture, emergency and connection generation. Restart, reconnect, a new route
attempt, route TTL expiry or emergency change revokes it. The attended workflow
does not restart between route proof and later capabilities; any restart repeats
the route proof. Legacy flat JSON and environment path/SHA are never accepted
as M110 live authority.

The same authenticated owner may then send exact v1
`arm_m110_attended_control` with approval ID, fresh 32-hex request nonce,
attended-launch HMAC and literal `ARM_M110_ATTENDED_ONE_SHOT_CONTROL`; no
additional fields are allowed. Inside
the writer scope the backend repeats the fixed route/identity/headroom checks,
requires the `m110_attended_runner` launch binding, clean source and healthy
WAL/emergency state, then creates process-private capability-only control for
that same connection without reconnecting. Normal operator mode stays
`analyze`, WING stays observation-only, and the standing live allowlist remains
empty. Ordinary Apply remains blocked unless an exact M110 capability is
atomically consumed. Disarm, disconnect, emergency or route expiry revokes
every unconsumed grant without changing normal mode.

The literal arm confirmation also acknowledges one operator-visible attended
no-touch interval for every Apply or Proof: it begins before the runner invokes
the executor and ends only when the runner displays the durable terminal
response. Throughout that complete interval the operator must not change any
console setting and every other OSC controller must remain disconnected. Audio
signal may continue. The wire result states
`attended_no_touch_required_during_execution:true`. The process mutation gate
blocks local writers. A physical change after the recorded verification cut is
a new concurrent console event, not retroactive evidence that the earlier cut
was false; it invalidates readiness freshness, and every later capability stage
must perform its own complete fresh preimage/CAS readback before any send.

The complete arm request is exactly:

```json
{
  "type": "arm_m110_attended_control",
  "contract_version": 1,
  "approval_id": "<1..128 safe ASCII>",
  "confirmation": "ARM_M110_ATTENDED_ONE_SHOT_CONTROL",
  "request_nonce": "<32 lowercase hex>",
  "authority_proof": "hmac-sha256:<64 lowercase hex>"
}
```

It uses the same non-coercing integer/string/regex rules as route capture.

### One-time capabilities

Add bounded server-owned token storage. Plaintext tokens are returned once and
never logged or persisted; the server stores only SHA-256. Each frozen record
binds runtime identity/nonce/HEAD/content, connection generation, console
serial/firmware, route proof ID/SHA, emergency generation, owner session,
proposal ID/review digest, canonical plan digest, exact scope/addresses/targets,
approval/idempotency and latest-start deadline.

Preparation endpoints bind server-owned reviewed proposals but never issue
tokens. The restore-only proof preparation request is exactly:

```json
{
  "type": "prepare_m110_restore_only_proof",
  "contract_version": 1,
  "proposal_id": "<server-reviewed pending proposal>",
  "bundle_selector": "reviewed_supervised_transaction"
}
```

For CP02 the referenced proposal must already exist in the same backend
process, have review contract v2 and contain exactly one supervised operation.
The server re-parses it, prepares the physical adapter without sending, builds
the canonical bundle, and returns only proposal/review/operation identity,
operation count and scope. It returns no token. CP03/CP04 add fixed server-owned
proposal constructors; the client never supplies addresses or a plan to those
constructors.

The only issuance request is exact v1:

```json
{
  "type": "issue_supervised_capability",
  "contract_version": 1,
  "capability_kind": "<fixed enum>",
  "proposal_id": "<server proposal>",
  "expected_review_contract_version": 2,
  "expected_review_content_digest": "sha256:<64 hex>",
  "expected_operation_digest": "sha256:<64 hex>",
  "approval_id": "<bounded nonempty>",
  "confirmation": "<kind-specific literal>",
  "request_nonce": "<32 lowercase hex>",
  "authority_proof": "hmac-sha256:<64 lowercase hex>"
}
```

No additional fields are accepted. Issuance requires bearer-authenticated
runtime launch binding `mode=m110_attended_runner`, the same connection owner
that prepared/reviewed the proposal, current process-private route authority,
clean committed source, zero unresolved WAL/emergency/global hazards and exact
proposal/review/operation identity. Limits are one unconsumed token per owner,
eight globally and 30 seconds to start. A second issuance revokes the previous
owner token. Disconnect, reconnect, route re-proof, emergency generation change
or proposal terminal state revokes every affected unconsumed token. Consumed
records are tombstoned for bounded replay detection and never reissued.

The successful issue response returns the plaintext token exactly once along
with the bound proposal/review/operation identity, server-generated
idempotency key and latest-start epoch. The attended runner immediately removes
the token field from the parsed response, holds the value only in a local
`bytearray`, places it in one exact Apply frame, removes it from that frame and
zeroes the bytearray as soon as WebSocket send completes. It is never copied to
session state, summaries, logs or artifacts.

The attended runner creates a pipe before spawning the backend. Of M110 launch
authority material, the environment contains only
`AUTOMIXER_M110_AUTHORITY_FD=<decimal inherited read-fd>`; launch nonce,
manifest, digest and authority secret are forbidden in environment, argv and
files. Bearer WebSocket authentication uses a distinct inherited
`AUTOMIXER_WS_AUTH_TOKEN_FD`; it never shares the authority packet. Ordinary
bounded runtime configuration such as loopback host/port and analyze mode is
not authority material. The parent writes exactly one packet to the authority
pipe and closes its write end:

1. four-byte unsigned big-endian manifest length, bounded to 64 KiB;
2. that many canonical UTF-8 JSON bytes;
3. exactly 32 random authority-secret bytes; and
4. EOF, with no trailing byte.

The manifest schema is exactly
`automixer.m110_attended_prestart_manifest.v1` and has only `schema`, `mode`,
`head`, `tree`, `launch_nonce` and `files`. `mode` is
`m110_attended_runner`; `launch_nonce` is fresh 32-lowercase-hex; `files` is a
path-sorted array of only `path`, `size` and `sha256`. Its paths must equal the
path-sorted server-constant `M110_RUNTIME_CRITICAL_PATHS` tuple exactly:

```text
Docs/reports/M110_CP05_SOFTWARE_VALIDATION.json
backend/audio_capture.py
backend/auto_soundcheck_engine.py
backend/contracts/m110_m006_restore_20260711.json
backend/decision_queue_wire_contract.py
backend/handlers/connection_handlers.py
backend/handlers/product_state_handlers.py
backend/handlers/soundcheck_handlers.py
backend/m110_cp05_hardware_gate.py
backend/m110_manual_write_wal.py
backend/m110_m006_restore.py
backend/m110_processor_enable.py
backend/maintenance_dsp_reset.py
backend/master_capture_route_proof.py
backend/operator_proposal_queue.py
backend/operator_proposal_result_store.py
backend/osc/enhanced_osc_client.py
backend/runtime_identity.py
backend/server.py
backend/soundcheck_mix_readiness.py
backend/soundcheck_transaction_plan.py
backend/supervised_console_transaction.py
backend/supervised_family_proof_plan.py
backend/supervised_proof_capability.py
backend/wing_capture_hardware_identity.py
backend/wing_client.py
backend/wing_supervised_console_adapters.py
ops/m110/CP05_HARDWARE_GATE.json
tools/wing_attended_soundcheck_runner.py
tools/wing_supervised_family_proof_runner.py
```

Paths are repository-relative canonical POSIX paths; size is the exact
non-negative byte size; SHA is `sha256:<64 lowercase hex>`.
Canonical JSON means UTF-8, keys sorted recursively, separators `,` and `:`, no
insignificant whitespace, duplicate keys, floats, NaN or Unicode normalization.

Before constructing the server or opening network/audio/console resources, the
backend consumes and closes the FD, requires exact EOF, independently resolves
the checkout, proves the tracked tree has no modifications, reads HEAD/tree and
every server-constant critical path with no symlink traversal, recomputes the
canonical manifest bytes and compares them byte-for-byte. Invalid, partial,
inherited-more-than-once or mismatched input aborts startup. Untracked generated
reports outside the critical path set are not source authority. The backend
keeps the secret only in process memory and exposes only its SHA in runtime
identity.

The runner retains the other in-memory copy solely to authenticate three exact
request kinds: route capture, arm and capability issuance. Each includes a
unique 32-hex `request_nonce` and `authority_proof`. The proof is HMAC-SHA256
with the 32-byte secret over
`automixer.m110.request.v1\n` followed by the canonical JSON bytes of the whole
request with only `authority_proof` omitted. The backend verifies it with a
constant-time comparison and atomically tombstones the nonce before work;
wrong/replayed nonces block before console pre-read. Bearer authentication and
same-owner checks remain independently mandatory. The secret is zeroized on
shutdown as far as Python permits and is never logged, persisted or returned.

This binding proves possession by the parent of this exact backend launch and
prevents a browser or unrelated client from arming it. It does not claim to
defend against a malicious same-user process able to inspect process memory or
write OSC to WING directly; that actor is already outside this repository's
software authority boundary. Persistent JSON/SHA artifacts still never confer
live authority.

### CP05 physical-contact gate

No M110 child may contact a physical console, USB capture device or discovery
surface until the backend-owned CP05 gate passes. The two committed canonical
markers are `ops/m110/CP05_HARDWARE_GATE.json` and
`Docs/reports/M110_CP05_SOFTWARE_VALIDATION.json`. The evidence requires all
focused, frontend, full backend, no-WING, static, lease-release and independent
review checks to be exactly `true`. The gate SHA binds the evidence; the
evidence source SHA binds the mode, object type, Git object ID and path of every
entry in the committed Git tree except the two markers themselves. A later
clean commit anywhere in the tracked tree therefore closes hardware access
until CP05 is rerun. Dirty tracked source also closes it.

The same verifier runs in the runner and inside the backend before transport
construction and route use. An attended backend has a centralized inbound
allowlist before handler dispatch: identity/status and fail-safe stop/disconnect
remain available even after the gate closes; fixed `connect_wing`, exact
48-channel/48 kHz/8-second capture and route proof require the gate;
scan/auto-connect, dLive, Mixing Station and every unrelated handler are
blocked. Preconnect also requires current launch source, an RFC1918 unicast host
(never host octet 0 or 255), fixed port 2223, analyze/zero-apply mode, no STOP
or writer lease, healthy WAL, no emergency and no active writer. Thus neither
raw WebSocket messages nor a future caller can bypass CP05.

#### Bounded supersession of M109 artifact-location policy — 2026-07-14

Seven exact failed read-only captures are archived byte for byte under
`artifacts/wing_readonly_preflight_invalid/`:

- `20260713T051955Z_wing_readonly_preflight`;
- `20260713T052026Z_wing_readonly_preflight`;
- `20260713T052335Z_wing_readonly_preflight`;
- `20260713T052452Z_wing_readonly_preflight`;
- `20260713T052547Z_wing_readonly_preflight`;
- `20260713T054127Z_wing_readonly_preflight`; and
- `20260713T054142Z_wing_readonly_preflight`.

Their twelve payload blobs and eight referenced sidecars are bound by original
and archive paths, size and SHA-256 in
`artifacts/wing_readonly_preflight_invalid/relocation-manifest-20260714.json`;
payload bytes and embedded original paths
remain unchanged. The archive is immutable negative provenance and is never
candidate intake. The canonical root is the candidate-intake surface, not the
complete historical store.

This decision supersedes ADR M109 only for those seven run IDs and twelve exact
hashes. It changes no production discovery code, does not delete history,
weaken deterministic invalid-artifact tests, satisfy CP05/HW01-HW05, grant
readiness or live authority, or establish wildcard/latest-valid selection.
Every future invalid capture remains fail-closed in the canonical root unless a
new ADR and exact relocation manifest separately authorize it. Rollback is the
exact reverse rename plus removal of this manifest, restoring the prior
13-total/6-valid/7-invalid intake history.

The fixed capability enum and confirmation literals are:

- `restore_only_proof` → `ISSUE_RESTORE_ONLY_PROOF_CAPABILITY`;
- `maintenance_dsp_reset_commit_once` →
  `ISSUE_MAINTENANCE_DSP_RESET_COMMIT_ONCE`;
- `processor_enable_commit_once` → `ISSUE_PROCESSOR_ENABLE_COMMIT_ONCE`;
- `soundcheck_child_commit_once` → `ISSUE_SOUNDCHECK_CHILD_COMMIT_ONCE`.

Preparation messages are also exact v1 and accept only a server proposal and a
server-defined bundle selector, never addresses or targets. Each preparation
stores the owner connection and immutable bundle digest used by issuance. The
later reset/soundcheck constructor endpoints create those proposals from fixed
selectors; the restore-only preparation endpoint itself does not create one.

The canonical proof/commit unit is
`automixer.canonical_attended_operation_bundle.v1`. It contains a bundle kind
and a server-ordered operation array, plus adapter/protocol source revision,
exact addresses/parameter model, scope, direction/value bounds, target values,
tolerances, audio contract and console/firmware/route/runtime identity. Proof
receipt ID and evidence expiry are deliberately outside operation identity;
the permanent lane checks them separately for exact same-process origin and
freshness. The five canonical bundle kinds are:

- `single_soundcheck_operation`: exactly one logical operation after a staged
  planner child is decomposed before review;
- `soundcheck_atomic_group`: a planner-declared indivisible group whose exact
  complete operation set must be proved and committed together;
- `maintenance_dsp_reset_bundle`: the fixed server-generated reset set of at
  most 64 operations;
- `m006_exact_preimage_restore_bundle`: one of the three server-owned exact
  M006 preimage stages; and
- `processor_enable_bundle`: one channel/processor complete conservative
  parameter block plus its enable-last operation.

`M110_SOUNDCHECK_BUNDLE_KINDS` is deliberately the narrower two-kind set
containing only `single_soundcheck_operation` and `soundcheck_atomic_group`;
the other three are fixed maintenance/processor bundles and are never accepted
as ordinary planner children.

The fixed reset proposal capture owns one 90-second total deadline across both
the 64-address reset sweep and the 184-address preservation sweep. A commit
recapture additionally reserves the final 30 seconds of the restored-proof
receipt window for prepare, capability issue and Apply. If that remaining
window is unavailable, the server returns a strict no-send blocked terminal;
it never starts a capture that can only expire before commit.

Every finite numeric target, tolerance and bound is encoded in the bundle as a
canonical decimal string, never a JSON float: parse with `Decimal(str(value))`,
reject boolean/non-finite input, emit fixed-point notation with no exponent or
leading plus, strip insignificant trailing fractional zeros and the trailing
dot, and normalize every signed zero to `"0"`. Integral channel/bus indices
remain JSON integers; categorical strings and booleans retain their native
types. The SHA-256 of those canonical bytes is the `operation_digest`. Proof and
permanent preparation independently regenerate the bytes. They may exclude
only transaction ID, idempotency key and the proof/commit execution label;
operation order and every safety binding remain identical.

The external execution message remains `apply_operator_proposal`. Public
`proof_run`, `restore_after_verify`, capability-kind, address, target and plan
switches never select authority. Atomic token consumption creates one of four
private non-serializable grants:

- `restore_only_proof` — exactly one canonical attended operation bundle;
- `maintenance_dsp_reset_commit_once` — fixed DSP bypass set;
- `processor_enable_commit_once` — exact complete block, enable last;
- `soundcheck_child_commit_once` — one decomposed immutable
  `single_soundcheck_operation` with an identical recent restored bundle
  digest.

The restore-only Apply object is exactly:

```json
{
  "type": "apply_operator_proposal",
  "proposal_id": "<bound proposal>",
  "approved": true,
  "approval_id": "<bound approval>",
  "idempotency_key": "<server-issued key>",
  "expected_review_contract_version": 2,
  "expected_review_content_digest": "sha256:<64 hex>",
  "expected_operation_digest": "sha256:<64 hex>",
  "supervised_capability_token": "m110cap_v1_<64 lowercase hex>"
}
```

Apply ordering is fixed. The queue first reserves durable result-evidence
capacity without changing proposal status. The server then re-derives the
exact binding and atomically consumes/tombstones the token. Only after that may
the queue claim the proposal as `applying` and reserve the bound idempotency
key. Writer acquisition, fresh CP05/route/hardware/global checks and fresh Main
47/48 headroom recheck still occur before the first adapter snapshot or send.
If claim or any later no-send check fails, the consumed grant becomes terminal
and cannot be replayed. Accept, dismiss, expiry and any Apply terminal revoke
only unconsumed authority for that proposal and clear its prepared bundle;
verified restored-proof receipts are retained until route/session revocation.

Expiry is a latest-start rule. Once a grant is consumed, rollback and recovery
authority remains until terminal state even after caller cancellation or socket
loss. Forward work still stops on disconnect/emergency/global hazard.

No capability updates the standing registry or normal allowlist.

### Proof terminal

Restore-only execution forces `proof_run=True` and
`restore_after_verify=True` internally. Success is `proof_passed` with durable
WAL `proof_restored`, identical nonempty applied/rollback operation sets,
physical state restored, verified audio, known write/rollback state, no manual
review/emergency, `accepted:false`, owner review still required, and no live
authority/readiness change. It is never treated as normal `applied`.

`physical_verified` and `physical_state_restored` are historical truth at the
recorded `verified_as_of_m110_verification_cut`, not a promise that no later
physical action occurred before the response was rendered. The terminal binds
the cut epoch plus reset-surface and preservation-manifest SHA-256 values,
states `current_state_guaranteed_at_response:false`, and requires a fresh
readback before the next stage. A proof receipt authorizes only that historical
proof; it never substitutes for the commit lane's fresh recapture.

The outbound v2 terminal is projected only from the durable result receipt. It
must report `proposal_status=apply_status=status=proof_passed`, `success:true`,
`proof_run:true`, `restore_after_verify:true`,
`capability_kind=restore_only_proof`, exact operation digest,
`proof_target_verified:true`, `physical_verified:true`,
`physical_state_restored:true`, `terminal_after_console_send:true`, verified
rollback/audio, known states, `accepted:false`, `owner_review_required:true`,
`live_authority_granted:false` and `readiness_closure_allowed:false`. The token
field or any token-shaped value in a terminal is a protocol failure.

### Maintenance reset

The first M110 reset scope is channels 1–16 and at most 64 addresses:

- `/ch/N/eq/on = 0`
- `/ch/N/flt/lc = 0`
- `/ch/N/gate/on = 0`
- `/ch/N/dyn/on = 0`

The backend creates the plan from fresh physical readback, captures a complete
preimage, prepares WAL before a send and double-verifies all writes. Before the
success-WAL attempt, a failure that leaves the reset applied restores the whole
transaction in reverse. The success-WAL attempt is irrevocable: later faults
never issue compensating console sends.

Preservation has two explicit, non-overclaiming proofs:

1. A finite fresh physical manifest reads before and after, with canonical
   sorted JSON rows and address-specific tolerances, for channels 1–16:
   `in/set/trim` (0.05 dB), `fdr` (0.05 dB), `mute` (exact), `pan` (0.1),
   `in/set/inv` (exact), `in/set/dly` (0.01 ms), `in/set/dlyon` (exact), and
   input route `in/conn/{grp,in,altgrp,altin}` (exact); plus Main 1 `fdr`/`mute`
   and the six fixed console/USB identity addresses. Every query must be
   `received`, exact-address, at least two confirmed round trips and inside a
   `120.0` second per-pass sweep deadline. Unreadable or missing rows block
   reset.
2. The maintenance adapter and send-boundary interceptor allow only the 64
   reset addresses. A temporary wildcard physical observer records only
   transition events actually delivered by WING. Before the verification cut,
   any delivered event outside the reset set blocks success and restores the
   reset surface when required. A later event invalidates freshness and cannot
   trigger compensation after the success-WAL attempt. Event silence never
   proves physical equality because not every WING surface is streamed. Sends/FX,
   buses/DCA/matrix, other Main values, scenes and names are reported only as
   `no_mutation_attempt_proven` from the send-boundary audit, with delivered
   observer events as supplemental evidence, not falsely as full physical
   equality when no finite read contract exists.

The before/after manifest bytes and no-mutation-attempt audit are both
digest-bound to the proposal and receipt. Reset remains a maintenance proposal
and cannot contribute soundcheck readiness by itself.

The success WAL is the irrevocable transaction terminal. Before it, the
observer remains active throughout the final 64+184 physical sweep and is
detached only after that sweep, while the process mutation gate remains active
and sealed. The success payload must prove that exact clean cut, and the durable
append must return the SHA-256 of the bytes it actually wrote. The production
server owns this append and computes the receipt from its own canonical fsynced
bytes; a callback-supplied or self-attested digest cannot authorize hardware.
After a success
attempt, cancellation, authority expiry or gate-release faults create a
no-write manual incident and emergency latch behind a confirmed sealed gate.
They never issue compensating sends that could leave durable `committed`
evidence while physically undoing the reset. If a sealed recovery gate cannot
be established, all external callbacks are suppressed and the terminal is
reported failed instead of risking a write. Incidents record the linearized
phase, `compensation_attempted:false` and require fresh physical reconciliation.
A failure before the success attempt may still restore the preimage under the
sealed gate.

### Exact M006 preimage restore and parent journals

The operator's M006 permission is consumed only by the fixed server-owned
restore path. It is not standing authority and it does not replay the M006
postimage. After the 64 processor-enable reset, the backend restores the exact
pre-M006 values captured by the immutable 51-operation fixture
`backend/contracts/m110_m006_restore_20260711.json`, bound to its tracked file
SHA, source event SHA and source transaction ID.

Restoration is split into three atomic stages in this fixed order:

- `foundation_level`: 23 operations, 33 scoped addresses, 19 changed;
- `tonal_dynamics`: 14 operations, 63 scoped addresses, 52 changed; and
- `spatial_balance`: 14 operations, 14 scoped addresses, 6 changed.

Across the three stages the exact changed-address union is 77 addresses. The
larger scoped set remains part of proof/CAS identity even when a row is already
satisfied and therefore emits no send.

Each stage first proves and rolls back the exact bundle, then recaptures and
commits the byte-identical bundle with compare-and-swap expected-current values.
Those expectations are the archived M006 postimage, except the four processor
enable surfaces that the preceding reset must already hold at zero. A mismatch
blocks before send. The restore changes only addresses proven different from
the preimage, keeps EQ/HPF/gate/compressor bypassed, and leaves every unknown or
manual surface outside the archived M006 transaction untouched. It therefore
means “remove the known M006 corrections,” not factory-reset the console.

Reset, M006 and staged soundcheck parents are journaled independently of child
WAL terminals. A parent commit is true only after the server appends and fsyncs
its own canonical record. Trusted wire receipts project exact applied,
already-satisfied and rollback operation IDs plus the server-bound parent
result; client terminal fields cannot forge them. Multi-operation child
outcomes and commit checkpoints append as atomic batches and replay all-or-none
after restart. A malformed later batch restores none of that journal, and an
unrecorded parent commit revokes forward authority instead of reporting a
successful stage.

### Processor enable and permanent soundcheck

After reset, one atomic `processor_enable_commit_once` grant owns both the
complete conservative parameter block and its enable state. It snapshots the
whole block while bypassed, writes and physically verifies parameters, enables
last, then audio-checks. Failure disables first, reverse-restores every
parameter and restores the original enable state. No earlier permanent child
may write those parameters. Unknown retained settings may never be made
audible by toggling only the enable bit.

Each permanent reset bundle, processor bundle or decomposed soundcheck operation
uses a separate
operator approval, idempotency key, proposal review digest and commit-once
grant. It requires a recent same-process `proof_passed` whose canonical
operation-bundle bytes and digest are identical under the exclusions defined
above. A multi-operation staged child is split into independently reviewed
single-operation bundles before proof or commit; grouping survives only in the
local stage journal and never broadens authority. Composite execution is
limited to the fixed reset, fixed M006 restore, complete processor block and a
planner-declared `soundcheck_atomic_group`; every other musical child is a
single operation. A proof for a sibling address, a subset/superset, reordered
operations or merely the same family/write kind grants nothing. Local
blockers remain local and trigger bounded replan; global hazards stop the run.
Final status is honest: `safe_core_ready`,
`show_mix_ready`, or `soundcheck_incomplete`.

### Runner boundary

The legacy family proof runner is permanently plan-only and contains no mixer
or network transport. All real proof execution lives in the attended runner,
which is WebSocket-only and contains no `WingClient`, UDP, OSC or direct write
call. After the
CP05 gate and exact operator confirmation, the isolated proof-owned child may
open OSC. One persistent authenticated WebSocket, one child process, the exact
WING transport and exact owned capture stream remain alive together from route
proof through all later capability stages; closing that socket revokes
authority. The runner establishes the inherited one-shot launch pipe above,
passes bearer tokens via separate inherited file descriptors, binds challenged
runtime identity, launches only clean committed source and guarantees shielded
child/capture/transport teardown even under cancellation. Secrets and tokens
never enter argv, logs or artifacts.

On the retained WebSocket the fixed order is route proof, one control arm, then
server-owned proposal creation, prepare, issue and Apply. The fixed reset path
runs proof first, records the durable same-process receipt, performs a fresh
complete recapture, then creates and applies the permanent commit proposal.
No address, target or transaction plan crosses from the runner. Re-arming is
not used between proposals because arm intentionally clears prepared bundles
and prior process-private proof receipts. Any malformed, blocked, duplicate or
wrong-stage terminal tears down the entire proof-owned session. The top-level
real `--execute` flow is implemented but remains unreachable until committed
CP05 passes. It runs route proof, arm, operator-confirmed Main mute, fixed reset,
three-stage M006 preimage restore, operator-confirmed Main unmute, fresh
analysis, interleaved processor/musical lanes, final audit, SHA-bound listening,
a second final audit and verified teardown in one proof-owned process.

### HW05 audit and listening boundary

HW05 cannot be inferred from child receipts. The backend first performs fresh
final readiness reconciliation, captures that exact console-change fence, and
then reads every address in the committed surface ledger twice in opposite
orders around a combined live-audio check, Main-unmuted readback and fresh route
check. Before and after every console read and every awaited operation it
revalidates connection/transport/emergency generations, authority identity,
the same readiness-fence object, current clock-sensitive `show_mix_ready`, no
global blocker and no in-progress Apply. A last-read change or freshness expiry
therefore blocks instead of returning a stale verified hash.

The canonical verified-state SHA-256 is shown to the operator only after that
audit. Listening confirmation is exact-SHA-bound and expires after five
minutes. The runner then repeats the complete final audit; the run can close
only when both state hashes, ledger hashes and run identities match and the
proof-owned backend terminates cleanly. Any interleaved backend
`readiness_invalidated` event is a permanent session-level latch before or
after a terminal and prevents closure. Otherwise the durable outcome is
`soundcheck_incomplete`, never an optimistic success.

## 4. Why this won

It preserves all existing global rails while giving the attended operator a
finite content-bound path to the requested result. It reuses the hardened queue
claim/result/WAL lifecycle, proves route origin inside one attended process,
revokes that authority on restart, distinguishes
validation from permanent change and prevents reset from producing a silently
unformed mix.

## 5. Rejected alternatives

- Standing allowlist widening outlives the reviewed run.
- Direct runner OSC bypasses the backend writer, WAL and emergency generation.
- JSON+SHA or a `producer` field is self-authored evidence.
- A generic query/write endpoint lets clients choose the safety boundary.
- Broad reset functions mutate preserved surfaces and are not atomic.
- Automatic promotion from one proof conflates evidence with policy authority.

## 6. Implementation plan

1. Implement fixed route artifact/handler/runner and same-process revocation.
2. Implement capability store, typed grants and verified proof terminal.
3. Implement fixed maintenance reset and preserved-surface verification.
4. Implement enable-last processor lane and exact child commit capability.
5. Run adversarial, frontend/no-WING and full-suite gates; independent review.
6. Commit clean source, then run the attended route→proof→reset→soundcheck
   hardware checkpoint and record immutable evidence.

## 7. Test plan

- Strict request schemas, authentication and packaged-offline blocks.
- Exact fixed address order; two physical passes; route/identity drift.
- Capture freshness/identity/progression/stereo signal/headroom boundaries.
- Unique CoreAudio/USB VID/PID/location/serial binding, exact three-digit USB
  interface suffix, stream-object swap and fresh hardware drift revocation.
- Persistent same-WebSocket route session, interleaved/duplicate terminals,
  independent semantic artifact verification and cancellation-safe teardown.
- CP05 pending/stale/full-tree bindings and raw M110 start/scan/alternate-mixer
  ingress block before any OSC, USB capture or discovery constructor.
- Audit-artifact path/symlink/size/tamper/fsync tests and proof revocation on
  restart/reconnect/TTL/source drift.
- Token replay, wrong owner, restart, reconnect, expiry, concurrency and every
  binding mismatch.
- Durable result capacity before consume, consume before claim, claim-race
  terminalization and proposal-scoped revocation on accept/dismiss/expiry.
- Generic proof switches and arbitrary plans block before pre-read/send.
- Cancellation and failure at every coordinator/WAL/readback/rollback await.
- `proof_passed` remains distinct in queue, result store, wire and readiness.
- Reset writes only the fixed set, proves equality of every finite-manifest
  surface, proves no outside mutation send, and reports whether WING delivered
  any outside transition event without treating event silence as equality.
- Processor enable order and complete-block binding.
- Reset, processor and decomposed soundcheck commits each require byte-identical
  canonical bundle bytes/digest from their own recent restored proof; no
  subset, superset, reorder or sibling inference.
- Normal allowlist and registry remain byte-for-byte unchanged.
- Standard full suite, no-WING gate, `git diff --check` and independent review.

## 8. Risks and rollback

The largest risks are token confusion, expiry after send, audit corruption,
processor enable transients and long reset rollback. Separate frozen grant
classes, exact digest revalidation, latest-start expiry, recovery-shielded
rollback, fixed roots/budgets and enable-last/disable-first ordering mitigate
them.

Source rollback removes M110 handlers and runners while leaving ordinary
proposal analysis available. Never restore direct runner writes, digestless
apply, legacy broad reset or self-authored route proof. Any ambiguous live
state latches emergency/manual review and stops the hardware checkpoint.

## Council record

Codex and three independent read-only reviewers agreed on fixed route
provenance, one-shot authority, restored-proof semantics, DSP-only reset and a
separate processor-enable lane. The main disagreement was a new proof execute
message versus canonical Apply. Canonical Apply won because it preserves one
external write entrypoint; private typed grants and adversarial no-mint tests
are mandatory conditions of that choice.

The 2026-07-14 independent patch review found no P0 defect. It identified and
closed two reset-path issues before CP05: physical sweep timeouts now remain
strict blocked protocol terminals, and commit recapture cannot consume the
entire restored-proof receipt window. The review also required the strict M110
manual WAL module to be included in both runtime identity and the exact CP05
critical-path manifest.

The final independent P1 cycle on 2026-07-14 first reproduced late readiness
invalidation on the fourth/last ledger read, expiry during the audio await and
invalidation while hashing the state. All three now fail closed at the exact
server audit pass. The same review then found an adjacent runner defect: an
interleaved `auto_soundcheck_status/readiness_invalidated` event could be
consumed before terminal or during terminal grace without failing the session.
A persistent session latch now makes both orders fatal and rejects every later
terminal before another WebSocket read. The reviewer verified benign progress
events remain accepted, reran all 97 runner tests and reported no remaining
P0/P1. This is software evidence only and does not close CP05 or hardware.
