# Codex proposal — M110 attended WING reset and soundcheck

## 1. Short thesis

Build one backend-owned attended capability system around the existing proposal
queue, console writer scope, coordinator and WAL. Use it for fixed route proof,
restore-only family proofs, DSP-only reset, processor enable and exact staged
soundcheck commits without widening the normal allowlist.

## 2. What I understood

The staged planner defect is already closed, but live authority is deliberately
zero. The remaining problem is not to remove the rail; it is to create a narrow
one-time attended lane whose authority is exactly the content the operator
reviewed. Current route evidence has no origin proof, `proof_run` is rejected at
the server boundary, and reset would disable processors the current soundcheck
cannot safely enable again.

## 3. Proposed solution

### Route proof

Add `capture_wing_master_route_proof` as an authenticated fixed endpoint. It
accepts no addresses or expected values, holds the global console scope and
audio-capture lock, reads serial/firmware and USB 47/48 route twice, brackets a
fresh 48-channel/48 kHz capture sample, requires signal on both channels and
`sample_peak + 3 dB <= -1 dBTP`, then atomically writes a content-addressed
audit artifact. Live authority is held only as a private object in the issuing
process. A restart revokes it unconditionally and requires the attended route
proof to run again; path/SHA is never a cross-process authority claim.

The route-proof owner then uses one exact authenticated arm message to enter a
same-process supervised capability-only state without reconnecting. Ordinary
Apply remains blocked; disconnect, restart, emergency or route expiry disarms
and revokes tokens.

The attended runner sends an exact canonical prestart source manifest and a
fresh 32-byte secret through a one-shot inherited pipe before server
construction. The backend independently recomputes the manifest. Route capture,
arm and issuance carry fresh-nonce HMACs; public environment metadata, files and
artifact SHA values never authenticate the launch session.

### One-shot authority

Add a bounded `SupervisedCapabilityStore`. The backend stores token digests,
never plaintext, and binds each record to runtime nonce/source manifest/HEAD,
connection generation, console serial/firmware, route proof digest, emergency
generation, proposal ID/review digest, canonical plan digest, approval,
idempotency, scope and deadline. Atomic consume burns the token before any
pre-read. Recovery writes remain cancellation-shielded after consumption.

Capability issuance is a separate authenticated v1 request available only when
runtime launch binding says `m110_attended_runner`, the request owner matches
the prepared proposal owner, exact review/plan digests are echoed, and a
capability-specific literal confirmation is present. One unconsumed token per
owner, eight globally, 30-second latest-start TTL; disconnect/emergency/new
route proof explicitly revokes all unconsumed tokens.

Use the existing authenticated `apply_operator_proposal` message as the single
external write entrypoint. A capability token is not a `proof_run` switch: only
an internally created typed grant can select a capability lane. Generic JSON
proof flags, restore flags, addresses and targets remain rejected.

Capability kinds are distinct:

- `restore_only_proof`: one exact canonical operation bundle; forced
  restoration;
- `maintenance_dsp_reset_commit_once`: exact fixed bypass set;
- `processor_enable_commit_once`: exact block parameters plus enable-last;
- `soundcheck_child_commit_once`: one reviewed staged child plan after matching
  proof evidence.

None changes the standing registry or normal allowlist.

### Terminal semantics

Add a shared verified-proof predicate. Successful restore-only proof is
`proof_passed`, WAL `proof_restored`, physical state restored, accepted false,
and never counted as a mix apply. Permanent reset/processor/soundcheck commits
retain ordinary verified `applied` semantics.

### Reset and enable

Reset is server-generated for channels 1–16 and only writes EQ bypass, HPF
bypass, gate bypass and compressor bypass. It captures all 64 preimages,
prepares one WAL record, applies in bounded order, double-reads every value and
rolls the complete set back on failure. A finite fresh physical manifest covers
critical channel gain/balance/input-route and Main/USB identity addresses.
For sends/FX, buses, scenes and names, the receipt claims only exact no-write
attempt and reports outside-address transition events actually delivered by
WING; event silence is not unmeasured equality.

After reset, one atomic processor capability owns the complete parameter block
and enable state. It snapshots while bypassed, writes and verifies parameters,
enables last, then performs audio verification. On failure it disables first,
reverse-restores parameters and restores the original enable state. No earlier
permanent child is allowed to commit those processor parameters.

Every permanent processor/reset/soundcheck commit requires a recent
same-process `proof_passed` whose canonical operation-bundle bytes and digest
are identical after removing only transaction/idempotency and proof/commit
labels. Fixed reset and atomic processor parameter-plus-enable are the only
composite bundle kinds. Staged children are decomposed to one logical operation
before review, proof and commit. A family/write-kind proof never authorizes a
sibling channel, address, direction or target.

### Runners

Route and family proof runners become WebSocket-only, plan-only by default,
launch a proof-owned backend from clean committed source, establish the one-shot
prestart manifest/secret pipe, pass bearer tokens via a separate FD, verify
challenged runtime identity and guarantee child teardown. They never
import `WingClient`, open UDP or send OSC directly.

## 4. Likely files

- `backend/master_capture_route_proof.py`
- `backend/supervised_proof_capability.py`
- `backend/supervised_family_proof_plan.py`
- `backend/maintenance_dsp_reset.py`
- `backend/server.py`
- `backend/handlers/connection_handlers.py`
- `backend/handlers/product_state_handlers.py`
- queue/result/wire/coordinator/WING adapter modules
- `backend/runtime_identity.py`
- route/family proof runners and dedicated tests/runbooks

## 5. Alternatives considered

- A larger standing allowlist is rejected because it outlives the reviewed run.
- Direct runner OSC is rejected because it cannot share writer/WAL/emergency
  ownership.
- JSON+SHA with a producer label is self-declared and insufficient.
- A separate generic proof execution endpoint duplicates the canonical apply
  evidence lifecycle; a typed internal grant behind canonical Apply is smaller.
- Broad legacy reset functions are non-atomic and mutate preserved surfaces.

## 6. Risks

The main risks are capability confusion, expiry during rollback, enable-state
transients, audit corruption and a long reset transaction. Distinct enums,
exact plan digests, latest-start deadlines, recovery-shielded grants,
enable-last/disable-first ordering, fixed roots and hard address/time budgets
contain those risks.

## 7. Test plan

Adversarially test malformed requests, unknown fields, untrusted addresses,
wrong runtime/console/route/proposal/plan, replay/concurrency/expiry/reconnect,
WAL failure, cancellation at every await, headroom and signal failures, partial
reset, preserved-surface drift, enable-state transients, truthful terminal
receipts and zero normal-authority changes. Then run frontend contracts, the
no-WING gate, full backend suite and independent review.

## 8. Where the other agent may disagree

The likely disagreement is using canonical `apply_operator_proposal` rather
than a new execution message. I prefer one external write entrypoint, provided
the token can only become a private typed grant after atomic server validation
and the ordinary path continues rejecting every proof/maintenance switch.
