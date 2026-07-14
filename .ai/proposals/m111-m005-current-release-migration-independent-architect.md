# Independent architect proposal: M111

This proposal records the read-only opposing review used after Kimi CLI failed
before analysis with an external membership error.

## 1. Short thesis

Conditionally allow a separate migration route, but never relabel the 11 July
artifact as native current provenance. A migration may satisfy the narrow M006
input gate only when immutable historical hardware evidence, a proof-owned
out-of-process current-code replay and a fresh owner console identity
attestation all pass independently.

## 2. What I understood

The historical artifact proves the old M005 behavioral contract, while the
current contract requires facts that did not exist at run time. The owner has
closed M005 behavior and authorized narrow M006, but has not supplied a fresh
serial-bound console identity challenge. Normal live authority remains zero.

## 3. Proposed solution

- Keep `m005_provenance_ready=false` and native
  `m005_current_release_accepted=false` for migrated evidence.
- Add route-specific `m005_migration_ready` and aggregate
  `m005_m006_input_accepted` fields.
- Hard-bind the accepted historical summary/events paths and their exact
  SHA-256 values because those files are ignored and untracked.
- Launch a sealed child process which freshly imports the canonical current
  server/handler code and is bound to clean HEAD/tree, runtime-component
  hashes, PID/start/nonce/challenge and loaded module identities.
- In that child, execute canonical `create_operator_proposal` then
  `apply_operator_proposal` for stale fader and gain-down cases. Require
  `operator_proposal_apply_blocked`, `no_console_send_proven=true` and zero
  calls to `_apply_manual_console_write`, fader/gain setters or raw send paths.
- Require a separate owner attestation with actual WING Rack serial, firmware
  3.0.5, `same_console_as_historical_m005=true`, reviewer and review time. Mark
  it identity-only: it proves neither historical runtime nor live authority.

## 4. Likely files to touch

The migration runner/verifier, a sealed child replay harness, M006 packet and
decision verifiers, focused tests, and a new ADR. Existing historical artifacts
must remain byte-identical.

## 5. Alternatives considered

- Retrospective native provenance: no-go because prestart/runtime facts are
  unrecoverable.
- In-process handler replay: too weak; the same process can self-affirm mocks
  below or beside the real gate.
- Unit tests plus owner statement only: insufficient source/process binding.
- Repeat the physical pull: strongest direct route, but not authorized by the
  owner and not required if the migration contract is fully met.

## 6. Risks

The migration path can silently become a provenance waiver, a replay harness
can test the wrong layer, and an attestation can be misdescribed as a hardware
query. Route-specific flags, exact identities and evidence-class labels must be
mandatory and mutation-tested.

## 7. Test plan

Mutate each historical hash/path/event, source/runtime identity, child
challenge/PID/module identity, replay terminal and writer-call counter. Mutate
each console model/firmware/serial/same-console/reviewer field. Verify native
and migration routes independently and prove that every broad family and all
live authority flags remain blocked.

## 8. Where Codex may disagree

Codex's initial proposal allowed an in-process current-handler replay as a
possible implementation. That is not sufficient. The accepted design should
require a fresh proof-owned child process and should avoid reusing native M005
success flags for migration.
