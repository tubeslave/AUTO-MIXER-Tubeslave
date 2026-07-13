# Independent proposal — M110 proof authority and reset boundary

## 1. Short thesis

Create a separate backend-issued one-time proof authority and distinct terminal
semantics. Never widen normal live kinds, never accept client-authored proof
plans, and never unblock the legacy direct-WING runner.

## 2. What I understood

The coordinator already knows how to restore after verify and emit
`proof_passed`, but the server rejects proof mode and later hardcodes proof
truth false. Queue, durable result and wire predicates recognize only normal
`applied`. Independently, reset disables processing that cannot currently be
re-enabled under a separate approval.

## 3. Proposed solution

- Add a fixed authenticated route endpoint and process-private proof authority;
  reject self-authored JSON even with a correct SHA and revoke on restart.
- Add prepare, issue and execute phases for one exact canonical operation-bundle
  proof. Only the fixed reset and atomic processor bundles are composite;
  soundcheck children are decomposed to one operation.
- Store only a token digest; bind it to socket owner, runtime, console,
  connection generation, route proof, proposal/review/plan digests, exact
  addresses and deadline.
- Burn the token before claim. Force `restore_after_verify`; shield rollback
  from caller cancellation.
- Make `proof_passed` a first-class safely restored terminal with WAL
  `proof_restored`; never map it to `applied` or use it to advance musical
  readiness.
- Keep reset separate from proof and soundcheck. Build its exact fixed scope
  server-side, require a complete preimage and verify restoration on failure.
- Use a second `processor_enable_commit_once` capability that atomically owns
  complete conservative parameters and permanent enable. It configures while
  bypassed, enables last, and disables first before reverse restoration.
- Require exact canonical operation-bundle byte/digest equality between a
  recent restored proof and every permanent reset/processor/soundcheck commit.
- Establish the attended launch with an independently recomputed exact source
  manifest and random secret over a one-shot inherited FD, then require HMACs
  for route capture, arm and issuance.

The independent preference is a dedicated `execute_supervised_family_proof`
message rather than overloading normal Apply. If canonical Apply is retained,
the public request must still contain no boolean proof switch, and only a
private non-serializable grant may activate proof behavior.

## 4. Likely files

- new capability, proof-plan, route-proof and maintenance-reset modules;
- server, connection and product handlers;
- transaction coordinator, queue, result store, wire contract and runtime
  identity;
- a WebSocket-only family proof runner with the direct `WingClient` path
  deleted;
- dedicated adversarial tests.

## 5. Alternatives considered

- Reuse M005 process binding as write authority: rejected; it is read-only.
- Let any authenticated browser mint proof authority: rejected; attended runner
  provenance is distinct.
- One token for a family or whole soundcheck: rejected; one token binds one
  logical operation/child transaction.
- Auto-promote a family after proof: rejected.

## 6. Risks

Proof could accidentally become a normal apply success, token expiry could
interrupt restoration, or processor enable could activate unknown settings.
Separate predicates, latest-start expiry, recovery grants, complete-block
binding and enable-last ordering are mandatory.

## 7. Test plan

Test token replay/wrong owner/restart/reconnect/expiry/concurrent consumption;
proposal and console drift; generic proof flags; arbitrary/multi-family plans;
apply/readback/audio/rollback and WAL failure; cancellation after each send;
token leakage; queue/result/wire distinction; reset preserve set; enable-state
ordering; and unchanged registry/normal allowlist.

## 8. Where the other agent may disagree

The independent design prefers a distinct proof-execute message and a very
short 15–30 second capability TTL. It also recommends a separate permanent
processor-enable capability even if all other soundcheck parameters use a
broader attended child-commit token.

## Provenance

This is an isolated read-only opposing review. No backend was started and WING
was not contacted.
