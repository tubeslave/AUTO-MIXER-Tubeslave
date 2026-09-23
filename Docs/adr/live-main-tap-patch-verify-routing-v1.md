# ADR: Live Main Tap PATCH_VERIFY Routing Proof v1

Date: 2026-09-24
Status: accepted for R3 migration

## Context

`PostConsoleMainTapEvidenceProvider` can measure Main only if its reserved USB capture slots really carry the declared post-console WING Main source. A configured slot number alone is not sufficient proof.

Legacy `WingClient.get_output_routing()` is not authoritative enough for this gate: it issues queries, sleeps for a fixed interval, then reads `WingClient.state`. That shared cache can contain an older route if a fresh reply is delayed or missing. The useful legacy pieces are the validated WING address layout (`/io/out/{group}/{zero_based_slot}/grp` and `/in`) and callback-driven query transport.

## Decision

Add a read-only PATCH_VERIFY routing proof under `backend/live_runtime`.

1. `WingWriteAdapter.read_output_route()` queries `/grp` and `/in` and requires fresh inbound callbacks for both values. It never uses `WingClient.state`.
2. `MainTapPatchContract` explicitly maps every reserved `PostConsoleMainTap` USB slot to a declared WING `MAIN` source channel. The route set must exactly cover the tap slots, with no hidden/default route.
3. `MainTapPatchVerifier` reads each configured USB output route and fails closed on timeout, transport failure, malformed readback, wrong source group or wrong source channel.
4. PATCH_VERIFY is read-only. This verifier does not repair routing and does not use BENCH_TEST write bypasses.
5. A successful routing proof is necessary but not sufficient to authorize autonomous Main decisions. Independent physical Main level/tap coherence remains a later HIL gate.

## Migration classification

- **KEEP_CORE:** raw WING OSC query/callback transport and the validated `/io/out/.../grp|in` address semantics.
- **ADAPT:** output-routing readback becomes fresh callback-scoped `live_runtime` evidence.
- **ARCHIVE:** legacy cache-backed `get_output_routing()` as an authority for safety gates once runtime references are severed; its historical implementation may remain for compatibility until cutover proof is complete.
- **DELETE_AFTER_PROOF:** none in this pass.

## Safety consequences

A stale local routing cache can no longer satisfy Main-tap PATCH_VERIFY. Missing physical readback is a failed proof, not permission to proceed. No routing mutation occurs during verification, so this change is safe to exercise outside a declared BENCH_TEST session.

## Tests

Focused tests cover exact stereo routes, mismatched routes, missing fresh readback, transport query failure, exact reserved-slot coverage and rejection of non-Main route contracts. The focused live CI includes the new PATCH_VERIFY test file.

## Remaining gate

Add independent Main-level coherence evidence or an equivalent physical HIL proof, then compose both proofs into startup/FSM `PATCH_VERIFY` before Main Director actions are allowed to leave proposal-only behavior.
