# Soundcheck plan decomposition and honest readiness — 2026-07-12

## Verdict

The monolithic-plan defect is closed in the local execution kernel. A local
operation blocker no longer suppresses unrelated safe corrections, while a
global hazard still blocks the complete run before another console send.

Scores use separate scopes:

- local staged soundcheck execution kernel: **96/100**;
- strict real-live soundcheck readiness: **88/100**;
- full-project local engineering readiness: **95/100**;
- strict public-release readiness remains **88/100**.

Only the first and third targets are locally achieved. Normal supervised live
authority remains exactly zero, so none of these scores grants permission to
write to a WING.

## Defect reproduced

The archived M006 run contained 51 logical operations: 1 trim, 10 HPF, 7 EQ,
5 gate, 2 compressor, 14 pan and 12 fader operations. Physical values changed
for 39 operations; 12 were already at the requested state.

Under the pre-fix current policy, 36 operations were locally admissible and 15
had local residuals. Because all normal work shared one prevalidated rollback
boundary, one residual rejected the entire transaction: the counterfactual
result was **0/51**, not 36/51.

## Implemented contract

- The parent manifest is revisioned, journaled and content-bound. Every normal
  correction is a semantic child transaction unless an explicitly validated
  coupled fader alternative must commit atomically.
- Disconnect, stale readback, emergency stop, writer conflict, missing rollback
  truth and failed Main 47/48 evidence remain global blockers.
- A local transition problem remains attached to its own child. Sibling
  children stay reviewable and executable if their dependencies remain fresh.
- Large safe moves become bounded steps. The only automatic musical
  substitution currently trusted is a complete selected-source fader-vector
  down-translation that preserves the final relative balance and never raises
  a fader.
- EQ boosts, compressor makeup increases, gain-up and send-up remain operator
  review; the system does not invent an unproven alternative.
- Selected-source fader plans bind fresh analysis to fader, mute, post-fader
  Main send, input routing and DCA topology/readback. Missing, changed or
  expired evidence revokes write authority.
- The same full source-path readback is required for all-hold plans and for
  later tonal/group children that depend on an earlier fader intent; zero
  physical operations is not treated as proof that the source is audible.
- Every selected-source, bus or master child carries the exact original
  source-analysis deadline. The coordinator rechecks that same deadline during
  execution instead of silently creating a fresh time window.
- The coordinator rechecks approval, authority and compare-and-swap state
  between members, after audio verification and before commit. Any possible
  console send makes the proposal terminal until physical reconciliation.
- Queue recovery accepts `no_console_send_proven` only as explicit evidence;
  missing WAL/events never imply that no send occurred. Verified rollback is a
  terminal restored-physical-state result that requires fresh analysis.
- Stage-local reanalysis uses a candidate/commit/discard lifecycle. Candidate
  revisions, analysis state and hypothetical observation writes stay isolated;
  server journal/queue publication and engine acknowledgement share the same
  derivation, monitor and console-writer fences. Failure or cancellation leaves
  engine/server at revision `N`, so retry produces exactly `N+1`.
- `ObservationMixerClient` delegates only an explicit read allowlist. Unknown
  callables and every write-like method are intercepted, while revision-local
  shadow getters remain symmetric with hypothetical setters.
- Parent-journal registration occurs only after applying-child conflicts are
  rejected. Journal recovery binds each child outcome to the exact manifest
  digest, transaction and operation identity.
- Final reconciliation repeats physical target and audible-path reads. A mute,
  route/DCA loss or target drift clears per-operation checkpoints and prevents
  later dependent stages from applying.
- `safe_core_ready`, `show_mix_ready` and `soundcheck_incomplete` are evidence
  states, independent from engine lifecycle or proposal derivation completion.

## Stage model

1. `foundation_level`: gain/trim, HPF and source fader balance.
2. `source_control`: gate/expander where justified.
3. `phase_alignment_review`: polarity and delay.
4. `tonal_dynamics`: EQ and compression.
5. `group_balance`: bus, group, DCA and matrix balance.
6. `spatial_fx`: pan, sends and FX-related work.
7. `destructive_high_risk`: mute, master, routing, scenes and presets under
   separate approval and proof gates.

## Adversarial verification

The regression suite now covers:

- one local blocker without parent-wide poisoning;
- mixed already-satisfied/apply members in one atomic fader group;
- drift of an earlier group member before a later send;
- approval or source-analysis expiry during execution;
- missing or extended source-analysis deadline identity;
- authority revocation after audio and before commit;
- no-send CAS abort without overwriting an operator move during rollback;
- selected-source mute, Main, pre/post, routing, DCA and fader drift;
- all-hold and non-fader child source-path drift;
- non-fader source-analysis deadline expiry;
- path loss after apply revoking `safe_core_ready` and the per-operation stage
  checkpoint;
- candidate derivation failure, import/journal retry, concurrent stop and
  repeated cancellation without revision drift or pending-candidate leakage;
- immutable accepted-bundle identity across initial import, reanalysis and
  idempotent re-import;
- default-deny observation calls plus shadow-consistent pan, mute, HPF,
  dynamics, send, bus, DCA and master reads;
- immutable nested targets/results at queue boundaries;
- canonical manifest content-digest rejection on first trusted import;
- terminal proposal/idempotency behavior whenever a console send may have
  occurred.

All tests in this report are offline/no-WING unless a historical artifact is
explicitly named. The real console was not contacted and no live write was
performed during this repair pass.

Final current-tree validation: `6364 passed, 1 skipped` in the complete backend
suite; `135` WING MVP truth rails; `190` frontend unit/contracts; production
frontend build; and `13` Chromium E2E tests. Independent final adversarial
review found no remaining actionable P0/P1/P2 in this implementation scope.

## Remaining strict-live blockers

- normal supervised live allowlist count: `0`;
- current-tree physical staged-child proof: absent;
- real-WING `safe_core_ready` proof: absent;
- real-WING `show_mix_ready` proof: absent;
- actual WING latency budget for the two-pass selected-source path sweep:
  unproven;
- constructive substitutions other than down-only fader-vector translation:
  operator-review-only;
- current-source M005 provenance rebind and M006 technical promotion;
- clean signed/notarized package and operator-present room/PA validation.

These are external/promotion gates, not reasons to widen safety policy in code.

## Score rationale

| Local execution-kernel area | Score |
| --- | ---: |
| Safety authority, CAS, rollback and emergency behavior | 25/25 |
| Decomposition, dependency handling and safe recomputation | 24/25 |
| Readiness truth and final reconciliation | 20/20 |
| Lifecycle, journal recovery and idempotency | 15/15 |
| Verification and physical reproducibility | 12/15 |
| **Total** | **96/100** |

The four missing points are deliberate: only one constructive substitute is
trusted, and the current staged path has not yet been timed and completed on
the physical console. Strict live readiness is therefore 88/100, not 95/100.

## Authority statement

`live_authority_changed=false`. The registry, safety matrix and effective
operator-mode live-write lists remain empty. M006 authorization is recorded as
owner intent for a future controlled proof; it is not standing authority.
