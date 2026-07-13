# Codex fallback opposing critique

This is an independent opposing pass by Codex because Kimi CLI was unavailable.
It is not represented as Kimi output.

## 1. Strongest part of the proposal

The proposal correctly refuses to widen `WING_DEPLOYMENT_LIVE_WRITE_KINDS`
before a canonical transaction path exists. Reusing `apply_operator_proposal`
as the sole live entrypoint preserves the system's strongest existing proof:
approval, idempotency, physical readback, rollback and emergency audit.

## 2. Possible mistakes or blind spots

- A completely generic parameter registry risks becoming a hidden raw-OSC API.
  Complex WING models need family adapters that understand model selection,
  address variants, quantization and semantic readback.
- Treating Main LR audio recheck as mandatory for every parameter is wrong for
  pre-trim USB taps, routing changes that target monitors, and changes made
  while the source is silent. The contract needs explicit `required`,
  `deferred`, and `not_observable_on_current_tap` outcomes.
- Existing linked-proposal rollback calls `_apply_manual_console_write`
  directly and does not use the transaction-bound rollback id. That path must
  be replaced, not layered under the new kernel.
- Scene/preset and routing are not ordinary soundcheck correction parameters.
  Supporting them in the same API is useful, but they require a destructive
  scope preview and separate approval, never a normal channel-strip batch.
- The proposed file list is broad. The first implementation checkpoint should
  land the core contract and compatibility adapter before touching frontend or
  every DSP producer.

## 3. Where the solution is too complex, fragile, or costly

One universal operation schema with arbitrary method names and argument lists
would be hard to statically audit and easy to misuse. It would also make
readback tolerances and rollback ordering data-driven in ways that reviewers
cannot easily reason about.

## 4. What should be kept

- Single Product Layer live entrypoint.
- Absolute targets and immutable pre-read snapshots.
- Reverse rollback and emergency generation checks before every send.
- Per-family lifecycle and hardware-proof activation.
- High-risk surfaces separated from normal DSP batches.
- No live promotion from offline tests alone.

## 5. Revised recommendation

Build a small generic transaction coordinator around explicit, code-reviewed
family adapters. Each adapter implements:

- `validate_request`;
- `snapshot_physical_state`;
- `apply`;
- `verify_physical_state`;
- `rollback_compare_and_swap`;
- `audio_recheck_policy`.

The coordinator owns approval, idempotency, emergency generation, sequencing,
artifact recording and rollback orchestration. Adapters own WING semantics.

Checkpoint 1 should add the coordinator, registry metadata and a compatibility
adapter for existing fader/gain without widening live kinds. Checkpoint 2 adds
one family adapter at a time with offline proof, then hardware proof. This
delivers the complete architecture without creating a generic raw-write escape
hatch.
