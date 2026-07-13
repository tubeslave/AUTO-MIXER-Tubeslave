# Independent critique of the Codex M110 proposal

## 1. Strongest part

The Codex proposal gives the whole attended workflow one coherent backend-owned
boundary and correctly keeps route proof, family proof, reset, enable and final
commit as different semantic operations.

## 2. Possible mistakes or blind spots

- Reusing Apply is safe only if client `proof_run`, `restore_after_verify`,
  capability kind and arbitrary plan fields never select the lane.
- A reset that checks only the four written bypass values does not prove the
  preserve set; a finite fresh physical manifest plus no-outside-send evidence
  and reporting of transition events actually delivered by WING are necessary.
  Event silence cannot prove unmeasured categories equal.
- `soundcheck_child_commit_once` must require identical canonical operation-
  bundle bytes/digest from a recent same-process proof, not merely family/
  write-kind. Composite authority is limited to fixed reset and atomic
  processor bundles; staged children are decomposed into single operations.
- Token values must be removed before tracing, error reporting and wire
  projection.
- Launch provenance must arrive as an independently recomputed exact manifest
  plus random secret over an inherited one-shot FD. Self-declared environment
  nonce/SHA metadata cannot authenticate route capture, arm or issuance.

## 3. Where it is too complex, fragile, or costly

One generic capability record can become an unsafe union type. Distinct frozen
record classes and exact terminal predicates are preferable even if they share
storage mechanics.

## 4. What should be kept

Keep fixed route reads, durable issuance origin, canonical Apply evidence
lifecycle, bounded reset, explicit processor enable and staged local progress.

## 5. Revised recommendation

Proceed with canonical Apply only after tests prove ordinary JSON cannot mint a
grant. Use separate immutable grant classes for proof, reset, enable and child
commit; each validates its exact bundle bytes/digest again immediately before
pre-read.
