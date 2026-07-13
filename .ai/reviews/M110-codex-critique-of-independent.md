# Codex critique of the independent M110 proposal

## 1. Strongest part

The independent proposal correctly separates restored proof from permanent
mix apply and identifies processor enable as a larger operation than a boolean
toggle because unknown retained parameters may become audible.

## 2. Possible mistakes or blind spots

- A dedicated proof-execute endpoint would duplicate claim, result-reservation,
  digest, wire and terminal evidence code already hardened in Decision Queue v2.
- A 15-second hard execution expiry could expire after the first send and must
  not revoke rollback authority.
- Restore-only proof and permanent soundcheck commit need a defined evidence
  linkage; the permanent canonical bundle bytes/digest must exactly equal the
  restored proof. Fixed reset and processor lanes may be composite, while
  staged children must be decomposed; no bundle may authorize family siblings.

## 3. Where it is too complex, fragile, or costly

Parallel external write entrypoints are harder to audit than one authenticated
Apply boundary. The server can keep public proof switches forbidden while
letting atomic token validation construct a private grant internally.

## 4. What should be kept

Keep distinct capability kinds, digest-only token storage, exact plan binding,
forced restoration, `proof_restored`, enable-last/disable-first ordering and no
automatic promotion.

## 5. Revised recommendation

Use canonical `apply_operator_proposal` for all permanent/proof executions, but
require a private typed grant for non-normal lanes. Capability expiry is a
latest-start deadline; once consumed, only emergency/disconnect rules may stop
forward work, while rollback/recovery authority persists until terminal proof.
