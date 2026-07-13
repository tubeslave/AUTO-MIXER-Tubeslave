# Safety critique — M109-CP02

## P0 findings

- Full `to_dict()` and deep copy occur before slicing, so both CPU/RAM and wire
  payload are unbounded. MixingAgent data is unbounded too.
- Create, detail, preflight and apply responses can also return complete
  proposals; fixing only the list leaves alternate oversized responses.
- Digest checks only in the UI/direct handler are bypassable by linked,
  transaction and sequence executors. Validation must precede claim,
  idempotency and every write-capable path.
- The digest must cover the complete supervised transaction plan; summary
  arrays and nested plans cannot be silently truncated.

## P1 findings

- MixingAgent IDs such as `pending-{index}` are unstable and can collide.
- Cursor identity must bind ordering/filter/snapshot ownership and expiry.
- Close 1009 currently risks reconnect/mock loops and false live truth.

## Required controls

1. Build lightweight metadata directly; never compact after full serialization.
2. Use the actual transport serializer and enforce 131,072 UTF-8 bytes for the
   whole response.
3. Keep raw/result/blobs and unbounded arrays out of list rows.
4. Order unresolved global hazards, other blocked rows, high/critical rows,
   then ordinary rows deterministically.
5. Return exact totals and counts-only overflow; if pinned metadata itself
   overflows, return bounded degraded state with apply UI disabled.
6. Use stable source-qualified analysis-only identity for MixingAgent rows.
7. Bind cursors to frozen, expiring connection-owned snapshots.
8. Return full review content through bounded verified chunks.
9. Digest every apply-authoritative field but exclude dynamic status/result.
10. On binding failure prove no mutation, reservation, claim or console send.
11. Apply the rule to every internal and external route without a trusted
    bypass.
12. Pending proposals are never evicted; terminal detail eviction is typed.
13. Close 1009 is terminal/manual-retry and never activates mocks.

## Adversarial test matrix

Cover 1 MiB raw/result, worst-case Unicode, old critical plus new low rows,
pinned overflow, MixingAgent collision/reindex, mutation between pages, nested
plan mutation, status-only digest stability, missing/mismatch on every apply
path, chunk assembly, close 1009/no mock, and a spy proving selection never
invokes full `to_dict()`.
