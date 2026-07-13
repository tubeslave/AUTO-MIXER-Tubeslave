# Independent critique of the Codex CP02 proposal

## 1. Strongest part

The proposal correctly places digest validation at the existing canonical
apply gate and rejects a request-version escape hatch.

## 2. Possible mistakes or blind spots

- The first draft treated large `raw` as non-authoritative metadata, but
  `raw.supervised_transaction_plan` can be the exact write plan.
- It did not specify how to avoid full `to_dict()` work before compacting.
- Cursor ownership, TTL, frozen ordering and byte-bounded snapshot retention
  were underspecified.
- It named only the direct frontend apply call, while linked/sequence/executor
  paths can also construct apply requests.

## 3. Where it is too fragile

Using a list digest without an explicitly verified detail receipt would create
the appearance of review binding without proving that full content was shown.
Returning detail in one frame would simply move the oversized-frame defect.

## 4. What should be kept

Keep strict digest enforcement, server-side proposal authority, exact transport
measurement and a read-only compatibility shim.

## 5. Revised recommendation

Freeze the complete authoritative projection at creation, chunk it at 64 KiB,
bind every apply path to its digest, and define a bounded frozen cursor store.
Pre-v2 rows must be regenerated rather than applied through a legacy path.
