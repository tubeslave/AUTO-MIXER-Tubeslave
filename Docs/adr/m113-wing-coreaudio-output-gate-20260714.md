# ADR — M113 WING CoreAudio Output Gate

## 1. Context

M001 and M110 require one physical WING USB/CoreAudio device operating as
48 input, 48 output at 48 kHz. The hardware verifier checked only the input
half, so a 48x0 row could verify and produce an authority-grade binding.

## 2. Options considered

Leave output validation to HW01, accept any positive or at-least-48 output
count, add exact validation only in the collector, introduce schema v2, or
enforce exact integer 48x48 in both collection and independent artifact
verification while retaining v1.

## 3. Decision

Require non-boolean integer 48 for both CoreAudio input and output counts.
Include both in the canonical binding and require the exact 48x48 shape in the
attended runner's independent evidence verifier. Keep the existing aggregate
blocked reason with per-check diagnostics. Retain schema v1 because M110-HW01
has never started and the repository contains no accepted physical v1 route
artifact; this is a correction of the original required contract, not a
migration of accepted evidence.

## 4. Why this won

It fails before capture authority is issued, makes the output prerequisite
cryptographically visible, rejects coercion and self-consistent forged
input-only artifacts, and changes no unrelated transport or live capability.

## 5. Rejected alternatives

Late or approximate checks permit unsupported modes. Collector-only validation
leaves the independent verifier stale. Schema v2 would add migration machinery
without a physical accepted v1 consumer.

## 6. Implementation plan

Add one strict channel-count helper, input/output predicates and the output
binding field; update the runner semantic shape and fixture; add adversarial
and forged-evidence tests; correct the M110 ADR wording; then synchronize the
offline validation handoff. Commit non-marker work before a separate CP05
marker rebind.

## 7. Test plan

Run focused hardware-identity, master-route and attended-runner tests; WING
safety/truth rails; the standard full suite; no-WING detached verification;
JSON/JSONL, memory and diff checks; then CP05 plus rails on the final committed
tree.

## 8. Risks and rollback

Missing or non-integer macOS output metadata now blocks, intentionally. All
successful binding hashes change. Roll back by reverting the non-marker M113
commit and recomputing the marker pair; never reuse a pre-M113 hardware binding.
This ADR grants no WING contact, reset, correction or live authority, and the
strict readiness score remains 88/100.
