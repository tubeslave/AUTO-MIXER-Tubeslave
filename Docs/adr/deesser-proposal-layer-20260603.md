# deesser-proposal-layer-20260603

## 1. Context

The long-term project goal requires automatic de-essing. The repo already had
sibilance vocabulary and detectors, but no registered de-esser method, no
proposal-only contract, and no safety boundary for live de-esser writes.

## 2. Options considered

- Reuse AI agent `adjust_deesser` live calls.
- Treat sibilance as ordinary static AutoEQ.
- Add a transport-free de-esser proposal layer.

## 3. Decision

Add `backend/deesser_vocal_sibilance.py` as a transport-free recommendation
module and register it as `proposal_only`. It outputs de-esser parameters and
evidence for operator review, but it does not create `requested_change` payloads
for live apply.

## 4. Why this won

It closes an explicit project-goal gap while preserving the supervised
fader/gain live boundary. The proposal layer can later feed UI, replay, and
training without becoming an unsafe console executor.

## 5. Rejected alternatives

Direct AI-agent de-esser apply bypasses the product proposal contract. Static
EQ alone does not model vocal sibilance as dynamic content and can over-dull
vocals or cymbals.

## 6. Implementation plan

1. Build bounded vocal sibilance analysis and recommendation objects.
2. Block non-vocal and low-confidence cases.
3. Expose proposal-only payloads with no live apply fields.
4. Register the method and proposal contract.
5. Update the project-goal capability matrix from missing to proposal-only.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_deesser_vocal_sibilance.py -q`
- Existing mode/capability/offline validation contract tests.

## 8. Risks and rollback

The main risk is overstating live readiness. The method is registered as
proposal-only, emits no queueable live changes, and includes blockers for
unsupported value type and missing hardware proof. Rollback is removal of the
module, registry entry, proposal contract, matrix update, tests, and docs.
