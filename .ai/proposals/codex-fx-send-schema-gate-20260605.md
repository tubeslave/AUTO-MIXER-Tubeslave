# Proposal: fx-send-schema-gate-20260605

## 1. Short thesis

Gate FX-send proposals on scoped FX parameter schema hints so assisted FX
recommendations do not proceed on unknown or mismatched FX-slot semantics.

## 2. What I understood

FX/send automation is part of the full Automixer goal, but FX writes are still
outside the supervised live executor. We can still improve assisted-mode quality
by requiring stronger read-only evidence before an FX-send recommendation becomes
proposal-only.

## 3. Proposed solution

Extend `FXSendAnalysisInput` with `current_fx_parameter_schema`. Normalize that
payload into recommendation evidence. If a schema payload is explicitly present
and is generic, missing, unscoped, or scoped to a different FX family than the
selected preset, keep the result as `analysis_only`.

Do not require schema payloads for legacy callers yet; absent evidence stays
allowed but visible as `not_provided`.

## 4. Likely files to touch

- `backend/fx_send_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `tests/test_fx_send_proposal.py`
- `.ai/briefs/fx-send-schema-gate-20260605.md`
- `Docs/adr/fx-send-schema-gate-20260605.md`

## 5. Alternatives considered

Require schema evidence for every FX-send proposal immediately. Rejected for now
because existing callers may not pass Product Layer snapshots yet.

Ignore schema evidence until physical WING fixtures exist. Rejected because
generic read-only hints are already available and can prevent unsafe assumptions.

## 6. Risks

Schema hints are not physical WING model proof. The gate treats them as
proposal-quality evidence only; live writes remain blocked and physical proof is
still required before promotion.

## 7. Test plan

- Compile changed Python files.
- Focused `fx_send_proposal` tests.
- Registry/adapter tests touching proposal-only contracts.
- Full suite with the standard pytest command.

## 8. Where the other agent may disagree

Kimi may argue to make schema evidence mandatory immediately. I would stage that
after all callers can pass scoped Product Layer FX slot schema payloads.
