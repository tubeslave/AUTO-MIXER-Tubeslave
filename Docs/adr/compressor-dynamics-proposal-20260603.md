# compressor-dynamics-proposal-20260603

## 1. Context

The long-term project goal requires automatic compression and compressor
console control. Existing compressor code includes controller/apply behavior,
but the current live runtime must remain supervised fader/gain only.

## 2. Options considered

- Reuse `auto_compressor.py` as the main assisted-mode compressor method.
- Keep compression as metadata only in the processing registry.
- Add a separate transport-free compressor proposal layer.

## 3. Decision

Add `backend/compressor_dynamics_proposal.py` as a proposal-only analyzer. It
uses channel dynamics evidence to estimate compressor threshold, ratio, attack,
release, knee, bounded makeup gain, target gain reduction, confidence, blockers,
and evidence. It does not create `requested_change` payloads for live apply.

## 4. Why this won

This moves an explicit project-goal capability from roadmap-only metadata into
a usable Product Layer proposal surface while preserving the current live-write
boundary. It separates compressor reasoning from legacy transport/apply code.

## 5. Rejected alternatives

Direct reuse of `auto_compressor.py` risks mixing analysis with console writes.
Metadata-only tracking does not generate evidence, settings, or testable
proposal payloads.

## 6. Implementation plan

1. Build bounded compressor analysis and recommendation objects.
2. Block low-need, already-compressed, quiet, or insufficient-headroom cases.
3. Emit proposal-only payloads with no live apply fields.
4. Register the method and proposal contract.
5. Update the project-goal capability matrix to reference the clean proposal
   layer.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_compressor_dynamics_proposal.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- Compile/smoke checks with the available Python runtime if full test
  dependencies are unavailable.

## 8. Risks and rollback

The main risk is overstating live readiness. The method is registered as
proposal-only, emits no queueable live changes, and includes blockers for
unsupported compressor value type and missing hardware proof. Rollback is
removal of the module, registry entry, proposal contract, matrix update, tests,
brief, proposal, and ADR; no live transport code is touched.
