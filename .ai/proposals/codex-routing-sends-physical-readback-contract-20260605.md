# Codex Proposal: Routing/Sends Physical Readback Contract

## Short thesis

Add a shadow-only routing/sends physical readback contract that defines what a
future hardware artifact must contain, without querying the console or widening
live authority.

## What I understood

`routing_and_sends` already exposes cached/static DTO evidence, but that is not
physical proof. The next safe slice is a contract and release visibility layer,
not a live read or write path.

## Proposed solution

- Add `backend/routing_sends_physical_readback_contract.py` with schema fields
  for static Dante routes, channel bus/main/matrix sends, output routes, coverage
  flags, minimum row counts, allowed physical sources, and forbidden mutation
  evidence.
- Register the method as `shadow_only` in `processing_registry` and reference it
  from `console_read_parameters`.
- Include the contract in read-surface fixture evidence, remediation planning,
  no-WING validation/summary output, Product Layer proof-quality normalization,
  and the frontend Module Status adapter.
- Add tests that validate both the happy path and malicious source claims that
  try to grant live authority.

## Likely files to touch

- `backend/routing_sends_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Focused backend/frontend tests and council docs.

## Alternatives considered

Promote cached routing/send DTO rows directly. Rejected because cached state is
not physical readback and would blur the proof boundary for routing and sends.

Add a live WING routing query now. Rejected for this slice because it would
change transport behavior and needs separate operator-supervised proof design.

## Risks

- Contract visibility could be mistaken for proof readiness.
- Summary counters may drift if fixture declarations are updated without tests.
- Routing/send schemas touch multiple promotion families, so false authority
  flags must remain explicit.

## Test plan

- Compile the touched backend modules.
- Run focused backend tests for routing/sends contract, fixture evidence,
  remediation plan, registry, no-WING, and Product Layer proof quality.
- Run frontend adapter tests and frontend build.
- Run no-WING quick check without frontend and memory lint.
- Run the standard backend pytest command before commit.

## Where the other agent may disagree

Kimi may argue for physical WING readback first. That is valid as the next
proof step, but this slice deliberately adds the contract gate before trusting
any artifact as evidence.
