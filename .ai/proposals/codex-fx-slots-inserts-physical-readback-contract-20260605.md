# Codex Proposal: FX Slots/Inserts Physical Readback Contract

## Short thesis

Add `fx_slots_inserts_physical_readback_contract` as a shadow-only, transport-free
schema contract before trusting future FX slot, FX parameter, or insert
assignment physical artifacts.

## What I understood

The system needs broader console read proof, but `fx_slots_and_inserts` is still
cache-backed. Cached FX DTOs should stay visible while the missing physical
readback fixture remains explicit.

## Proposed solution

- Add a backend contract builder and validator for FX slots, parameters, insert
  assignment rows, coverage flags, and no-mutation safety flags.
- Wire it into processing registry snapshots, goal-capability support paths,
  read-surface remediation expected artifacts, no-WING release summaries, Product
  Layer proof-quality parsing, and frontend module-status rows.
- Add targeted tests for valid artifacts, live/cache rejection, incomplete rows,
  no-WING drift, and UI contract visibility.

## Likely files to touch

- `backend/fx_slots_inserts_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- `tests/`
- `.ai/briefs/`, `.ai/proposals/`, `Docs/adr/`, `.ai/memory/project.md`

## Alternatives considered

1. Reuse matrix insert proof. Rejected because FX slots also need model and
   parameter schema evidence.
2. Wait for a hardware runner. Rejected because future artifacts would remain
   underspecified.
3. Treat cached FX DTOs as enough. Rejected because cache evidence is not
   physical proof.

## Risks

The main risk is that a schema contract is mistaken for completed hardware
proof. The implementation keeps the physical fixture gap and live authority
false in backend, no-WING, Product Layer, and frontend summaries.

## Test plan

- Focused backend contract and no-WING tests.
- Product Layer proof-quality tests.
- Frontend adapter contract tests and build.
- Full backend pytest before commit.

## Where the other agent may disagree

Kimi may prefer defining the physical readback runner first. The schema-first
approach is safer here because it creates a reviewable contract without any
transport or live write path.
