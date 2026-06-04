# Codex Proposal: Read Surface Remediation Plan

Task id: read-surface-remediation-plan-20260604
Date: 2026-06-04

## 1. Short thesis

Full autonomous mixing cannot be credible until the project has complete,
audited read DTOs for console state. The current inventory needs an actionable
read-only remediation plan.

## 2. What I understood

Write proof gates are improving, but the project goal also demands reading all
available console parameters. Current read coverage is intentionally incomplete:
channel processing, matrices, routing/sends, snapshots, presets, and dLive
parity need explicit follow-up.

## 3. Proposed solution

Add `read_surface_remediation_plan`, a pure builder that:

- reads the declarative read-surface inventory;
- creates remediation actions for partial/planned/missing rows;
- names missing contracts such as Product Layer DTOs, coverage flags, stale
  cache metadata, physical query fixtures, destructive diff markers, and dLive
  parity mapping;
- exposes the plan in the processing registry snapshot.

## 4. Likely files to touch

- `backend/read_surface_remediation_plan.py`
- `backend/processing_registry.py`
- `tests/test_read_surface_remediation_plan.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_processing_registry.py`

## 5. Alternatives considered

- Keep read gaps as prose in the inventory.
- Start implementing live read queries immediately.
- Fold read remediation into write proof reports.

## 6. Risks

The plan could be mistaken for runtime read authority. It must be transport-free
and mark `mutates_console_state=false`, `live_write=false`, and
`full_autonomous_ready=false`.

## 7. Test plan

- Unit-test action creation for current partial/missing surfaces.
- Unit-test complete-surface status.
- Assert registry visibility without executor status.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may prefer implementing one DTO immediately. The remediation plan is the
right next step because it ranks the whole read-surface problem before any one
surface can be claimed complete.
