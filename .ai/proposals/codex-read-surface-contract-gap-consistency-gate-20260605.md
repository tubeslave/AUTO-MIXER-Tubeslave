# Proposal: read-surface-contract-gap-consistency-gate-20260605

## 1. Short thesis

Add row-level consistency checks to the no-WING read-surface fixture schema so
cached DTO evidence cannot masquerade as physical console proof.

## 2. What I understood

Full autonomous operation requires trustworthy console read awareness. Current
rows correctly show cached DTO coverage with physical readback gaps, but the
release gate should reject contradictory rows if future edits change status,
contracts, or gaps incorrectly.

## 3. Proposed solution

Extend `validate_read_surface_fixture_evidence_snapshot` to validate:

- `surface_id` and `target_console_surface` are non-empty strings
- `covered_contracts` and `remaining_gaps` are lists
- non-missing rows have at least one contract
- missing rows have no contracts, paths, or test names
- partial rows do not include `physical_readback_fixture` and still report
  `physical_readback_fixture_missing`
- present rows require `physical_readback_fixture` and must not report
  `physical_readback_fixture_missing`
- rows without physical proof keep `hardware_parity_unproven`

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-contract-gap-consistency-gate-20260605.md`
- `Docs/adr/read-surface-contract-gap-consistency-gate-20260605.md`

## 5. Alternatives considered

- Trust the snapshot builder status. Rejected because release gates should
  validate evidence invariants independently.
- Promote physical evidence now. Rejected because no physical WING proof exists
  in this no-WING path.

## 6. Risks

Future status semantics may need new invariants. Explicit violation IDs make
that migration visible.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may want this in the builder dataclass. I would keep builder checks light
and put release-proof consistency at the release boundary.
