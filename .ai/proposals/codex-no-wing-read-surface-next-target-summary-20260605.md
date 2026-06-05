# Proposal: no-wing-read-surface-next-target-summary-20260605

## 1. Short thesis

Make the next read-surface hardware proof target a first-class no-WING release
summary field.

## 2. What I understood

The Product Layer and Settings can now display a next readback target. The
source release artifact should expose the same target so operators and tools
share one audited value.

## 3. Proposed solution

Add `next_physical_readback_surface_id` to
`build_read_surface_fixture_evidence_snapshot()`, validate it in
`validate_read_surface_fixture_evidence_snapshot()`, propagate it through
`read_surface_fixture_evidence_summary()`, and render it in Markdown.

## 4. Likely files to touch

- `backend/read_surface_fixture_evidence.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_fixture_evidence.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/no-wing-read-surface-next-target-summary-20260605.md`
- `Docs/adr/no-wing-read-surface-next-target-summary-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Keep deriving the target only in Product Layer. Rejected because release
  artifacts should be the source for proof evidence.
- Add a ranked remediation workflow. Rejected until physical readback execution
  is specified.
- Allow target/list drift. Rejected because that would weaken operator evidence.

## 6. Risks

Operators may treat the next target as permission to query hardware. Mitigation:
the field remains inside no-WING read-only evidence and the summary keeps
`live_write=false` and `full_autonomous_ready=false`.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_fixture_evidence.py tests/test_no_wing_release_check.py -q`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer a richer remediation priority model. I would first preserve one
explicit, validated release artifact target, then add ranking later.
