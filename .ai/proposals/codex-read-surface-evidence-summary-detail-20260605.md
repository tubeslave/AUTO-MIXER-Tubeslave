# Proposal: read-surface-evidence-summary-detail-20260605

## 1. Short thesis

Surface read-surface evidence proof quality in the top-level no-WING summary,
not only in the verbose check result.

## 2. What I understood

Recent release-gate work made read-surface fixture evidence more trustworthy:
summary counts are recomputed, declarations are checked against real pytest
functions, and row status must match contracts/gaps. Operators should see those
facts directly in the release summary.

## 3. Proposed solution

Extend `read_surface_fixture_evidence_summary` and the Markdown writer with:

- `declared_fixture_path_count`
- `declared_test_name_count`
- `physical_gap_surface_count`
- `summary_recomputed_from_rows`
- `fixture_declaration_validation`
- `contract_gap_consistency_enforced`

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-evidence-summary-detail-20260605.md`
- `Docs/adr/read-surface-evidence-summary-detail-20260605.md`

## 5. Alternatives considered

- Keep details only in `results`. Rejected because summary consumers should not
  parse deep check internals for proof quality.
- Add a new artifact file. Rejected because the top-level summary already owns
  release proof facts.

## 6. Risks

The summary could imply more readiness than exists. Mitigation: keep
`full_autonomous_ready=false`, keep physical gaps visible, and name the fields as
validation/enforcement facts rather than approval.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer a separate proof-quality section. I would keep it inside the
existing read-surface fixture evidence section to avoid another artifact surface.
