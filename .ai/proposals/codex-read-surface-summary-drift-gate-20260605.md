# Proposal: read-surface-summary-drift-gate-20260605

## 1. Short thesis

Recompute read-surface fixture summary facts from row data inside the no-WING
release check so stale or malformed summaries cannot make coverage look better
than it is.

## 2. What I understood

The long-term autonomous console goal depends on trusted read-surface evidence.
The snapshot already declares safety flags and remaining physical gaps, but a
bad summary could misrepresent row coverage if the release check only reads
summary fields.

## 3. Proposed solution

Extend `validate_read_surface_fixture_evidence_snapshot` to:

- require a `surfaces` list
- recompute status counts from rows
- recompute `surface_with_fixture_count`
- recompute `physical_fixture_missing_surface_ids`
- reject unknown row statuses
- treat non-integer summary counts as validation violations

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-summary-drift-gate-20260605.md`
- `Docs/adr/read-surface-summary-drift-gate-20260605.md`

## 5. Alternatives considered

- Trust the builder summary. Rejected because release gates should validate
  evidence, not only echo it.
- Move this into the snapshot builder. Rejected because the release check is the
  boundary that must fail closed on malformed artifacts.
- Require physical fixtures now. Rejected as out of scope for this validation
  slice.

## 6. Risks

The gate could become too strict if summary shape changes. Mitigation: compare
against the canonical status set and row contract, and keep violations explicit.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`

## 8. Where the other agent may disagree

Kimi may want the release check to validate fixture path existence too. That is
reasonable as a later step; this slice closes summary drift first.
