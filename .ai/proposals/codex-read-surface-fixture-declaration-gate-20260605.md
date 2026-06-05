# Proposal: read-surface-fixture-declaration-gate-20260605

## 1. Short thesis

Promote fixture declaration validation from a builder unit test into the no-WING
release schema so autonomy evidence cannot rely on stale test names.

## 2. What I understood

Read-surface readiness currently depends on declarative rows that name fixture
files and pytest functions. The builder test checks those declarations, but the
release proof boundary should independently fail if a report row points at a
missing file or nonexistent test.

## 3. Proposed solution

Extend `validate_read_surface_fixture_evidence_snapshot` to inspect every row:

- `fixture_paths` and `test_names` must be lists
- non-missing rows require at least one fixture path and test name
- fixture paths must be relative `tests/...` files that exist under the repo
- declared test names must be present as pytest functions in at least one
  declared fixture file
- `missing_fixture_evidence` rows must not carry stale declarations

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-fixture-declaration-gate-20260605.md`
- `Docs/adr/read-surface-fixture-declaration-gate-20260605.md`

## 5. Alternatives considered

- Rely on `tests/test_read_surface_fixture_evidence.py` only. Rejected because
  release reports should validate their own evidence boundary.
- Execute each named test from the schema validator. Rejected because the schema
  validator must remain a cheap read-only guard and the release runner already
  executes the fixture evidence test file separately.

## 6. Risks

The gate may fail if test functions are renamed without updating evidence rows.
That is the intended failure mode; stale evidence should block release proof.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer validating AST declarations in `backend/read_surface_fixture_evidence.py`.
I would keep the builder simple and put fail-closed verification at the release
boundary.
