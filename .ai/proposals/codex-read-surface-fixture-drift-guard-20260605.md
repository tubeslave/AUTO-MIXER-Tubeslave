# Codex Proposal: read-surface-fixture-drift-guard-20260605

## Short thesis

Keep the read-surface fixture evidence map honest by testing that every
declared fixture file and pytest function actually exists.

## What I understood

The project now exposes cached DTO fixture coverage as a read-only Product Layer
evidence snapshot. This is useful only if the declarative map keeps matching
the test suite. A renamed fixture test should fail focused validation rather
than silently leave stale evidence in readiness reports.

## Proposed solution

Add a focused pytest in `tests/test_read_surface_fixture_evidence.py` that
builds the snapshot, opens each declared `tests/...` fixture path, parses it
with Python `ast`, and verifies that all declared `test_names` are present as
pytest functions. Missing-evidence rows must keep both lists empty. Add that
focused pytest to the no-WING release check so quick release validation catches
stale read-surface evidence before supervised proof reports are generated.
Also emit a read-surface fixture evidence summary in the no-WING release report
so reviewers see remaining physical readback gaps instead of inferring them
from test names.

## Likely files to touch

- `tests/test_read_surface_fixture_evidence.py`
- `tools/no_wing_release_check.py`
- `.ai/briefs/read-surface-fixture-drift-guard-20260605.md`
- `Docs/adr/read-surface-fixture-drift-guard-20260605.md`

## Alternatives considered

The alternative is leaving this as reviewer discipline. That is weaker because
fixture evidence is now a machine-readable readiness surface.

## Risks

The guard is intentionally test-only, so it can fail when a fixture is renamed
without updating the map. That is the desired failure mode. It does not prove
physical readback and must not be described as live readiness.

## Test plan

- Focused read-surface fixture evidence pytest.
- no-WING quick release check.
- Related read-surface/registry pytest.
- Full backend pytest.
- Memory lint and diff hygiene.

## Where the other agent may disagree

Another agent may prefer a runtime validator. A pytest guard is smaller and
keeps the Product Layer snapshot side-effect free: it still does not execute
tests, query hardware, or touch mixer transports.
