# Codex Proposal: WING MVP Selected Blocker Summary Surface

## Short thesis

Mirror `selected_mvp_blocker` and `selected_mvp_blocker_status` into the
top-level no-WING summary surface so operator-facing evidence is visibly tied to
the current WING read-only connection proof blocker.

## What I understood

The project goal remains the supervised WING Rack MVP path toward a broader AI
mixing engineer. Current guard output says the repo is drifting and the next
valid blocker is still `WING read-only connection proof`. The no-WING packet
already has the selected blocker, but the summary top level does not expose it.

## Proposed solution

Add the two selected-blocker fields to
`build_wing_readonly_mvp_blocker_anchor_summary_surface()` and assert they match
the handoff packet artifact in the existing summary test.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-selected-blocker-summary-surface-20260617.md`
- `Docs/adr/wing-mvp-selected-blocker-summary-surface-20260617.md`
- `Docs/reports/AUTOMIXER_WING_MVP_SELECTED_BLOCKER_SUMMARY_SURFACE_20260617.md`

## Alternatives considered

- Leave the fields nested only in `wing_readonly_preflight_handoff_packet_artifact`.
  This keeps the data available but makes the operator-facing top-level summary
  weaker.
- Add a new independent summary surface. That is unnecessary because the
  existing MVP blocker anchor surface already owns this context.

## Risks

The change is intentionally read-only and summary-only. The main risk is stale
or mismatched fields, so the test asserts the top-level values are direct packet
mirrors.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root <tmp>`
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer waiting for a real WING artifact instead of adding another
summary surface field. The counterpoint is that this field reduces the chance
of mistaking offline packet evidence for blocker closure during the real
artifact handoff.
