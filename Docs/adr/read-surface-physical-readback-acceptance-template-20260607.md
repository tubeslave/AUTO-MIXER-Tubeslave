# ADR: Read Surface Physical Readback Acceptance Template

## Context

The long-term Automixer goal requires verified access to console state before
automation can safely reason about channels, buses, matrices, FX, scenes,
presets, and raw mixer state. The repository already has physical-readback
contracts, artifact intake, an aggregate review queue, and an acceptance ledger.
Operators still need a concrete pending record template after a valid review
candidate appears.

## Options considered

1. Require operators to hand-write acceptance records.
2. Add template generation inside the acceptance ledger.
3. Add a separate offline template builder over the review queue.

## Decision

Use option 3. Add
`tools/build_read_surface_physical_readback_acceptance_template.py` as a
transport-free offline CLI. It renders JSON or Markdown templates for valid
review candidates and leaves the decision fields as operator-edited
placeholders.

## Why this won

The template builder gives operators a concrete artifact without making the
acceptance ledger responsible for file generation or side effects. It also keeps
the no-WING release run honest: when there are no valid candidates, the output
states `operator_candidate_missing` instead of creating a fake acceptance
record.

## Rejected alternatives

- Hand-written records are error-prone and easy to mismatch against the
  candidate artifact source id or contract id.
- Ledger-internal template generation mixes validation with artifact creation.
- Automated hardware capture/review remains premature before the real
  read-only WING proof has been captured and manually reviewed.

## Implementation plan

1. Add the offline template builder CLI.
2. Generate records only from review-queue items accepted as physical fixture
   candidates.
3. Keep `decision`, `reviewed_by`, `reviewed_at`, and notes as placeholders.
4. Keep fixture-closure, live-authority, and autonomous-readiness flags false.
5. Run the builder as a no-WING smoke command in the release-check run
   directory.

## Test plan

- `python3 -m py_compile tools/build_read_surface_physical_readback_acceptance_template.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py tests/test_no_wing_release_check.py::test_no_wing_release_check_quick_mode_still_runs_write_surface_scan tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_fixture_acceptance_ledger_snapshot`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-acceptance-template --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk is limited to offline operator guidance. A stale or unedited template must
be rejected by the acceptance ledger because the decision and reviewer fields
remain placeholders. Rollback is to remove the template builder, the no-WING
smoke command, tests, brief/proposal, ADR, report, and index entry. No live
WING/OSC/MIDI/TCP transport or write path is affected.
