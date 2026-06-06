# Codex Proposal: Read Surface Fixture Acceptance Template

## Short thesis

Add an offline acceptance-template builder over the existing review queue so the
operator can create ledger-compatible records without granting fixture closure
or live authority.

## What I understood

The project needs to move toward reading all console state, but the current
safe path still depends on operator-captured physical readback artifacts. The
repo already validates candidate artifacts and acceptance records. The missing
piece is a concrete pending record template that connects those two layers.

## Proposed solution

Create `tools/build_read_surface_physical_readback_acceptance_template.py`.
The tool should:

- read the aggregate artifact review queue;
- emit templates only for valid review candidates;
- leave `decision`, `reviewed_by`, `reviewed_at`, and notes as operator-edited
  placeholders;
- keep live, mutation, fixture-closure, and autonomous-readiness flags false;
- support JSON and Markdown output;
- run as a no-WING smoke command.

## Likely files to touch

- `tools/build_read_surface_physical_readback_acceptance_template.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_fixture_acceptance_template.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-fixture-acceptance-template-20260607.md`
- `.ai/proposals/codex-read-surface-fixture-acceptance-template-20260607.md`
- `Docs/adr/read-surface-physical-readback-acceptance-template-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_PHYSICAL_READBACK_ACCEPTANCE_TEMPLATE_20260607.md`
- `Docs/INDEX.md`

## Alternatives considered

1. Let operators hand-write acceptance records.
2. Make the acceptance ledger generate templates internally.
3. Add a hardware capture/review runner.

## Risks

- A template could look like an accepted record if placeholders are too
  permissive.
- A template could accidentally imply fixture closure or live readiness.
- no-WING checks could start depending on physical artifacts.

## Test plan

- `python3 -m py_compile tools/build_read_surface_physical_readback_acceptance_template.py tools/no_wing_release_check.py`
- Focused pytest for the new template tool and no-WING command-plan test.
- no-WING quick smoke.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue this should be a backend function instead of a CLI. The CLI is
still the better operator handoff because it produces a concrete artifact in
the same no-WING run directory as the capture and preflight packets.
