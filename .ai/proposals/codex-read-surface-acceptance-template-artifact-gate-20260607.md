# Proposal: Read-Surface Acceptance Template Artifact Gate

## Short thesis

The acceptance template is part of the physical-readback review path, so the
release check should validate the exact generated JSON artifact, not only the
tool that writes it.

## What I understood

Automixer needs broad read-surface proof before it can safely widen live console
control. The acceptance template is a manual handoff, not proof by itself. It
must remain pending, operator-edited, and unable to close fixtures.

## Proposed solution

Add a no-WING validator for
`read_surface_physical_readback_acceptance_template.json`. The validator checks
top-level safety flags, status/count consistency, required operator edits,
operator sequence, focused validation command, pending record placeholders, and
false authority flags on every template record.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-acceptance-template-artifact-gate-20260607.md`
- `Docs/adr/read-surface-acceptance-template-artifact-gate-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_ACCEPTANCE_TEMPLATE_ARTIFACT_GATE_20260607.md`
- `Docs/INDEX.md`

## Alternatives considered

- Trust the generator tests only. That misses artifact-path or release-check
  integration drift.
- Let the ledger reject bad records later. That still allows a misleading
  generated handoff to appear in release artifacts.
- Convert the template into fixture closure. Rejected because closure must stay
  a separate manual decision.

## Risks

The main risk is overclaiming. The validator explicitly fails any generated
template that grants live authority, fixture closure, execution permission, or
full autonomous readiness.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py tools/build_read_surface_physical_readback_acceptance_template.py`
- Focused pytest for the template tool and no-WING validator tests.
- no-WING quick run with `/tmp` artifact root.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue that the next step should be physical WING capture. That remains
the real blocker, but this gate makes the offline handoff harder to misuse while
hardware proof is still pending.
