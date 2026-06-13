# Proposal: WING MVP Next Manual Artifact Required By Packet Next Action

## Short thesis

Make the no-WING handoff packet explicitly name the next required real WING
artifact, then require that field across the current WING MVP rails.

## What I understood

The offline handoff packet can guide an operator, but it must not satisfy the
`WING read-only connection proof` blocker by itself. The next required evidence
remains a real `wing_readonly_preflight.v1` summary captured from WING.

## Proposed solution

- Render `next_manual_artifact_required` explicitly in the handoff Markdown.
- Carry the same requirement through the first blocker, MVP doc, read-only
  runbook, Codex autoloop docs, generated progress, and docs index.
- Add regression tests for Project Director, WING MVP rails, no-WING release
  summary, and handoff packet rendering.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`

## Alternatives considered

- Leave the existing plain `Next Required Artifact` Markdown section as-is.
  Rejected because it does not expose the packet key future agents must verify.
- Require the real artifact by prose only. Rejected because regression tests can
  drift without a literal field contract.

## Risks

- The blocker row is long and easy to update incompletely.
- Future cycles may still over-trust offline packets unless the explicit real
  artifact requirement is visible in Project Director output.

## Test plan

- Run focused packet, no-WING release, Project Director, and WING MVP rail tests.
- Run Project Director and direction guard command checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may prefer keeping the renderer unchanged and relying on JSON-only
validation. Rendering the key in Markdown keeps the operator handoff auditable
without opening or parsing JSON.
