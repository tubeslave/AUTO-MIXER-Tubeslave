# Proposal: WING MVP Backend Listener Template Authority Summary Fields

## Short thesis

Expose the generated listener confirmation template's execution and readiness
closure denial markers in no-WING summary output so operator review can see the
template is not authority for any backend/WING session.

## What I understood

The first blocker remains WING read-only connection proof. The generated
backend-listener confirmation template is only an operator-edit aid. It must not
grant execution authority, live authority, or readiness closure while it is
pending.

## Proposed solution

- Copy `execution_allowed_by_template: false` from the generated template into
  the failed `backend_port_free` summary detail.
- Copy `readiness_closure_allowed_by_template: false` from the generated
  template into the same detail.
- Render both fields in `summary.md`.
- Assert both JSON and Markdown summary surfaces in tests.
- Update the read-only preflight runbook, blocker canon, and report index.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/reports/`

## Alternatives considered

Only relying on the generated JSON template is weaker because the failed
summary is the first artifact an operator or Project Director sees. Adding a
new authority gate is unnecessary; this change exposes already-existing false
markers.

## Risks

The summary becomes more verbose. The fields are explicit safety denials, so
the verbosity is acceptable for the first hardware preflight boundary.

## Test plan

Run no-WING release check tests, WING rail tests, Project Director/direction
guard, memory lint, no-WING quick fail-closed validation, and the standard full
pytest command.

## Where the other agent may disagree

Kimi may prefer keeping template authority details inside the JSON template.
The stronger operator-visible summary evidence is worth the small duplication
because it prevents mistaking a generated template for execution or readiness
authority.
