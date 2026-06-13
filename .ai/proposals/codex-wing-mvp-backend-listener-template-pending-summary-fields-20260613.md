# Proposal: WING MVP Backend Listener Template Pending Summary Fields

## Short thesis

Expose the generated listener confirmation template's pending acceptance markers
in no-WING summary output so operator review can see the template is not an
accepted confirmation.

## What I understood

The first blocker remains WING read-only connection proof. The generated
backend-listener confirmation template helps an operator prepare a confirmation,
but it must remain visibly pending until the operator replaces placeholders and
marks the template reviewed.

## Proposed solution

- Copy `accepted_by_template: false` from the generated template into the
  failed `backend_port_free` summary detail.
- Copy placeholder operator fields and a placeholder-field list into the same
  detail.
- Render those fields in `summary.md`.
- Assert both JSON and Markdown summary surfaces in tests.
- Update the read-only preflight runbook.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/reports/`

## Alternatives considered

Only relying on the generated JSON template is weaker because an operator may
review the failed summary first. Adding new gate semantics is unnecessary; this
change only exposes already-existing pending metadata.

## Risks

The summary shows placeholder strings by design. They are not secrets and make
the pending state explicit.

## Test plan

Run no-WING release check tests, WING rail tests, direction guard, memory lint,
no-WING quick fail-closed validation, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer fewer summary fields. The stronger operator-visible evidence is
worth the small verbosity because this gate protects the first hardware step.
