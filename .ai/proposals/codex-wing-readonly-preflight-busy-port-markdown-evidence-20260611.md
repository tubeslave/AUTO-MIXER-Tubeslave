# Proposal: WING Read-Only Preflight Busy-Port Markdown Evidence

## Short thesis

The no-WING summary markdown should state the same busy-port fail-closed confirmation fields as the JSON artifact.

## What I understood

The first WING MVP blocker is still waiting on real hardware proof. The prestart diagnostic can still reduce operator error by making the busy-port stop condition clear in human-readable reports.

## Proposed solution

- Extend `_backend_port_markdown_details` to render:
  - `preflight_blocked_until_listener_confirmed`
  - `intended_backend_confirmation_required`
  - `safe_to_stop_listener_by_gate`
  - `fail_closed_reason`
- Extend the markdown summary test to cover these fields and the listener command line.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/wing-readonly-preflight-busy-port-markdown-evidence-20260611.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_BUSY_PORT_MARKDOWN_EVIDENCE_20260611.md`

## Alternatives considered

- Leave the fields only in JSON: rejected because operators commonly inspect the markdown report first.
- Add a new report file: rejected because the existing summary is the right operator-visible surface.

## Risks

This changes markdown output shape for busy-port failures. The status and JSON summary are unchanged.

## Test plan

- Focused markdown/backend-port tests.
- Adjacent no-WING/WING focused stack.
- Full backend suite before commit.

## Where the other agent may disagree

Kimi may prefer a richer incident-style report. That would be more documentation churn; this slice keeps the existing summary and adds the exact missing fields.
