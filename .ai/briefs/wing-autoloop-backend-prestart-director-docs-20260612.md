# Brief: wing-autoloop-backend-prestart-director-docs-20260612

## Task

Synchronize the Codex autoloop README and Project Director tests with the
structured backend-port prestart gate, and keep direction guard scoring from
treating Project Director forbidden-scope policy as completed drift work.

## Context

The Codex cycle prompt and WING MVP blocker table now require
`backend_port_prestart_gate` to match `expected_backend_port_prestart_gate`
before any supervised fader/gain proof. The autoloop README still documented
only the final `supervised_prestart_gate` requirement. A Project Director JSON
report can also include `forbidden_scope` strings such as `memory-trust` and
`queue-import`; those are negative policy boundaries and should not trip drift
scoring by themselves.

## Safety Boundary

- Do not contact WING hardware.
- Do not open a backend WING session.
- Do not execute OSC/WING writes.
- Do not change Project Director live authority or blocker status.
