# Brief: WING MVP Backend Listener Template Edit Gate

Date: 2026-06-13

## Task

Prevent a generated backend-listener confirmation template from becoming an
accepted confirmation while it still carries pending-template markers or
placeholder operator fields.

## Context

The no-WING backend-port gate writes a pending confirmation template when a
listener is already bound to `127.0.0.1:8765`. That template reduces manual
copying risk, but it must remain blocked until an operator performs a real
review and edits the template fields.

## Scope

- Reject accepted confirmations that leave placeholder operator fields.
- Reject accepted confirmations that leave `template_status:
  pending_operator_edit`.
- Require `accepted_by_template: true` when the JSON still carries the
  template-origin marker.
- Keep standalone manually written accepted confirmation JSON valid.

## Out Of Scope

- No WING hardware contact.
- No backend listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No readiness update or blocker closure.
