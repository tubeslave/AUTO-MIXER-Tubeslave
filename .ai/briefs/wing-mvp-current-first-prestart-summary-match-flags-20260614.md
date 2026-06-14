# WING MVP Current First Prestart Summary Match Flags

The first open WING MVP blocker remains `WING read-only connection proof`.

The handoff packet validator now exposes packet-vs-plan match flags for the
prestart summary type, status, connection policy, no-contact booleans, and
backend check status. Project Director should not treat `Current First Command`
as fully covered unless those flags are also part of the baseline.

Scope:

- Keep this change in offline rail documentation and Project Director baseline
  markers.
- Do not contact WING, the backend WebSocket, OSC, or live hardware.
- Do not close readiness or grant live authority.
- Regenerate the latest MVP progress report after updating the marker list.

Validation:

- Focused Project Director and WING MVP rail tests.
- Project Director JSON confirms baseline coverage still passes.
- Full standard pytest before commit.
