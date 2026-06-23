# Brief: WING MVP Backend Prestart Gate Object Summary JSON Surface

Date: 2026-06-15

## Task

Expose the structured `backend_port_prestart_gate`,
`expected_backend_port_prestart_gate`, and
`declared_expected_backend_port_prestart_gate` objects as top-level no-WING
`summary.json` fields.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Preserve the prestart gate as a no-authority requirement before any
  backend/WING session.
- Keep the change limited to no-WING reporting, tests, and Council evidence.

## Expected Evidence

- Top-level `summary.json` exposes the backend prestart gate object, expected
  gate object, declared expected gate object, and existing match verdicts.
- The top-level backend prestart gate still shows
  `grants_live_authority: false` and
  `must_pass_before_backend_wing_session: true`.
- Focused and standard tests pass.
