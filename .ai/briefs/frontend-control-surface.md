# Task frontend-control-surface: Remote frontend for live auto-mixer control

## Problem

Create a focused frontend direction for controlling AUTO-MIXER Tubeslave from MacBook,
iPad, and iPhone. The UI must expose the most important operator controls without
turning into a dense metrics dashboard.

## Constraints

- Preserve live-sound safety behavior and never encourage gain increases above safe limits.
- Keep mixer state rollback visible before applying automated changes.
- Use existing backend/WebSocket capabilities where possible:
  - mixer connection: WING, dLive, Mixing Station
  - audio device discovery/selection
  - channel selection and meters
  - undo snapshots / restore
  - bypass and emergency stop
  - pending AI actions / action history
- Do not edit production frontend code during this planning/design step.
- Do not add production dependencies without a written reason.
- Dark theme, touch-friendly controls, responsive layouts for MacBook/iPad/iPhone.

## Definition of Done

- Provide 2-3 UI variants with clear tradeoffs.
- Visualize each variant.
- Recommend one default direction for implementation.
- Identify likely frontend/API files for the next implementation step.

## Test Command

`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
