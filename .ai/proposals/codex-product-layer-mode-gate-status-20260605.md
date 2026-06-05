# Codex Proposal: product-layer-mode-gate-status-20260605

## Short thesis

Bring the four-mode autonomy gate into the Product Layer and Settings status
table as read-only evidence.

## What I understood

The project goal requires Analysis, Assisted, Supervised AutoMix, and Full
Autonomous modes to be explicit. The backend gate already explains mode status,
but operators using the Product Layer UI currently see only promotion readiness
and gap buckets, not the mode gate rows themselves.

## Proposed solution

Build `autonomy_mode_transition_gate` inside
`build_processing_registry_snapshot()` using the same
`autonomy_promotion_readiness` snapshot. In the frontend adapter, map
`autonomy_mode_transition_gate.rows` to `mode_gate:*` `ModuleStatusRecord`s.
Then let Settings Mode Policy summarize those rows directly.

The frontend rows should:

- label the project mode names;
- show `available`, `limited_available`, and `blocked` status tones;
- keep `canWriteLive=false`;
- keep `liveWriteKinds=[]`;
- include Supervised `fader/gain` only in text/pipeline metadata;
- surface Full Auto gap counts in `nextStep`.

## Likely files to touch

- `backend/processing_registry.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/views/SettingsView.tsx`
- `tests/test_processing_registry.py`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`

## Alternatives considered

The mode gate could remain visible only in proof reports. That is weaker for
operators because the Settings surface is the active Product Layer policy view.
The UI could parse promotion gaps alone, but then it would miss the four-mode
runtime gate semantics.

## Risks

The largest risk is making Supervised appear like a new frontend live-write
surface. The adapter must leave these rows status-only and use empty
`liveWriteKinds`.

## Test plan

- Backend registry snapshot tests.
- Frontend operator adapter contract tests.
- Frontend Settings source contract and typecheck.
- Compile checks for backend Python and frontend test suite.

## Where the other agent may disagree

Kimi may prefer keeping proof evidence out of the main Product Layer snapshot.
The counterpoint is that this gate is already a registered shadow-only
evaluation method, and exposing it as read-only status improves operator
explainability without changing execution.
