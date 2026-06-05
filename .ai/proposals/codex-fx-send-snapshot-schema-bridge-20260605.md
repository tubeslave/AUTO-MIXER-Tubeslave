# Proposal: fx-send-snapshot-schema-bridge-20260605

## 1. Short thesis

Let `fx_send_proposal` consume read-only `fx_inserts_snapshot` schema evidence
directly, so FX send recommendations are gated by the same Product Layer view
operators inspect.

## 2. What I understood

FX automation remains proposal-only. The previous schema gate improved safety,
but it expected callers to pass a normalized `current_fx_parameter_schema`.
Because the canonical read source is `fx_inserts_snapshot`, the next step is a
small bridge that derives the scoped schema payload from that snapshot.

## 3. Proposed solution

Add `current_fx_inserts_snapshot` to `FXSendAnalysisInput` and a transport-free
`fx_parameter_schema_from_snapshot()` helper. It scans cached FX slot rows for a
mapped schema matching the selected FX type. If no scoped match exists, it
returns the snapshot-level schema summary, which keeps generic/unknown schemas
blocked by the existing gate.

## 4. Likely files to touch

- `backend/fx_send_proposal.py`
- `backend/processing_registry.py`
- `tests/test_fx_send_proposal.py`
- `.ai/briefs/fx-send-snapshot-schema-bridge-20260605.md`
- `Docs/adr/fx-send-snapshot-schema-bridge-20260605.md`

## 5. Alternatives considered

Make callers normalize schema evidence themselves. Rejected because it duplicates
logic and makes future proposal call sites easy to wire incorrectly.

Import `build_fx_inserts_snapshot` directly. Rejected because `fx_send_proposal`
should stay transport-free and should not know how Product Layer snapshots are
built.

## 6. Risks

Snapshot evidence is cached DTO evidence, not physical proof. The bridge carries
freshness, read coverage, and safety flags into evidence, and all proposals
remain non-queueable and non-live-applyable.

## 7. Test plan

- Compile changed Python files.
- Focused `tests/test_fx_send_proposal.py`.
- Broad Product Layer/registry/proposal tests.
- Full standard pytest suite.

## 8. Where the other agent may disagree

Kimi may argue to require snapshots for every FX-send proposal now. I would keep
legacy compatibility until all call sites pass Product Layer snapshots.
