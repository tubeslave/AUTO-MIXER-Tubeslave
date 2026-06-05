# Proposal: fx-send-batch-snapshot-bridge-20260605

## 1. Short thesis

Let batch FX-send proposal generation use one shared read-only FX snapshot for
schema gating across multiple channel analyses.

## 2. What I understood

The single-analysis bridge can already derive schema evidence from
`fx_inserts_snapshot`, but `build_fx_send_proposals()` still requires each
analysis input to carry its own snapshot. A multi-channel decision flow should be
able to pass one Product Layer snapshot once.

## 3. Proposed solution

Add a keyword-only `current_fx_inserts_snapshot` argument to
`build_fx_send_proposals()`. For each analysis that lacks explicit
`current_fx_parameter_schema` and `current_fx_inserts_snapshot`, use
`dataclasses.replace()` to pass the shared snapshot into `analyze_fx_send_need()`.

## 4. Likely files to touch

- `backend/fx_send_proposal.py`
- `backend/processing_registry.py`
- `tests/test_fx_send_proposal.py`
- `.ai/briefs/fx-send-batch-snapshot-bridge-20260605.md`
- `Docs/adr/fx-send-batch-snapshot-bridge-20260605.md`

## 5. Alternatives considered

Require callers to enrich every input. Rejected because it duplicates a
mechanical step and increases wiring mistakes.

Mutate the input dataclasses. Rejected because the inputs are frozen and should
remain deterministic evidence records.

## 6. Risks

Batch-level snapshot evidence could be stale. The existing freshness and schema
gates still apply to each recommendation, and the snapshot is carried as evidence
instead of apply authority.

## 7. Test plan

- Compile changed files.
- Focused `tests/test_fx_send_proposal.py`.
- Broad registry/Product Layer proposal tests.
- Full standard pytest suite.

## 8. Where the other agent may disagree

Kimi may prefer making the shared snapshot mandatory. I would keep it optional
until all call sites are wired through Product Layer state.
