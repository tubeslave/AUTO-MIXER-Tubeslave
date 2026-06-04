# Codex Proposal: Supervised Proof Artifact Report Gate

## 1. Short thesis

Turn the supervised proof evaluator into an operator-runnable offline report
gate, so promotion discussions are based on current artifacts rather than
memory or scattered logs.

## 2. What I understood

The final Automixer target requires supervised and eventually autonomous
console control, but promotion must be evidence-driven. The repository now has
a proof artifact evaluator; the next practical gap is a repeatable way to find
proof summaries, evaluate them, and write a machine-readable readiness report.

## 3. Proposed solution

Add `tools/evaluate_supervised_proof_artifacts.py`. The tool:

- recursively discovers `summary.json` files under provided artifact roots;
- ignores unrelated summaries;
- evaluates proof-like summaries through the read-only evaluator;
- separates dry-run-ready, physical supervised candidates, blocked artifacts,
  invalid summaries, and ignored summaries;
- can require a minimum number of physical promotion candidates through an exit
  code gate;
- writes a JSON report without connecting to any mixer transport.

## 4. Likely files to touch

- `tools/evaluate_supervised_proof_artifacts.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `backend/processing_registry.py`
- `.ai/briefs/supervised-proof-artifact-report-20260604.md`
- `Docs/adr/supervised-proof-artifact-report-20260604.md`

## 5. Alternatives considered

- Keep using the evaluator only from tests: rejected because operators need a
  durable artifact report before real proof promotion.
- Treat every `summary.json` as a proof artifact: rejected because `artifacts/`
  already contains unrelated reports.
- Require physical candidates by default: rejected because dry-run and no-WING
  runs should still be reportable without failing.

## 6. Risks

- Unrelated summaries could create noise. The report ignores summaries that are
  not proof-like.
- A dry-run artifact could be treated as live proof. The report keeps
  `require_physical` and promotion candidates separate.
- A report could be mistaken for permission. The safety block explicitly sets
  live write and live widening to false.

## 7. Test plan

- Verify the tool is transport-free.
- Verify discovery only returns `summary.json`.
- Verify dry-run, physical, ignored, and invalid summaries are separated.
- Verify exit codes for minimum physical candidates.
- Verify no-WING release checks can include the report command without live
  command tokens.

## 8. Where the other agent may disagree

Kimi may argue the next step should be the real WING proof itself. That remains
the next hardware milestone, but this report gate makes the proof evidence
auditable before any safety matrix update.
