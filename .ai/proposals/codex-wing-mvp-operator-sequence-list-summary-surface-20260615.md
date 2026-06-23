# Proposal: WING MVP Operator Sequence List Summary Surface

## Short thesis

Promote the handoff `operator_sequence` and `expected_operator_sequence` lists to
the flat no-WING summary surface while preserving all existing fail-closed
count, digest, and contract checks.

## What I understood

The current WING read-only proof rail is intentionally blocked until a real
`wing_readonly_preflight.v1 summary.json` exists. The no-WING handoff packet
already proves that its operator sequence count and SHA256 digest match the
contract, but reviewers still need the ordered checklist visible in the same
flat surface used by Project Director and operator review.

## Proposed solution

- Read the nested handoff artifact detail lists and emit string-only
  `operator_sequence` and `expected_operator_sequence` fields in the flattened
  no-WING handoff summary.
- Copy those fields through the top-level JSON and Markdown summaries.
- Test that both fields equal the shared expected handoff operator sequence.
- Add the field names to the Project Director baseline marker list so drift in
  the first command is visible.
- Update WING MVP blocker/runbook docs and traceability artifacts.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/briefs/wing-mvp-operator-sequence-list-summary-surface-20260615.md`
- `Docs/adr/wing-mvp-operator-sequence-list-summary-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_LIST_SUMMARY_SURFACE_20260615.md`

## Alternatives considered

- Keep relying on nested detail output only. Rejected because operator review
  and Project Director first-command coverage operate on the flat summary.
- Add only Markdown output. Rejected because the machine-readable JSON summary
  must carry the same evidence.

## Risks

- Summary shape drift could hide a real sequence mismatch. Mitigation: keep the
  existing count/hash/contract checks and add focused tests for the promoted
  lists.
- Generated autoloop docs could become stale. Mitigation: update the baseline
  source and regenerate.

## Test plan

- `PYTHONPATH=backend python -m py_compile ...`
- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
- `git diff --check`

## Where the other agent may disagree

Kimi may argue that exposing the full ordered lists duplicates the digest and
count fields. I would keep the lists because the duplicate surface is intentional
review ergonomics: hashes prove identity, but operators need to inspect the
actual ordered checklist before opening a backend/WING session.
