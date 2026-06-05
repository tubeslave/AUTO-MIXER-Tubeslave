# Proposal: cycle-report-plain-explanation-adjacency-20260605

## 1. Short thesis

Reject autoloop reports that separate mandatory questions from the
plain-language explanation with another top-level section.

## 2. What I understood

Cycle reports are durable memory. If an inserted `##` section sits between the
mandatory questions and the plain-language explanation, the report still has all
headings but loses the intended human-readable ending structure.

## 3. Proposed solution

Add a validator helper that locates both headings and scans only the text
between them for another `##` heading. Add a regression test with an inserted
`## Extra Evidence` section.

## 4. Likely files to touch

- `tools/codex_autoloop/validate_cycle_report.py`
- `tests/test_codex_cycle_report_validator.py`
- `.ai/briefs/cycle-report-plain-explanation-adjacency-20260605.md`
- `Docs/adr/cycle-report-plain-explanation-adjacency-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Rely only on broad ordered end-section checks. Rejected because those checks
  can allow an inserted section before the final plain-language explanation.
- Move the plain-language section earlier. Rejected because the existing report
  contract treats it as the final operator-facing summary.

## 6. Risks

The rule could reject a report with useful extra evidence. That is acceptable:
extra evidence should live before the mandatory questions, not between the
operator questions and the plain-language explanation.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_codex_cycle_report_validator.py -q`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer allowing a short appendix between the sections. I would keep the
end of the report strict because it is the human-readable safety summary.
