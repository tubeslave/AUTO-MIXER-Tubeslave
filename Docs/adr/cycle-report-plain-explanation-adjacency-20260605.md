# ADR: cycle-report-plain-explanation-adjacency-20260605

## Context

Autoloop reports are part of project memory and are used for plain-language
operator summaries. The mandatory questions section should directly precede the
plain-language explanation. A report can otherwise remain syntactically ordered
while inserting another top-level section in the operator summary path.

## Options considered

1. Add a dedicated adjacency validator for the two final operator-facing
   sections.
2. Rely on existing ordered-section validation.
3. Allow arbitrary top-level sections before the final plain-language
   explanation.

## Decision

Use option 1. The validator now scans between `## Три обязательных вопроса` and
`## Объяснение простым языком` and rejects another top-level `##` heading there.

## Why this won

It preserves report readability and makes the durable memory surface easier to
parse for operators without touching live mixer behavior.

## Rejected alternatives

Existing ordering checks are too broad for this adjacency rule. Allowing extra
sections there makes the final human summary less predictable.

## Implementation plan

1. Add `_validate_mandatory_questions_directly_precede_plain_explanation`.
2. Call it from `validate_report`.
3. Add a regression test with an inserted top-level section.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_codex_cycle_report_validator.py -q`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: useful evidence placed in the wrong location will now fail validation. The
rollback is removing the helper and test, but the preferred fix is moving extra
evidence before the mandatory questions.
