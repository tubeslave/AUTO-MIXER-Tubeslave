# Modulation And Spatial FX Rules

## Context

The source-knowledge layer already logs EQ, compression, pan, phase, and FX
candidates with source IDs. The latest research pass added guidance from Sound
On Sound, iZotope, FabFilter, Valhalla DSP, Lexicon, and Eventide about reverb,
delay, modulation, pre-delay, early reflections, and ducked returns.

## Options considered

1. Store the new information only in a prose research note.
2. Add a single broad "use FX tastefully" rule.
3. Add multiple bounded, searchable rules by FX decision type.
4. Change live OSC behavior to automatically apply more modulation and delay.

## Decision

Add multiple advisory source-knowledge rules:

- `fx.shared_filtered_returns_context`
- `fx.predelay_by_role_preserve_attack`
- `fx.delay_depth_when_reverb_masks`
- `fx.ducked_returns_front_clarity`
- `fx.modulation_support_width_texture`
- `fx.modulation_reverb_order_intent`
- `fx.early_late_density_by_role`

## Why this won

The offline mixer can now log more precise `selected_rule_ids` and
`source_ids` for FX candidates. The split also keeps later learning data useful:
a delay-as-depth decision is different from a modulation-width decision or a
ducked-return clarity decision.

## Rejected alternatives

A prose-only note would not affect retrieval/logging. A single broad rule would
make the training data too vague. Automatic live FX changes were rejected
because spatial effects can affect intelligibility, feedback risk, and monitor
translation.

## Implementation plan

- Register source metadata in `backend/source_knowledge/data/sources.yaml`.
- Add paraphrased advisory rules to `backend/source_knowledge/data/rules.jsonl`.
- Add retrieval/validation tests in `tests/test_source_knowledge.py`.
- Update `Docs/authoritative_mixing_research_2026-04-26.md`.
- Run the full pytest command.
- Render a new offline mix with source-knowledge logging and MERT shadow scoring.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_source_knowledge.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is low because these are source-knowledge rules and do not directly alter
OSC behavior. Rollback is removing the added source rows, rules, tests, and
this ADR/brief.
