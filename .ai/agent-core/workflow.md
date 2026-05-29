# Agent Core Workflow

## Purpose

This workflow keeps autonomous agent work narrow, auditable, and safe for a
live-sound repository.

## Default Cycle

1. Read memory and project rules.
2. Check git status and protect unrelated local changes.
3. Classify the task as documentation, tests, offline analysis, dry-run tooling,
   supervised approval, or live-write request.
4. Check safety policy before planning any mixer-adjacent change.
5. Inspect the smallest relevant files.
6. Plan the smallest useful diff.
7. Execute safe changes.
8. Run focused tests first, then broader tests if practical.
9. Log decisions and session results.
10. Run memory lint when memory files changed.
11. Explain the result simply.

## Council Integration

The existing AI Council flow remains valid:

- briefs live in `.ai/briefs/`;
- proposals live in `.ai/proposals/`;
- reviews live in `.ai/reviews/`;
- accepted ADRs live in `Docs/adr/`;
- durable lessons live in `.ai/memory/project.md`.

The new memory layer does not replace the Council flow. It gives agents a
stable place to read current Automixer goals, safety policy, active decisions,
experiments, and session reports before using the Council artifacts.

## Stop Conditions

Stop and report instead of executing when:

- the task requires live OSC/WING writes without explicit approval;
- the task would overwrite unrelated local changes;
- the task needs secrets or credentials;
- the safe rollback path is missing for a supervised write;
- current evidence contradicts the requested action.
