# Proposal: WING Autoloop Handoff Gates Docs

Date: 2026-06-12

## Short thesis

Keep Codex autoloop instructions aligned with the current WING handoff gate
contract.

## What I understood

Future autonomous cycles read `tools/codex_autoloop/codex_cycle_prompt.md` and
`tools/codex_autoloop/README.md`. Those files should carry the same no-WING
handoff fields that Project Director now requires before supervised proof
planning.

## Proposed solution

- Add blocker-anchor match, anchor digest match, digest-review operator step,
  supervised gate presence/finality, and accepted-candidate prestart trace
  requirements to the autoloop prompt.
- Mirror the same requirements in the autoloop README.
- Extend tests so prompt and README stay aligned.
- Record brief, ADR, and validation report.

## Likely files to touch

- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `Docs/INDEX.md`

## Alternatives considered

- Rely only on the blocker table. Project Director reads it, but the wrapper
  prompt still frames autonomous cycle behavior before task selection.
- Rely only on the no-WING validator. That catches invalid artifacts later but
  does not keep cycle instructions current.

## Risks

- Prompt/README become more verbose.
- Future field renames must update the prompt, README, and tests together.

## Test plan

- Focused Project Director and WING MVP rails tests.
- Project Director and direction guard JSON smoke checks.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may argue that Project Director selected action is enough. I would still
update the autoloop docs because they are the top-level cycle contract for
non-interactive runs.
