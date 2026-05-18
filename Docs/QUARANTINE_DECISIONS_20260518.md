# AUTO-MIXER Quarantine Decisions

Date: 2026-05-18
Status: quarantine resolved for visible `automixer/logs` orphan files

No WING/OSC writes, Paperclip dispatches, issue mutations, or live runtime
activation were performed.

## Decision

Remove the untracked `automixer/logs` files from the working tree instead of
restoring the old Decision Engine v2 source tree.

Files removed from the working tree:

- `automixer/logs/__init__.py`
- `automixer/logs/human_logger.py`

## Reason

These files became visible only after `.gitignore` was fixed to stop treating
nested `logs` directories as generated output.

They are not self-contained in the current checkout:

```text
ModuleNotFoundError: No module named 'automixer.decision.models'
```

They depend on the old `automixer.decision.models` module from the historical
Decision Engine v2 work. The old v2 source exists in git history at commit
`910e877`, but restoring it is a much larger architecture decision than keeping
two logger helpers.

## Search Result

Current source references were limited to the orphan files themselves:

- `HumanDecisionLogger`
- `format_action_human`
- `automixer.decision.models.ActionDecision`
- `jsonable`

No current Product Layer, backend runtime, frontend, or test file imports
`automixer.logs`.

## Recovery Path

If Decision Engine v2 is intentionally restored later, restore it as a complete
slice from commit `910e877`, including:

- `automixer/decision/`
- `automixer/analyzer/`
- `automixer/safety/`
- `automixer/executor/`
- `automixer/knowledge/`
- related v2 docs and tests

Do not restore only `automixer/logs/*.py`; that produces a broken package.

## Verification

After removing the orphan files, run:

```bash
git status --short --branch --untracked-files=all
PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py -q
```
