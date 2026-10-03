# Proposal: Decision Engine v2

## 1. Short thesis

Add v2 as an opt-in orchestration layer, not a rewrite: it should consume current analyzer/rule/critic outputs, produce explainable `ActionPlan` objects, pass them through a separate Safety Gate, and only then let an Executor translate safe actions to mixer calls.

## 2. What I understood

The project already has analysis, rules, ML experiments, critics/evaluation, and OSC/mixer clients. The missing piece is a clean decision boundary that prevents all modules from becoming one mandatory live pipeline.

## 3. Proposed solution

- Create role packages in `automixer/`: `analyzer`, `knowledge`, `critics`, `decision`, `safety`, `executor`, `experiments`, `logs`.
- Keep backend and mix-agent code intact.
- Add `DecisionEngine` that emits an `ActionPlan` with reason, confidence, risk, source modules, expected effect, and `safe_to_apply`.
- Add `SafetyGate` with configurable gain/EQ/pan/compression bounds, live-mode restrictions, rate limiting, hysteresis, cooldown, dry-run, and emergency bypass.
- Add a small `ActionPlanExecutor` that delegates to mixer clients only after Safety Gate approval.
- Add an offline experiment runner for rules-only, rules+critic, rules+critic+decision-engine, and decision-engine dry-run variants.
- Add flags to the soundcheck CLI without changing defaults.

## 4. Likely files to touch

- New: `automixer/**`
- New: `Docs/ARCHITECTURE_V2*.md`, `Docs/DECISION_ENGINE.md`, `Docs/SAFETY_GATE.md`, `Docs/OFFLINE_EXPERIMENTS.md`
- Update: `backend/auto_soundcheck_engine.py`
- Update: `config/automixer.yaml`
- New tests under `tests/`

## 5. Alternatives considered

- Wiring existing `backend/ai/RuleEngine`, `AutoFOHSafetyController`, and `MixingAgent` into one live path. Rejected because it makes ML/critic/rules mandatory and increases live risk.
- Replacing `AutoFOHSafetyController`. Rejected because legacy safety is already used and tested; v2 should be a new gate for v2 plans, not a destructive replacement.

## 6. Risks

- Duplicate safety concepts can confuse future contributors.
- If v2 live apply is enabled without enough context, it may be too conservative or block useful corrections.
- Existing legacy modules use mixed naming (`leadVocal`, `backVocal`, `electricGuitar`), so v2 must normalize roles carefully.

## 7. Test plan

- Focused pytest for v2 decision engine, safety gate, executor dry-run, experiment reports, config loading, and legacy default flags.
- Standard project command before merge:
  `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer deeper integration into `backend/autofoh_safety.py` to avoid two safety layers. My recommendation is to keep v2 isolated first because live sound safety and legacy stability matter more than elegance in this step.
