# Decision Engine v2

## 1. Context

AUTO-MIXER-Tubeslave has working live control, audio analysis, heuristics, offline rendering, and evaluation modules, but decisions are spread across multiple places. The project needs a clear boundary between analysis, knowledge, criticism, decisions, safety, execution, and logging.

## 2. Options considered

- Keep extending `backend/auto_soundcheck_engine.py`.
- Promote `mix_agent` to the live decision layer.
- Add a separate v2 orchestration layer under `automixer/`.

## 3. Decision

Add a separate, opt-in v2 layer under `automixer/`.

## 4. Why this won

It preserves the working live pipeline, avoids heavy ML in live mode, and creates a small, testable contract: Decision Engine produces `ActionPlan`; Safety Gate filters; Executor sends only gated actions.

## 5. Rejected alternatives

Extending the old soundcheck engine directly would deepen the current coupling. Making `mix_agent` mandatory for live would mix offline evaluation, rules, and optional ML/LLM behavior into the live path.

## 6. Implementation plan

- Add role packages under `automixer/`.
- Add knowledge JSON for instrument categories.
- Add `DecisionEngine`, `SafetyGate`, `ActionPlanExecutor`, human logs, and offline experiment runner.
- Add disabled-by-default config.
- Add CLI flags without changing default live behavior.

## 7. Test plan

- Unit tests for `ActionPlan` creation.
- Unit tests for Safety Gate blocking/rate limiting/dry-run behavior.
- Executor dry-run test proving no mixer writes.
- Offline experiment report test.
- Config loading test.
- Legacy default flag test.

## 8. Risks and rollback

Risk: v2 may duplicate concepts already present in AutoFOH safety. Rollback is straightforward: leave `decision_engine_v2.enabled: false` and do not pass `--use-decision-engine-v2`. No legacy module is removed.
