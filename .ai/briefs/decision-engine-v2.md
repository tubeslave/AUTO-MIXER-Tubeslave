# Brief: Decision Engine v2

## Task

Implement a safe v2 architecture layer for AUTO-MIXER-Tubeslave:

Analyzer -> Knowledge/Rules -> Critic -> Decision Engine -> Safety Gate -> Executor -> Logs/Test Harness.

## Constraints

- Preserve the existing live/OSC pipeline unless explicitly enabled.
- Do not delete legacy modules.
- Do not add heavy ML dependencies to the live path.
- Decision Engine must only produce an ActionPlan; it must not send OSC.
- Any live write must pass through a Safety Gate.
- Default behavior remains disabled and conservative.

## Inputs

- Existing AutoFOH analysis/safety modules in `backend/`.
- Existing offline mix-agent modules in `mix_agent/` and `tools/offline_agent_mix.py`.
- Local external references:
  - `external/AutomaticMixingPapers`
  - `external/automix-toolkit`
  - `external/FxNorm-automix`

## Output

- Audit and v2 docs under `Docs/`.
- New role-oriented packages under `automixer/`.
- Config-gated v2 runtime.
- Offline experiment harness with JSON and Markdown reports.
- Tests for decision plans, safety, dry-run executor behavior, reports, defaults, and config loading.
