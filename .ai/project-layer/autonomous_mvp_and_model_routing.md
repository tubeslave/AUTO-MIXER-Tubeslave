# Autonomous MVP And Model Routing

Status: active for Codex Goal-mode automation

## Preserved Project Goal

The end goal remains a professional AI Mixing Engineer for live and studio use:
automatic analysis, source/role detection, gain staging, EQ, dynamics,
de-essing, panning, frequency separation, cross-channel unmasking, phase
correction, FX, balance, master-bus optimization, adaptation during a show, and
eventual controlled console operation.

The current autonomous development MVP is narrower because live sound safety and
tariff limits matter. Codex should advance the full goal by finishing one safe
hardware proof rail first.

## MVP Now

The active MVP is still `Docs/MVP_WING_SUPERVISED.md`:

1. Prove WING read-only connection.
2. Prove physical console readback.
3. Prove supervised fader write with operator approval.
4. Prove supervised gain-down / trim with operator approval.
5. Prove rollback.
6. Prove emergency stop.
7. Prove connection-loss / timeout fail-closed behavior.
8. Produce an operator-visible evidence report.
9. Update readiness only from accepted real proof.

Everything else from the long-term goal continues as offline, dry-run,
proposal-only, read-only, or test-only work until this proof is accepted.

## Automation Strategy

Automated Codex cycles must optimize for steady progress, not maximum model
spend.

- Run one bounded cycle and exit.
- Use `tools/codex_autoloop/project_director_agent.py` to select the first
  non-closed MVP blocker.
- Use `tools/codex_autoloop/check_model_budget.py` before `codex exec`.
- Use the selected model profile from
  `tools/codex_autoloop/model_budget_profile.json`.
- Prefer lower-cost evidence gathering and testable scoped diffs.
- Skip a cycle when local daily/weekly/monthly profile caps are exhausted.
- Keep high-reasoning GPT-5.5 for rare architecture/safety review windows.

## Virtual Subagent Routing

| Subagent | Model profile | Main job | Output |
| --- | --- | --- | --- |
| Director | budget_balanced | Select the next MVP blocker and reject drift | Cycle task direction |
| Scout | budget_light | Read repo state, reports, TODOs, and evidence | Short factual summary |
| Implementer | budget_balanced | Make a small patch tied to the selected blocker | Code/docs/tests |
| Safety Reviewer | budget_strict | Check fader/gain-only boundary, rollback, fail-closed, approval | Safety notes in report |
| Heavy Architect | premium_high | Rare full architecture review | ADR/proposal only |

These are not separate simultaneous writers. They are role lenses inside the
single Codex cycle unless a human explicitly creates separate worktrees.

## Limit Protection

The local budget gate protects the user's tariff by enforcing conservative
reservation caps:

- `budget_light`: routine evidence and docs.
- `budget_balanced`: normal scoped implementation.
- `budget_strict`: safety review.
- `premium_high`: rare GPT-5.5 high-reasoning architecture review.

Provider-side usage can still differ from the local ledger. If the provider
quota is already exhausted, Codex should resume at the next available window and
continue with the first open MVP blocker.

## Safety Boundary

No automation setting here permits:

- `danger-full-access` as the required operating mode.
- `--yolo`.
- Live OSC/WING writes outside the existing supervised fader/gain proof path.
- EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA, or full
  autonomous writes before separate ADR plus real hardware proof.
