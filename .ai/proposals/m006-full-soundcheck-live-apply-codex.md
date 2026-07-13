# Codex proposal: M006-FULL soundcheck live apply

## 1. Short thesis

Расширять нужно не список разрешённых строк в `operator_mode_policy.py`, а
канонический Product Layer transaction kernel. Все soundcheck typed actions
должны превращаться в строго типизированные параметрические транзакции,
исполняемые только через `apply_operator_proposal`; readiness каждой семьи
активируется отдельным hardware proof.

## 2. What I understood

Владелец ожидает, что реальный саундчек сможет отправлять на WING не только
fader/gain-down, но EQ, dynamics, gate, spatial, FX, buses, routing, scenes и
master. DSP-расчёты и большая часть transport wrappers уже существуют. Сейчас
широкие typed actions либо proposal-only, либо идут через отдельный
`GatedCorrectionActionExecutor`, а канонический транзакционный manual kernel
умеет только scalar `fader/gain/mute/pan`.

Нельзя просто добавить новые строки в `WING_DEPLOYMENT_LIVE_WRITE_KINDS`: это
разрешит actions, у которых нет унифицированного pre-read, post-read,
compare-and-swap rollback, emergency-stop и audio recheck.

## 3. Proposed solution

### A. Parameter contract registry

Добавить `backend/supervised_console_parameter_registry.py` с immutable
контрактами:

- family and write kind;
- target scope and WING address resolver;
- value type/range/quantization/tolerance;
- typed transport method adapter;
- required pre-read and post-read addresses;
- rollback order and destructive-risk class;
- direction policy and required evidence;
- audio-recheck policy;
- current lifecycle: `proposal_only`, `proof_candidate`, `hardware_proven`,
  `live_allowed`.

Registry должен описывать все перечисленные семьи, но live allowance получает
только из accepted readiness artifacts, а не из наличия transport method.

### B. Composite transaction kernel

Добавить `backend/supervised_console_transaction.py`:

1. validate operator mode, approval, idempotency and family readiness;
2. resolve exact parameter operations;
3. fresh physical pre-read every touched address;
4. capture immutable rollback snapshot;
5. re-check emergency generation immediately before every send/retry;
6. apply operations in dependency order;
7. fresh physical post-read every address;
8. run family-specific semantic validation;
9. run required Main LR audio recheck;
10. on any failure rollback in reverse order with compare-and-swap;
11. emit one transaction artifact and one operator result.

Existing `_apply_manual_console_write` becomes a compatibility adapter for
single-parameter fader/gain/mute/pan transactions. No second writer appears.

### C. Proposal and soundcheck integration

- Map existing `TypedCorrectionAction` objects to transaction proposals.
- AutoSoundcheck remains derivation-only and queues absolute targets.
- Add a deterministic apply plan ordered by risk/dependency:
  gain-down -> HPF -> gate -> cut-only EQ -> makeup-neutral dynamics ->
  polarity/delay -> pan -> fader -> sends/FX -> buses/DCA -> master.
- Mute, routing and scenes are separate high-risk transactions, never silently
  folded into a channel-strip batch.
- No relative deltas in repeating cycles; every proposal binds old value,
  absolute target and evidence timestamp.

### D. Frontend

Expose per-family lifecycle, blockers, exact affected addresses, risk,
transaction steps, physical readback, rollback availability and audio recheck.
The Apply control is enabled only when backend preflight returns ready for that
exact proposal and family.

### E. Promotion sequence

Implementation and proof order:

1. preserve proven fader/gain rail and land M005 fixes;
2. scalar channel controls: mute/pan/polarity/delay;
3. HPF and cut-only EQ;
4. gate threshold;
5. compressor threshold/ratio with makeup neutral;
6. FX/send/return;
7. buses/groups/DCA/matrix;
8. master output processing;
9. routing and scene/preset last.

Each step gets unit/replay/fault proof before one low-risk real-WING proof.

## 4. Likely files to touch

- `backend/supervised_console_parameter_registry.py` (new)
- `backend/supervised_console_transaction.py` (new)
- `backend/server.py`
- `backend/handlers/product_state_handlers.py`
- `backend/operator_mode_policy.py`
- `backend/auto_soundcheck_engine.py`
- `backend/correction_action_executor.py`
- `backend/autonomy_promotion_readiness.py`
- `backend/safety_coverage_matrix.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- targeted tests plus M006-FULL hardware proof runner

## 5. Alternatives considered

### Enable all existing direct setters

Rejected. It preserves multiple writers and inconsistent proof semantics.

### Promote every `GatedCorrectionActionExecutor` action

Rejected. It is a parallel live path and lacks Product Layer proposal,
idempotency and transaction audit semantics.

### One giant scene/snapshot apply

Rejected. Destructive blast radius is too large and per-parameter rollback and
root-cause attribution disappear.

### Keep everything proposal-only indefinitely

Rejected by the product requirement. Proposal-only remains the default only
until each family completes staged proof.

## 6. Risks

- Composite partial apply across several addresses.
- OSC quantization/model-specific paths causing false mismatch.
- Audio recheck taps not hearing pre-trim or some control surfaces.
- Feedback, monitor-send and PA-output risk.
- Existing direct handlers bypassing the new kernel.
- Large blast radius if all families are enabled together.

Mitigation: immutable registry, reverse rollback, family proof flags, explicit
high-risk separation, static write-surface scan and staged promotion.

## 7. Test plan

- Registry completeness and duplicate-address validation.
- Per-family bounds, direction, model and risk tests.
- Transaction unit tests for success, send failure, timeout, stale/cache-only
  readback, quantization, mismatch, partial apply and reverse rollback.
- Emergency-generation change between steps.
- Idempotency and concurrent-writer rejection.
- Existing fader/gain regression suite.
- AutoSoundcheck proposal-only and no-direct-write assertions.
- no-WING release gate remains fail-closed.
- Virtual/replay scenarios, then one real-WING proof artifact per family.
- Main LR audio A/B where the surface affects the audible master.
- Full backend/frontend/E2E suite at milestone boundaries.

## 8. Where the other agent may disagree

An opposing reviewer may prefer to extend `_apply_manual_console_write`
directly instead of adding a transaction module, or may exclude routing/scenes
from the same milestone because their destructive scope differs from DSP
parameters. My recommendation is one kernel but separate high-risk transaction
classes and later proof order, so the user-visible goal remains complete
without pretending every surface has equal risk.
