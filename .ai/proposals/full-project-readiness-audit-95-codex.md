# Proposal: full project readiness audit to 95, Codex

## 1. Short thesis

Treat release truth and live authority as the first repair batch. Demote every
normal supervised write kind that is not bound to an accepted owner/readiness
decision, make hardware proof tools fail closed before transport creation, and
isolate test/audit evidence before touching lower-severity UX or cleanup work.

## 2. What I understood

The repository has strong unit coverage and a large amount of real-WING work,
but the current dirty M006 tail combines implementation, proof output, runtime
permission, and release claims. Passing tests cannot compensate for a live
authority bypass or mutable evidence contaminated by tests.

## 3. Proposed solution

1. Demote static family contracts to `hardware_proven` or `proposal_only` with
   no normal live kinds until a validated owner promotion input exists.
2. Add a non-bypassable execute/typed-confirmation gate to the local family
   proof runner. A successful proof remains proof, not owner acceptance.
3. Isolate all pytest manual-write artifacts under `tmp_path` while preserving
   persistence behavior inside each test.
4. Make multi-address apply failure and adapter exceptions roll back the
   current partial operation, restore previous operations, and activate the
   system emergency stop when restoration cannot be verified.
5. Fix operator UI truth: connection recovery state, evidence-bound M005
   status, full family blockers/approval policy, per-run reset, and active
   channel selection.
6. Replace split-brain status text and mutable-log evidence references with
   run-scoped, hash-bound evidence and explicit remaining external gates.

## 4. Likely files to touch

- `backend/supervised_console_parameter_registry.py`
- `backend/supervised_console_transaction.py`
- `backend/wing_supervised_console_adapters.py`
- `tools/wing_supervised_family_proof_runner.py`
- `tests/conftest.py` and focused regression tests
- `frontend/src/services/useOperatorBackend.ts`
- `frontend/src/views/SoundcheckView.tsx`
- M006/active milestone truth surfaces and a final audit report

## 5. Alternatives considered

- Keep the current live registry and trust physical proof summaries. Rejected:
  those summaries explicitly deny live authority and do not replace owner
  promotion.
- Delete the new transaction kernel. Rejected: the architecture is useful; the
  bug is promotion and failure semantics, not the existence of adapters.
- Disable all audit persistence in tests. Rejected: tests should still exercise
  persistence, only inside temporary directories.

## 6. Risks

- Demotion will intentionally make current normal supervised M006 transactions
  non-executable until owner promotion is represented safely.
- Error-path hardening can expose tests that assumed optimistic partial apply.
- Truth sync must preserve historical real-WING evidence without converting it
  into standing authority.

## 7. Test plan

- Focused registry/queue/server/coordinator/adapter/proof-runner regressions.
- Assert no focused or full pytest run changes the canonical manual-write log.
- Full backend suite, no-WING quick gate with a temporary artifact root,
  frontend tests/typecheck/build, and Playwright/browser smoke.
- JSON, JSONL, diff, and documentation cross-surface checks.

## 8. Where the other agent may disagree

An independent reviewer may prefer deleting broad live adapters or preserving
fader/gain normal authority. My recommendation is narrower: keep all semantic
adapters, but permit normal execution only from an explicit validated promotion
source rather than a static code constant.
