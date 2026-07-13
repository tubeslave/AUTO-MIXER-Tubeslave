# Independent Codex proposal: immutable batch manifest, not a stateful runner

This is an independent Codex-subagent fallback. Kimi CLI returned HTTP 402, so
this document is not presented as a Kimi opinion.

## 1. Short thesis

The defect comes from treating musical stage, operator approval and rollback
atomicity as the same boundary. Fix the rollback boundary first: publish an
immutable parent soundcheck manifest and one child `ConsoleTransactionPlan` per
rollback-coupled semantic operation. Stage is ordering/readiness metadata, not
an atomic transaction. Keep the existing coordinator unchanged and fail-closed.

## 2. What I understood

The planner currently creates one normal plan, coordinator preflights every
operation before the first write, queue applies one `can_apply` result to the
whole plan, and the engine can announce completion without mix-wide readiness.
Current standing live authority is still zero, so this is an architecture/truth
correction, not permission to write to WING.

## 3. Proposed solution

Add a parent manifest containing `run_id`, revision, stable content digest,
ordered child plans, unresolved actions and readiness. Each child carries its
stage, order, `required_for`, dependencies, origin correction ID and batch
revision.

Rules:

- One child is one semantic operation unless an explicit dependency requires
  joint rollback.
- Global gates are rechecked before each child; a global failure pauses the
  batch.
- A local transition failure creates a structured residual for that child and
  does not erase independent siblings.
- `already_satisfied` requires physical readback.
- `split_required` returns to the planner. The transport layer must not invent
  intermediate targets, because repeated bounded steps could launder a
  cumulative limit.
- A musical replacement requires fresh analysis, a new manifest revision and
  a new approval digest.
- Readiness uses `required_for`, not the number of successful writes.

## 4. Likely files to touch

- `backend/soundcheck_transaction_plan.py`
- `backend/soundcheck_mix_readiness.py`
- `backend/operator_recommendation_bridge.py`
- `backend/auto_soundcheck_engine.py`
- status handlers and frontend soundcheck display
- focused regression tests

The queue/server do not need a broad batch runner in the first P1 because every
child already travels through the canonical apply entrypoint and its global
gate independently.

## 5. Alternatives considered

- Stateful stage runner now: rejected until durable batch WAL, crash resume,
  approval invalidation and fresh preflight semantics exist.
- Coordinator skip-on-error: rejected because it weakens atomicity.
- Automatic musical substitutions in transport: rejected because they require
  global mix analysis.
- Constructive fader/send promotion: rejected because current checks are
  post-write/static-route and lack trusted feedback-clear preflight.

## 6. Risks

- More child proposals can overwhelm the UI. Suppress duplicate generic
  applyable proposals and aggregate the manifest visually.
- A partial sequence can leave an intentionally partial mix. Publish it as
  partial/incomplete with exact residuals.
- Stable content hashes can collide across runs in the queue. Combine a unique
  run ID with stable content fingerprint.
- A child marked applied can later have rolled back. Readiness must require
  terminal committed status, not populated applied-operation IDs.

## 7. Test plan

- M006 51-operation shape decomposes into independent children; no normal
  monolith remains.
- Offline snapshot classifier reproduces 36 admissible, 15 local residual and
  12 already-satisfied.
- Safe sibling survives a local blocker; global blocker yields zero writes.
- Manifest revision changes after a replacement; old approval cannot resolve
  it.
- No duplicate apply surfaces for one logical correction.
- Required residual prevents both ready statuses.
- Live allowlists remain unchanged; no-WING checks remain green.

## 8. Where the other agent may disagree

The primary proposal may want the full scheduler immediately. This proposal
argues that the first merge should establish the manifest, child boundary and
truth contract first; a durable scheduler can later consume that stable model.
