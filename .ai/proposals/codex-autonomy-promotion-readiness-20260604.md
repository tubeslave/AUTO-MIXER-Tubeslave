# Codex Proposal: Autonomy Promotion Readiness Snapshot

## 1. Short thesis

After proposal coverage is in place, add a proof-gate snapshot that makes the
path from proposal-only to supervised and autonomous control measurable and
fail-closed.

## 2. What I understood

The Automixer target is full autonomous operation, but the current checkout only
has a narrow supervised fader/gain executor and partial physical proof. The
system needs an explicit readiness view that prevents proposal-only coverage
from being mistaken for live authority.

## 3. Proposed solution

Create `backend/autonomy_promotion_readiness.py` with rows for fader, gain,
EQ, dynamics, gate, pan/phase/delay, FX, bus/group/matrix, mute, and
routing/scene/preset promotion. Each row combines:

- project capability status;
- safety proof status;
- write-surface policy;
- read-surface coverage;
- blockers and next required evidence.

Expose the snapshot from `build_processing_registry_snapshot()` and update the
project-goal matrix next step to proof-gated promotion.

## 4. Likely files to touch

- `backend/autonomy_promotion_readiness.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_autonomy_promotion_readiness.py`
- `tests/test_processing_registry.py`
- `Docs/adr/autonomy-promotion-readiness-20260604.md`

## 5. Alternatives considered

- Promote the next action family directly: rejected because proof gaps are still
  broad and must be visible first.
- Keep the readiness information spread across existing inventories: safe, but
  it makes autonomous-readiness claims hard to audit.
- Add readiness as docs only: rejected because tests should enforce the
  fail-closed invariants.

## 6. Risks

- Readiness rows could be read as permission if status names are vague.
- Existing capability rows declare fader/gain executor support, while safety
  proof only allows narrow fader. The readiness layer must surface that mismatch
  rather than hiding it.
- Full Autonomous must not become ready just because proposal coverage exists.

## 7. Test plan

- Verify the readiness module is free of mixer transport and apply calls.
- Verify Full Autonomous readiness is false.
- Verify fader is supervised-only, not autonomous-ready.
- Verify gain is blocked by missing safety proof despite executor support.
- Verify broad console surfaces remain blocked by proof/read coverage/manual
  policy.

## 8. Where the other agent may disagree

Kimi may argue the next step should be a physical gain proof instead. The
counterpoint is that the readiness snapshot makes exactly that proof dependency
testable while keeping all broad surfaces fail-closed.
