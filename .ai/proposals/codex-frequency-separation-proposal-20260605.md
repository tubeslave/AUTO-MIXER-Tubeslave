# Proposal: frequency-separation-proposal-20260605

## Short thesis

Add a proposal-only frequency separation analyzer that turns source/stem
contribution evidence into explainable frequency-lane recommendations without
creating live EQ or fader authority.

## What I understood

The target Automixer needs to decide which instruments should own which
frequency lanes before it can safely derive EQ, fader, or arrangement moves.
`cross_adaptive_eq` now handles linked unmasking pairs, but frequency separation
needs a higher-level proposal record that explains source/stem contribution and
role priority.

## Proposed solution

Create `backend/frequency_separation_proposal.py` with transport-free dataclasses
and `analyze_frequency_separation_need`. The analyzer should:

- normalize optional current-state freshness;
- block stale provided freshness;
- compare active source band energies by source role, priority, confidence, and
  stem contribution;
- protect the stronger lane owner and propose a bounded subtractive split for
  the supporting/competing source;
- return Product Layer style proposals with no queue import and no live apply.

Register the method in `processing_registry`, expose a `frequency_split`
proposal-only contract, and update the `frequency_separation` capability row.

## Likely files to touch

- `backend/frequency_separation_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_frequency_separation_proposal.py`
- `.ai/briefs/frequency-separation-proposal-20260605.md`
- `Docs/adr/frequency-separation-proposal-20260605.md`

## Alternatives considered

- Reuse `cross_adaptive_eq` only. Rejected because it produces pairwise mirror EQ
  candidates, not a source/stem lane contract for frequency separation.
- Emit direct EQ proposals. Rejected because this layer should explain ownership
  of frequency lanes before deriving concrete EQ moves.
- Require freshness for legacy missing context. Rejected for compatibility with
  offline callers until upstream context freshness is wired everywhere.

## Risks

- The heuristic role-to-band affinity is intentionally conservative and will
  need replay tuning against real sessions.
- Missing freshness remains a compatibility gap until source/stem spectrum
  snapshots are consistently provided upstream.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/frequency_separation_proposal.py \
  backend/processing_registry.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_frequency_separation_proposal.py
PYTHONPATH=backend python -m pytest \
  tests/test_frequency_separation_proposal.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue the analyzer should produce concrete EQ-band proposals
immediately. I would keep this first layer as a lane decision: it is easier to
review, can feed EQ/fader/pan proposals later, and does not imply live authority.
