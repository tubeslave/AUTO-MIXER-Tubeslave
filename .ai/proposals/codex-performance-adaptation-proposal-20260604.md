# Codex Proposal: Performance Adaptation Proposal Layer

## 1. Short thesis

Move musician-level adaptation, live adaptation, and real-time correction rows
from shadow-only to Assisted proposal support by adding a transport-free,
rate-limited level-drift proposal contract.

## 2. What I understood

The Automixer needs to react when musicians get louder or quieter during a show.
That cannot imply autonomous live writes yet. The useful next step is to create
an evidence-rich proposal layer that lets the system explain a safe candidate
ride while preserving the existing supervised fader/gain boundary.

## 3. Proposed solution

Create `backend/performance_adaptation_proposal.py` with:

- target normalization for channels, stems, groups, buses, DCA, and matrices;
- baseline-vs-current level drift calculation;
- bounded upward/downward deltas with true-peak reserve and fader ceiling;
- blockers for missing baseline/readback/rollback, missing operator intent,
  manual override, low persistence, cooldown, latency, feedback, bleed, and
  insufficient confidence;
- Product Layer style proposal output with `kind="performance_adaptation"`,
  `can_apply_live=False`, `can_import_to_queue=False`, and no
  `requested_change`.

Register the method as proposal-only and attach it to the project-goal rows for
musician-level adaptation, concert-time adaptation, and real-time corrections.

## 4. Likely files to touch

- `backend/performance_adaptation_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_performance_adaptation_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/performance-adaptation-proposal-20260604.md`

## 5. Alternatives considered

- Keep adaptation shadow-only: safe, but it does not provide an operator-facing
  path toward the requested autonomous behavior.
- Emit real fader/gain proposals immediately: rejected because performance
  adaptation needs replay validation and rate-limit proof before queue import.
- Apply rides directly through `auto_fader`: rejected because that would widen
  live authority without the required proof.

## 6. Risks

- Operators may mistake an adaptation proposal for an apply-ready ride.
- Upward adaptation can amplify bleed or feedback if evidence is weak.
- Realtime proposals may be noisy without persistence and cooldown guards.

## 7. Test plan

- Verify the module is free of mixer transport and apply calls.
- Verify persistent loud-source drift creates a proposal-only downward
  recommendation.
- Verify upward drift is blocked by headroom, feedback, and bleed risks.
- Verify missing baseline/readback/rollback/manual intent blocks proposals.
- Verify registry, proposal contract, and capability matrix wiring.

## 8. Where the other agent may disagree

Kimi may argue live adaptation should stay shadow-only until replay scoring is
implemented. The counterpoint is that proposal-only output is not promotion; it
defines the evidence contract replay scoring must satisfy later.
