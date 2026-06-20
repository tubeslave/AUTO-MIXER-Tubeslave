# ADR: Unmasking, frequency and balance context gates

## Context

M083 made source identity and arrangement-role candidates more trustworthy, but
downstream mix-decision modules still need their own proof boundary. Cross-
channel unmasking, frequency separation and bus/group balance are closer to
audible mix changes than classification. They must not emit proposal candidates
from stale, missing or live-authority context.

These modules advance the long-term AI Mixing Engineer goal because they map
musical context into EQ, spectral-lane and balance decisions. They still cannot
grant live console authority or replace real WING proof.

## Options Considered

1. Keep downstream modules permissive and rely on source/role gates alone.
2. Require each downstream proposal family to verify fresh read-only/no-authority
   context before emitting proposals.
3. Promote unmasking, frequency separation or bus balance into queueable live
   actions after focused unit tests.

## Decision

Require cross-channel unmasking, frequency separation and bus/group balance
proposal candidates to consume fresh read-only/no-authority context. Missing,
stale or live-authority context keeps the result analysis-only. Cross-channel
unmasking also consumes arrangement-role priority evidence before deciding which
channel should be protected and which channel should move.

## Why This Won

This is real capability progress, not another WING-report polish: the project
moves from identifying sources and roles into safer downstream mix decisions.
The safety boundary stays intact because the outputs remain proposal-only,
readback/rollback-bound and unable to contact WING or apply live changes.

## Rejected Alternatives

- Relying only on M083 would leave downstream modules able to interpret weak or
  live-authority context as proposal evidence.
- Promoting these families to live apply would skip M007/M008 replay gates,
  hardware proof, readback, rollback, safety-kernel integration and operator
  approval.

## Implementation Plan

- Add no-authority freshness checks to cross-adaptive EQ masking context.
- Require fresh read-only/no-authority arrangement context for cross-channel
  unmasking and derive effective priorities from arrangement/source roles.
- Add no-authority freshness checks to frequency separation context.
- Add no-authority freshness checks to bus/group/DCA balance context.
- Update proposal-only contract evidence requirements.
- Add tests for fresh context, missing context and live-authority context.
- Record M084 evidence while keeping M001/M002 blocked on real WING/operator
  validation.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_cross_adaptive_eq.py tests/test_frequency_separation_proposal.py tests/test_bus_group_balance_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m084_unmasking_frequency_balance_context_gates_stay_readonly -q`
- Standard backend suite.
- Memory lint and JSON/JSONL validation.

## Risks and Rollback

Risk: proposal candidates become more conservative until fresh read-only context
is available. This is acceptable because these families can lead directly to
audible EQ and balance decisions.

Rollback: remove the downstream no-authority checks, the new tests and M084
state/report updates. This does not affect live transport because the change is
offline/proposal-only.

No live transport, no OSC write, no proposal apply, no readiness closure and no
live authority are introduced by this decision.
