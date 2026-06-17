# Brief: WING MVP Accepted-Candidate Request-Sequence Trace Value-Gate Summary Surface

## Task

Document and validate the no-WING proof-plan surface that exposes the accepted
candidate request-sequence trace value gate in `summary.json` and operator
Markdown.

## Context

The selected blocker remains `WING read-only connection proof`. Before any
supervised fader/gain proof, the no-WING packet must prove that accepted review
source ids have a valid request-sequence trace matching the expected plan. The
proof must fail closed when the value-gate summary surface is missing or
mismatched.

## Required Fields

- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_required`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_violation_ids`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_violations`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_self_check_required`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_required_fields`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_mismatches`
- `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_match`

## Safety Boundary

This is an offline no-WING/reporting contract. It must not contact WING, start
the backend, query console state, mutate console state, grant live authority, or
close readiness.

## Test Plan

- Compile `tools/no_wing_release_check.py` and the focused tests.
- Run the focused request-sequence value-gate summary-surface regression.
- Run no-WING quick into a temporary artifact root.
- Verify the no-WING artifact manifest.
- Run the focused no-WING/WING preflight/Project Director suite.
- Run the full pytest suite before commit.
