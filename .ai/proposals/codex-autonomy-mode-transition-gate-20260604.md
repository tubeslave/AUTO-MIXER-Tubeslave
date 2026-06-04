# Codex Proposal: Autonomy Mode Transition Gate

Task id: autonomy-mode-transition-gate-20260604
Date: 2026-06-04

## 1. Short thesis

The four target modes need an explicit gate that can say "available",
"limited", or "blocked" from current proof evidence without changing runtime
state.

## 2. What I understood

Operator mode policy already defines Analysis, Assisted, Supervised, and Full
Autonomous. Static readiness and proof overlay now describe promotion evidence.
The missing part is a tested, machine-readable bridge that explains whether
the current proof state permits entry, only narrow supervised operation, or no
autonomous path.

## 3. Proposed solution

Add `autonomy_mode_transition_gate`, a read-only builder that combines:

- `operator_mode_policy`;
- `autonomy_promotion_readiness`;
- `autonomy_evidence_overlay`.

It marks Analysis and Assisted available, Supervised limited to current narrow
policy when static live scope exists, and Full Autonomous blocked. It does not
change the runtime mode.

## 4. Likely files to touch

- `backend/autonomy_mode_transition_gate.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `backend/processing_registry.py`
- focused tests and ADR/brief artifacts

## 5. Alternatives considered

- Put this logic inside `operator_mode_policy`.
- Put this logic inside static readiness.
- Let the UI infer mode availability from separate snapshots.

## 6. Risks

The word "available" could be misread as permission to widen live authority.
The gate must distinguish runtime entry from mode expansion and keep
`live_widening_allowed=false`.

## 7. Test plan

- Unit-test Analysis/Assisted availability and Full Autonomous blocking.
- Unit-test Supervised narrow-limited behavior from physical/remediation proof.
- Assert proof report embedding and registry shadow-only status.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may prefer to make Supervised fully blocked while proof remediation is
pending. The current repo already has a narrow supervised fader/gain runtime
policy, so the gate should preserve that fact while blocking expansion.
