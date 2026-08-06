# ADR: Offline Fail-Closed Remediation

## 1. Context

An offline review found multiple reachable paths that reported success or
plausible runtime state without the evidence needed to support those claims.
The repository controls live-sound behavior, and the current checkout has no
authority to contact or mutate a real console.

## 2. Options considered

1. Leave the paths quarantined but unchanged.
2. Implement broad missing DSP and hardware integrations using simulated data.
3. Replace false success with real offline inspection and explicit unavailable
   results, preserving all live and hardware gates.

## 3. Decision

Choose option 3. Unsupported control readback fails closed; PyAudio discovery
never invents devices; absent automation returns a typed unavailable contract;
duplicate handlers are rejected; agent-ops inspect actual files; fake genre
weighting is disabled; EQ proposals require measured, non-synthetic evidence.

## 4. Why this won

It closes concrete software defects without claiming physical proof, adding
dependencies, changing console mappings, or widening the current empty live
allowlist.

## 5. Rejected alternatives

Simulated runtime devices and constant quality scores were rejected because
they can be confused with real evidence. Broad DSP implementation was rejected
because it cannot be validated safely in this no-console cycle.

## 6. Implementation plan

- Add fail-closed guards at each confirmed false-success boundary.
- Preserve existing response type names where clients may depend on them.
- Add regression tests that assert authority and readiness remain denied.
- Leave hardware-only providers and physical proof gates explicitly open.

## 7. Test plan

Use focused tests, relevant subsystem suites, full backend validation, frontend
test/type/build validation, static compile/diff checks, and the canonical
no-WING gate.

## 8. Risks and rollback

The primary compatibility change is that missing PyAudio now yields an empty
device list and `muq_genre` raises instead of returning a fabricated score.
Rollback is a normal source revert; no persistent mixer, WAL, queue, or
hardware state is mutated by this decision.
