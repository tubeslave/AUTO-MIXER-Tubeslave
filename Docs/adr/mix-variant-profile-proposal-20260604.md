# ADR: Mix Variant Profile Proposal Layer

Date: 2026-06-04

## 1. Context

The project goal requires automatic genre and task variants. Existing offline
mix tools can render or suggest variants, but the project lacked an Assisted-mode
contract for proposing a variant profile to the operator. The current live write
boundary remains supervised fader/gain only.

## 2. Options considered

- Keep genre variants offline-only.
- Add a proposal-only mix profile contract.
- Apply genre/task presets directly to console surfaces.

## 3. Decision

Add `mix_variant_profile_proposal` as a transport-free proposal-only method.
Move `genre_variants` to `proposal_only` with an Assisted-mode ceiling. The
profile can fan out later into proposal-only bus balance, FX, master-bus, and
scene/preset layers, but it cannot apply live.

## 4. Why this won

This advances the explicit project requirement without widening live authority.
The method captures the evidence a real engineer would need before accepting a
variant: genre confidence, task confidence, arrangement context, reference
metrics, Ayaic target planes, true-peak headroom, feedback risk, and a no-style
clone guard.

## 5. Rejected alternatives

- Direct preset application was rejected because it can touch faders, EQ,
  dynamics, FX, scenes, and presets at once.
- Offline-only status was rejected because it leaves no operator-facing path for
  Assisted mode.
- Named-artist style cloning was rejected because reference comparison should
  inform targets, not copy a specific artist's mix identity.

## 6. Implementation plan

1. Add a pure Python profile resolver and analyzer.
2. Emit only non-queueable proposal payloads with no `requested_change`.
3. Register the method in processing metadata and proposal contracts.
4. Update the capability matrix row for `genre_variants`.
5. Add tests for safe proposals, blockers, and live-boundary invariants.

## 7. Test plan

- Compile touched Python files.
- Run focused mix variant tests.
- Run proposal-boundary tests covering registry and project-goal matrix.
- Run `git diff --check` and smoke-check the emitted payload flags.

## 8. Risks and rollback

Risk: a UI may treat a profile proposal as executable. Mitigation: the payload is
non-queueable, has `can_apply_live=False`, includes unsupported value-type
blockers, and contains no requested change. Rollback is to remove the method from
registry/profile/matrix wiring; no direct live apply path is introduced.
