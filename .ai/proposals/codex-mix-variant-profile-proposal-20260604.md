# Codex Proposal: Mix Variant Profile Proposal Layer

## 1. Short thesis

Move genre/task variants from offline-only to Assisted proposal support by
adding a transport-free mix profile contract that proposes intent and target
planes, not live console writes.

## 2. What I understood

The Automixer must eventually form different mix variants for genres and tasks.
Current live authority is still supervised fader/gain only, so the next safe
step is not applying a genre preset. The useful step is a structured profile
proposal that downstream proposal layers can consume later.

## 3. Proposed solution

Create `backend/mix_variant_profile_proposal.py` with:

- genre/task normalization;
- deterministic profile resolution using Ayaic level planes;
- reference loudness, arrangement context, headroom, feedback, confidence, and
  operator-intent evidence;
- explicit blockers for active performance, missing reference metrics, missing
  arrangement context, style clone requests, and insufficient true-peak
  headroom;
- Product Layer style proposal payloads with `kind="mix_variant_profile"`,
  `can_apply_live=False`, `can_import_to_queue=False`, and no
  `requested_change`.

Wire it into the processing registry, proposal contracts, and project-goal
capability matrix as `proposal_only` / `assisted`.

## 4. Likely files to touch

- `backend/mix_variant_profile_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_mix_variant_profile_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/mix-variant-profile-proposal-20260604.md`

## 5. Alternatives considered

- Keep variants offline-only: safe, but it does not create an Assisted-mode path
  toward the stated project goal.
- Apply profile changes directly through faders, FX, master bus, or presets:
  rejected because it fans out into many unproven write surfaces.
- Reuse `offline_agent_mix` only: useful for renders, but not a live operator
  proposal contract.

## 6. Risks

- Operators may expect a profile proposal to apply immediately.
- Reference matching can drift toward style imitation if not bounded.
- Genre variants can touch many surfaces, so follow-up fan-out must stay
  proposal-only until each surface has separate proof.

## 7. Test plan

- Verify the module has no mixer transport or direct apply tokens.
- Verify healthy genre/task evidence creates a proposal-only payload.
- Verify active performance, style clone, missing evidence, and headroom
  blockers.
- Verify registry, proposal contract, and capability matrix wiring.

## 8. Where the other agent may disagree

Kimi may argue that genre variants should remain offline-only until ABX or
reference scoring exists. That is valid for quality claims, but an Assisted
profile proposal is a narrower contract: it says what the operator can review,
not that the mix already matches a reference.
