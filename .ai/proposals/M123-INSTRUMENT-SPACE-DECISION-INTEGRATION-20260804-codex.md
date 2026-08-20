# Proposal: M123 Instrument Space decision integration

## 1. Short thesis

Use Instrument Space as a conservative evidence composer, not as a second
automixer. It should relate measured channel objects, identify conflicts and
attach those findings to existing operator proposals while leaving action
selection and all console authority in the established pipelines.

## 2. What I understood

The current visualizer already maps pan, spectral centroid and level. The
missing production step is a deterministic relation/decision layer that makes
the map useful to Automixer and behaves identically for WING and dLive.

## 3. Proposed solution

1. Add a transport-free decision-support module with versioned DTOs.
2. Compute pairwise pan proximity, log-frequency proximity and level
   competition only from available measured values.
3. Classify pairs as `clear`, `watch` or `review_required`; never emit an EQ,
   pan or fader value from centroid proximity alone.
4. Emit ordered next-analyzer families: level balance first when appropriate,
   then pan/mono-phase review and ERB/frequency-separation review.
5. Compare an optional previous snapshot for bounded drift evidence.
6. Embed the result in Product Layer and annotate soundcheck proposals with
   matching, non-authoritative evidence.
7. Fail closed on an unsafe or write-capable decision-support contract.

## 4. Likely files to touch

- `backend/instrument_space.py`
- `backend/instrument_space_decision_support.py`
- `backend/operator_product_state.py`
- `backend/operator_recommendation_bridge.py`
- frontend Instrument Space adapter/types/panel
- focused backend and frontend contract tests
- project plan, ADR and capability truth

## 5. Alternatives considered

- Direct scene-to-console corrections: rejected because it bypasses existing
  family-specific evidence and hardware gates.
- Replacing ERB masking with centroid distance: rejected because centroid is
  only a coarse triage signal.
- Keeping the feature visual-only: rejected because it does not satisfy the
  requested integration into correction decisions.

## 6. Risks

- False masking positives from centroid-only evidence.
- Operator confusing visual placement with physical localization.
- Proposal metadata accidentally interpreted as apply authority.
- Extra pairwise CPU cost at high channel counts.

Mitigations: bounded O(n^2) work over at most the Product Layer channel set,
strict evidence completeness, coarse triage wording, immutable authority
denials and dependency tests.

## 7. Test plan

- Golden relation scoring and stable ordering.
- Missing/stale/unsafe contract rejection.
- Optional snapshot drift tests.
- Product Layer integration without console queries.
- Soundcheck proposal annotation without changing requested values or
  transaction authority.
- Frontend contract rejects write-capable support.
- Targeted backend/frontend tests, build, diff check and broader safe rails.

## 8. Where another reviewer may disagree

A reviewer may prefer waiting for full-band ERB spectra before generating any
relation. This proposal keeps centroid relations deliberately at triage level
and routes them to existing analyzers, which gives immediate product value
without pretending they are correction proof.
