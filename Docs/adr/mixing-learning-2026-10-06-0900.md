# ADR: strengthen debleeding validation without duplicating the knowledge base

- Date: 2026-10-06
- Status: Accepted
- Update: ML-2026-10-06-0900

## Context

A new ICASSP 2026 paper reports improved bleed-reduction isolation metrics but lower SAR in several comparisons. A separately rediscovered multichannel permutation-equivariance paper was already stored and fully studied in ML-2026-09-14-1800. Two vendor pages describe alignment and bleed-control workflows, but do not provide independent controlled validation.

## Decision

1. Add Bleed No More as a new full-text Source Card and record the inaccessible advertised repository.
2. Do not duplicate arxiv:2606.16551 or count it as independent confirmation.
3. Strengthen existing RULE-ML-20260914-1800-03 so SI-SDR/SIR gains cannot pass without SAR, artifact, transient, phase/room and loudness-matched preference gates.
4. Refine existing RULE-ML-20260914-1800-04 to keep polarity, time alignment and phase coherence as separate decisions.
5. Revise EXP-ML-20260914-1800-01 instead of creating a duplicate debleeding experiment.
6. Keep vendor claims as bounded workflow/tool hypotheses; do not buy, integrate or auto-apply them.

## Consequences

The knowledge base gains a concrete isolation-versus-artifact counterexample and a stricter controlled A/B. No runtime behavior changes. The debleed experiment remains not_run and blocked by missing public implementation and rights-clear test material. All rules remain auto_apply:false.
