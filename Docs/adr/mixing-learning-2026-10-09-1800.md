# ADR: DPNMM artifact readiness and playback-test gate

## 1. Context

The 18:00 deep review had to decide whether the DPNMM paper and author code found at 15:00 were ready to support a controlled project playback experiment. The paper reports objective gains for environment-conditioned noise masking, but the repository has no public checkpoint, packaged dataset or visible root licence. The deeper audit therefore focused on code-to-paper semantics and entry-point reproducibility without running audio, training or inference.

## 2. Options considered

1. Treat the paper's reported results as sufficient and implement a project analogue directly.
2. Repair or reinterpret the author repository inside the production project and run it.
3. Keep the source as scientific background, downgrade artifact readiness and require an isolated parity gate before any musical A/B.
4. Reject the source entirely because the public artifact is incomplete.

## 3. Decision

Choose option 3. Retain the paper's bounded playback-level finding, but mark the public artifact `not_run_blocked`. Revise the candidate rule and experiment to require a pinned, licensed, self-consistent wrapper, explicit sample-rate handling, realised-filter bounds and measured smoothing response before any audio comparison. Keep `auto_apply:false` and do not modify project DSP or runtime.

## 4. Why this won

Static checks found four deterministic entry-point/configuration mismatches. The code also shows that Bark-control clamps are not direct per-bin limits after overlapping synthesis, and that the labelled 250 ms smoothing constants do not match the visible recurrence when evaluated at the 512-sample hop. These findings do not refute the paper's reported experiment, but they prevent the released snapshot from serving as a drop-in reference implementation.

## 5. Rejected alternatives

- Direct implementation: rejected because it would silently replace unresolved author-code behaviour with a project interpretation.
- Production repair: rejected because this run is research-only and the licence/checkpoint/data gates remain unresolved.
- Full rejection: rejected because the peer-reviewed study and official results still provide useful evidence about the trade-off between masking and playback-level preservation within their stated task.
- Copying `+10/3`, `-5/3` or 250 ms as presets: rejected because these numbers are internal control/label values whose realised behaviour differs after synthesis or recurrence timing.

## 6. Implementation plan

- Update the existing Source Card and Artifact Card without creating a duplicate source family.
- Add five bounded Knowledge Cards.
- Revise `RULE-ML-20261009-1500-01` and `EXP-ML-20261009-1500-01`.
- Preserve video/reference queues and experiment priority.
- Do not change audio, production code, DSP, models or live control paths.

## 7. Test plan

- Validate JSON syntax and stable IDs.
- Confirm all new/revised rules have `auto_apply:false` and the experiment remains `not_run_blocked`.
- Confirm the repository diff contains only the report, JSON patch, ADR and index entry.
- Record the static call-signature audit and syntax compile separately from any unperformed runtime test.
- Before a future audio A/B, require a synthetic parity harness for entry-point startup, sample-rate handling, deterministic forward output, realised filter bounds and smoothing step response.

## 8. Risks and rollback

- **Risk:** static derivations are mistaken for measured audio. **Guard:** label them explicitly as source-code implications and keep the experiment blocked.
- **Risk:** paper evidence is discarded because the artifact is incomplete. **Guard:** keep the scientific Source Card and narrow only artifact readiness.
- **Risk:** implementation numbers become production presets. **Guard:** prohibit transfer and require realised-response measurement.
- **Rollback:** remove this report, JSON patch, ADR and index entry; no production audio/code/DSP rollback is needed.
