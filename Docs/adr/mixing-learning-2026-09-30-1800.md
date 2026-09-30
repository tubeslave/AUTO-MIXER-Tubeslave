# ADR: publish the 2026-09-30 18:00 Mixing Learning deep review

## 1. Context

The preceding search update queued a full statistical, metric-transfer and implementation audit for AEROMamba/PAQM. The shared knowledge base needs the condition-specific significance results, an independent objective-metric comparator, pinned revisions, licence state and revised evaluation gates without changing runtime behavior.

## 2. Options considered

1. Keep the search report unchanged until audio or training can be rerun.
2. Publish the statistical and code review as universal PAQM guidance.
3. Publish a bounded deep review, revise candidate rules, keep all experiments unrun and block code integration where rights or failure accounting are unresolved.

## 3. Decision

Choose option 3. Add `ML-2026-09-30-1800.md` and its JSON patch, update the shared index, and keep every rule `auto_apply:false` and every experiment `not_run`.

## 4. Why this won

It preserves useful evidence while preventing three overclaims: that one dataset establishes a universal perceptual loss, that a less-coupled metric is universal ground truth, or that a public repository is integration-ready. It also records a concrete denominator bug and licence conflict that materially affect reproducibility and use.

## 5. Rejected alternatives

- Option 1 discards verified public evidence and leaves known risks undocumented.
- Option 2 overgeneralizes restoration evidence to mixing and ignores PianoEval, domain-dependence, statistical multiplicity and licensing limits.

## 6. Implementation plan

- Add one full-text comparator source card and update the AEROMamba source card.
- Pin the AEROMamba and torchpaqm revisions.
- Record skipped-batch accounting and the GPL-vs-MIT conflict.
- Revise metric and invalid-batch rules; add multiplicity and licence gates.
- Revise the auxiliary bandlimit protocol while retaining the drum-room topology test as the next practical A/B.
- Leave video cards queued because no permitted transcript was available.

## 7. Test plan

- Parse the JSON patch.
- Check Markdown/JSON IDs, mode, slot, predecessor, `auto_apply:false` and `not_run` states.
- Confirm no runtime, DSP, audio, model or dependency files change.
- Run repository CI before merge.

## 8. Risks and rollback

Risk is limited to documentation interpretation. If a statistic, revision or licence status is corrected, revert this documentation commit or publish a superseding Mixing Learning update; no runtime rollback is required.
