# ADR: Mixing Learning search 2026-10-09 15:00 MSK

## 1. Context

The 15:00 search had to extend the shared scientific/video queue without re-adding recent automatic-mixing versions already indexed, without treating metadata as completed reading, and without changing production audio or DSP. The search found one older peer-reviewed gap about masking ambient noise with a transformed playback signal and one fully readable engineer demonstration about rock bass.

## 2. Options considered

1. Promote DPNMM as a general spectral de-masking rule for Automixer.
2. Retain DPNMM only as environment-conditioned playback research with an isolated, blocked experiment.
3. Reject the paper because it is not a multitrack mixing study.
4. Copy the video's frequencies/plugins as a starting preset.
5. Retain only the bounded articulation hypothesis and require a one-factor DI A/B.

## 3. Decision

Choose options 2 and 5. DPNMM is recorded as a primary source for playback adaptation, explicitly outside multitrack de-masking. The Scheps video is recorded as an engineer demonstration supporting an articulation-first bass hypothesis, not a preset. All rules remain `auto_apply:false`; experiments remain `not_run` or `not_run_blocked`.

## 4. Why this won

The decision preserves the source's actual causal task, reports the objective-only evaluation, and avoids importing a simulated mono headphone result into stem mixing. For the video, changing only DI contribution is testable on project-owned material, while copying ASR-derived frequency/plugin settings would couple several factors and overstate transcription certainty.

## 5. Rejected alternatives

- General spectral target: rejected because ambient-noise masking and stem de-masking are different tasks.
- Automatic playback adaptation: rejected because licence, checkpoint, example, stereo and listener-validation gates remain open.
- Frequency/plugin preset: rejected because the transcript is auto-generated/translated and the demonstrated recording is session-specific.
- Treating recent same-title video uploads as independent evidence: rejected until content/version identity is resolved.

## 6. Implementation plan

- Add one literature Source Card and one artifact audit.
- Add one fully studied Video Source Card and three metadata-only queued Video Source Cards.
- Add five bounded Knowledge Cards, two candidate rules and two one-factor experiment plans.
- Append the update to the shared index only; do not modify production code, audio, DSP or model assets.

## 7. Test plan

- Validate JSON syntax and stable IDs.
- Confirm `auto_apply:false` and `not_run`/`not_run_blocked` in every new rule/experiment.
- Confirm no production path changed.
- Review dedup keys and ensure metadata-only videos contain no inferred technique details.

## 8. Risks and rollback

- **Risk:** users may read DPNMM as a mixing EQ method. **Guard:** task-boundary language in source, knowledge, rule and experiment cards.
- **Risk:** ASR numbers become presets. **Guard:** require visual verification and use no transferred range in the rule.
- **Risk:** author repository is mistaken for independent validation. **Guard:** same-author-family field and `citation_check:partial`.
- **Rollback:** remove the three new files and index entry; no production audio/code/DSP rollback is required.
