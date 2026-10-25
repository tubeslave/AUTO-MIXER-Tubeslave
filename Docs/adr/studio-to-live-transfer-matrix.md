# Studio -> Live transfer matrix

The existing live stack has useful meters/safety/control plumbing, but several decision rules are heuristic and can be directionally wrong. The new live pipeline reuses the decision architecture proven in the studio pipeline.

| Studio capability | Live transfer | Live modification |
|---|---|---|
| Perceptual Critic | YES | rolling windows, no offline whole-song assumptions |
| one-hypothesis iteration | YES | proposal -> bounded write -> short verify window -> keep/rollback |
| Masking Director | YES | pairwise target/masker evidence; prefer culprit/group correction |
| Dynamics Director | YES | infer from crest/envelope/GR; bounded threshold/ratio changes |
| Vocal Director | YES | prominence/intelligibility target; no pitch/editing/saturation by default |
| Drum/Bass/Guitar/Keys Directors | YES | role-aware EQ/dynamics/balance policies |
| Space Director | YES | FX sends/returns only; section-aware and bounded |
| Section Context | YES | online state: sparse/body/peak; slow transitions/hysteresis |
| Artifact Critic | PARTIAL | detect pumping/clipping/harsh correction regressions |
| Mastering Director | NO | replaced by Main Director; no loudness maximization/limiting target |
| Cleanup/de-bleed | NO control | analysis may estimate bleed confidence, but no destructive restoration in live path |
| timing/pitch editing | NO | never part of live control |
| full-song automation | NO | causal slow rides only |
| reference matching | OPTIONAL | target corridors only, never force a live show to a mastered record |

## Important correction to legacy live logic

Legacy spectral masking code uses a rule equivalent to:
"if vocal dominates background by >6 dB, cut the background."

That is not evidence that the vocal is masked. The new logic measures target audibility and spectral overlap. A background cut is proposed only when:
1. the target is active;
2. target intelligibility/prominence is below its corridor;
3. masker energy overlaps the target band;
4. the candidate masker materially contributes to that overlap.

This is a concrete example of why the studio decision architecture is replacing isolated heuristics.
