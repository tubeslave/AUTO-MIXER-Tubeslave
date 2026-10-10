# ADR: scope the «Пена» evidence pack before listener decisions

- **Status:** accepted for research-governance documentation
- **Date:** 2026-10-10
- **Update:** `ML-2026-10-10-1800`

## 1. Context

The project-owned `ML-AUDIO-PENA-20261010_evidence_patch.json` contains a finished local mix, stems and 21 rendered A/B pairs. The 15:00 search read selected fields and correctly withheld preference claims, but described the pack broadly as 21 fair one-factor pairs and queued the parallel-drum comparison first.

The 18:00 review read all 3,243 lines. It confirmed file integrity and loudness matching, then audited the declared factor and dependency of every pair.

## 2. Options considered

1. Keep `PENA-AB-01` first and treat every pair as an equivalent one-factor test.
2. Reject the whole pack because no listening verdict exists.
3. Preserve the technically valid pack, classify causal scope, order dependent tests, and start listening with the narrowest useful scalar comparison.

## 3. Decision

Choose option 3.

- The pack is accepted as rendered/QC evidence, not preference evidence.
- All 21 pairs may be auditioned locally.
- Sixteen pairs are narrow scalar/state comparisons.
- `PENA-AB-05`, `18`, `19` and `20` are frozen envelope/curve/topology interventions.
- `PENA-AB-01` is a multi-processor chain on/off comparison.
- `PENA-AB-11` precedes `12`; `PENA-AB-02` precedes `21`.
- The next listening task becomes `PENA-AB-15`: mix-bus compressor attack 10 versus 30 ms.
- No result can become portable until a complete DSP manifest, blind answer key and independent-multitrack transfer test exist.
- All rules remain `auto_apply:false`.

## 4. Why this won

This preserves high-quality work already done: source hashes, reopen checks, raw files, BS.1770-matched copies and objective measurements. At the same time, it prevents a whole-chain preference from being misreported as proof of each component and prevents dependent tests from being interpreted out of order.

The attack comparison is the best first listener task because it declares one scalar change while holding threshold, ratio, release, detector and knee constant. It also tests a musically important rock-mixing trade-off without automatically privileging vocals.

## 5. Rejected alternatives

- **Unqualified “one factor” label:** too broad for a chain containing HPF, compression, saturation and EQ or for a whole EQ curve/limiter topology.
- **Discarding the pack:** wastes valid integrity and comparison evidence merely because subjective evaluation is pending.
- **Promoting objective metrics to preference:** loudness, peak, spectrum and correlation do not establish punch, smoothness, groove or artistic fit.
- **Starting with the parallel chain:** useful for a whole-bus verdict, but weaker for causal rule learning than the attack-only pair.

## 6. Implementation plan

1. Publish the full audit and JSON patch.
2. Record `KC-ML-20261010-1800-01..03` and `RULE-ML-20261010-1800-01`.
3. Run three randomized blind repeats of the existing matched `PENA-AB-15` pair.
4. Record choice, confidence and protected-quality observations before revealing X/Y.
5. Recover or generate the complete parameter/answer-key manifest.
6. Continue prerequisite chains and remaining pairs one at a time.
7. Repeat any accepted candidate on an independent licensed/project multitrack before generalization.

## 7. Test plan

For `EXP-ML-PENA-AB15-LISTEN-20261010-1800`:

- use exactly the existing 51.28–76.88 s hidden X/Y matched files;
- three randomized blind passes;
- do not inspect the evidence patch during scoring;
- retain raw FLAC metrics separately;
- score punch, snare/kick attack, cymbal smoothness, pumping, low end, vocal placement, section hierarchy, mono and headroom;
- accept locally only if one state wins at least 2/3 passes without protected-quality regression;
- otherwise record reject or ambiguous.

No automation applies the result.

## 8. Risks and rollback

- **Unblinding risk:** embedded metrics may reveal mapping. Mitigation: score first, inspect later.
- **Listener fatigue:** 21 pairs invite sequential bias. Mitigation: one pair per decision block.
- **Baseline risk:** the frozen mix has not been approved by Dmitry. Mitigation: keep every inference local.
- **Reproducibility risk:** concise factor descriptions omit complete DSP settings. Mitigation: do not claim a portable recipe until the manifest is recovered.
- **Rollback:** remove the listening-priority change and return every pair to `pending_listening`; no audio, DSP or production code requires rollback.
