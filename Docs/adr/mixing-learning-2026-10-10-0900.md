# ADR: Mixing Learning search 2026-10-10 09:00 MSK

## 1. Context

The 09:00 search had to extend the shared scientific/video/reference queue after the 2026-10-09 DPNMM review, without duplicating the already indexed BERT-APC source family or claiming that companion articles are video transcripts. The fresh arXiv list contained a new reference-free pitch-correction preprint; official mastering materials also offered two bounded, testable workflow hypotheses.

## 2. Options considered

1. Merge the new pitch-correction paper into the existing BERT-APC card as a version duplicate.
2. Keep it as a separate dependent-comparator card and require harm-to-correct-notes plus intent review.
3. Promote the paper's `±1..±3` semitone labels as an automatic correction range.
4. Treat the two official companion articles as complete studies of their linked videos.
5. Preserve video status as metadata-only while using the articles as separate engineering sources.
6. Add the Cambridge-MT candidate as downloadable/approved based on search metadata alone.

## 3. Decision

Choose options 2 and 5. Add `arXiv:2610.11524v1` as a separate preprint linked to the existing BERT-APC family, retain its central repair-versus-harm trade-off, and keep any project experiment human-reviewed and blocked. Add the two official articles as bounded engineering sources, while leaving their linked videos `queued_source_no_transcript`. Keep the Cambridge-MT item `queued_rights_recheck` because its live terms could not be revalidated.

## 4. Why this won

The new paper has distinct authors, editing heads, tonal reranker and private evaluation data, so it is not a duplicate. At the same time, it inherits the MusicBERT/context premise and compares against BERT-APC, so it is not independent confirmation. Its own numbers show why a harm gate is necessary: the full model repairs more errors than BERT-APC but also damages more originally correct notes. The mastering sources describe useful one-factor diagnostics, but neither is a controlled scientific study and neither supplies a readable video transcript in this run.

## 5. Rejected alternatives

- Automatic semitone range: rejected because label support and synthetic perturbation design are not production thresholds.
- Accuracy-only promotion: rejected because repair gains coexist with a 3.19% harm rate and expressive alternatives.
- Companion article equals transcript: rejected because video-only content, timing and on-screen parameters were not retrieved.
- Combined side cut plus mid boost: rejected because it changes two factors and obscures causality.
- Approved Cambridge-MT download: rejected until the specific item terms and live metadata are rechecked.

## 6. Implementation plan

- Add three Source Cards, two metadata-only Video Source Cards and one rights-recheck Reference Card.
- Add five Knowledge Cards, three bounded candidate rules and three unrun experiment plans.
- Preserve the current executable experiment priority.
- Do not modify production audio, DSP, model, runtime code or live-control paths.

## 7. Test plan

- Validate the JSON patch.
- Confirm every rule has `auto_apply:false` and every experiment is `not_run` or `not_run_blocked`.
- Confirm literature/video/reference stable keys are not already present.
- Confirm the linked videos contain no technique or numeric claims not grounded in the official articles.
- Confirm the repository change is limited to report, JSON patch, ADR and index entry.

## 8. Risks and rollback

- **Risk:** professional-reference alignment erases intentional pitch expression. **Guard:** human intent annotation, harm count and blind naturalness review.
- **Risk:** session examples become mastering presets. **Guard:** no adopted dB/frequency range and one-factor calibration.
- **Risk:** metadata-only video is treated as studied. **Guard:** separate article depth from video status.
- **Risk:** catalogue listing is mistaken for verified usage rights. **Guard:** `queued_rights_recheck`, no download.
- **Rollback:** remove this report, JSON patch, ADR and index entry; no production audio/code/DSP rollback is required.
