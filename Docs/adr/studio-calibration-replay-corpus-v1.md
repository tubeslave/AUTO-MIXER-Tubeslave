# ADR: STUDIO Calibration Replay / Corpus v1

## Decision

Define a deterministic, append-only evidence corpus for pairs of Perceptual Critic output and later human listening feedback. The corpus is calibration evidence only. It cannot update production thresholds, waive protected regressions, change an Autonomous Iteration transition, or promote an audio baseline.

Each record is canonical JSON with a digest over its payload and a pointer to the previous record digest. Replay must re-run the existing disagreement classifier from stored critic evidence plus normalized human review and require the same classification. A stored result that no longer replays is invalid evidence.

## First real seed case

The first intended corpus record is the existing Belye Stai v2 vocal disagreement:
- target: `vocal_intelligibility`;
- metric: `0.6800658645 -> 0.6714029435`, delta `-0.0086629210`;
- machine result: rejected only for `target_not_improved`;
- protected regressions: none;
- later human review: accepted for foreground stability, phrase consistency and stable mix position;
- replay classification: `target_proxy_miss_candidate`.

The machine decision remains rejected. The human review remains positive. Corpus replay preserves both facts rather than rewriting either history.

## Record contract

A record binds: source evidence identifiers, baseline/candidate identifiers when available, critic target and decision, failures and protected regressions, normalized human review observations, replay classification, previous-record digest, and the record digest.

The record also states that it is evidence-only and requires human listening for subjective acceptance. No corpus summary is itself an acceptance threshold.

## Replay and append rules

1. Serialize records canonically: UTF-8 JSON, sorted keys, finite numbers, one record per line.
2. Recompute the record digest with `record_id` excluded.
3. Require each `previous_record_id` to equal the preceding record's digest.
4. Re-run `classify_human_machine_disagreement()` from stored critic evidence and human review.
5. Require the stored classification, calibration action, preserved failures/protected regressions and metric hypotheses to match replay.
6. Reject duplicate source reports.
7. Never silently rewrite earlier records. Appending is permitted only after replaying the complete existing prefix successfully.

## Validation performed in this task

An isolated local prototype exercised eight cases: deterministic record construction, two-record chain replay, digest failure after record mutation, duplicate rejection without file mutation, rejection of an already-corrupt corpus, rejection of annotation drift, preservation of a protected-regression safety conflict, and rejection of reports without annotation-only evidence. All eight passed.

The prototype was deliberately not claimed as repository runtime code. Attempts to add the executable Python implementation through the connected repository write path were blocked by the tool safety layer. The repository therefore receives the validated contract and audit evidence only in this task; production integration remains pending rather than being falsely reported as complete.

## Boundaries

No audio DSP, mastering, LIVE code, accepted Belye Stai vocal settings, Perceptual Critic thresholds, Autonomous Iteration policy, neural audio, paid service, or audio baseline is changed.
