# ADR: STUDIO Autonomous Calibration Annotation v1

## Decision

Persist post-listening human/machine calibration evidence inside the normal `studio-autonomous-iteration-v2` report under a dedicated `human_calibration` key. The annotation is descriptive only. It may not modify the historical Perceptual Critic result, Autonomous Iteration transition, delivery routing, protected regressions, baseline state, or baseline-promotion decision.

Human calibration is therefore a second ledger, not a second judge. The first ledger records what the machine decided at render time; the calibration ledger records what a listener later preferred and whether that suggests a proxy miss worth studying.

## Why

The Belye Stai v2 vocal was a real disagreement case: the existing intelligibility target rejected the candidate, while later level-matched listening preferred the vocal's stability and mix position. Perceptual Human Calibration v1 correctly records that as `target_proxy_miss_candidate`, but the evidence previously lived outside ordinary Autonomous Iteration reports.

This task closes that audit gap without giving the new evidence authority it has not earned.

## API

`add_human_calibration_annotation(report, human_review)` returns a deep-copied report. With `human_review=None` it is a semantic no-op. With a valid review it calls the existing calibration classifier and adds only `human_calibration` metadata.

`persist_human_calibration_annotation(path, human_review)` performs the same operation on an existing `iteration_report.json` using a temporary file and replace operation.

A second annotation cannot silently overwrite the first. Invalid human evidence fails closed.

## Invariants

The following fields are snapshotted before annotation and must remain unchanged afterward: iteration status, `baseline_promoted`, `baseline_after`, delivery role/source/rollback state, transition status/action, transition baseline before/after, promotion/rollback flags, and protected regressions.

Human acceptance cannot waive a protected regression. A target-only rejection plus human acceptance can be classified as `target_proxy_miss_candidate`, but remains a rejected historical machine transition. Existing Perceptual Critic thresholds are unchanged.

## Validation

Tests cover target-only disagreement, protected-regression conflict, absent review, malformed review, deterministic serialization, persisted annotation, and overwrite protection. Full repository CI is required before merge.

No audio DSP, mastering, LIVE code, neural audio, paid services, or audio baseline are changed by this task.
