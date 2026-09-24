# ADR: STUDIO Calibration Replay / Corpus v1

## Context

Perceptual Human Calibration and Autonomous Iteration annotation already preserve machine decisions separately from later listening preferences. The missing piece is a replayable, append-only corpus of those annotated reports. Exact raw KEYS/PLAYBACK, OH and isolated vocal A/B bytes remain unavailable in the current runtime; this is an independent, audio-free validation task.

Status: runtime and tests implemented and tested locally, but executable repository write remains blocked. This ADR does not claim runtime integration, repository CI or merge.

## Options considered

1. Recompute or overwrite earlier machine decisions after human feedback: rejected.
2. Store unverified classification labels in a flat JSON list: insufficient history integrity.
3. Snapshot supplied annotated iteration reports in canonical, digest-linked JSONL and re-run the existing classifier: selected.

## Decision

Use `build_calibration_record`, `append_calibration_report` and `replay_calibration_corpus` in the proposed `audio_workbench/mixing/calibration_corpus.py`. The existing classifier remains unchanged. A record binds the entire supplied report snapshot, its canonical SHA-256, stable source report ID, optional provenance references, previous record digest and full replay output. Caller-owned mutable values are detached.

The corpus is evidence-only. It cannot update production thresholds, waive protected regressions, change iteration transitions or delivery routing, or promote an audio baseline. Subjective acceptance still requires human listening.

The expected annotation is reconstructed from the existing classifier and exact annotation-only flags. Unknown fields, changed flags, classifier drift or result drift fail closed. Every original critic failure and protected regression stays in the snapshot and replay result.

## Why this won

One complete snapshot avoids silently losing audit fields as reports evolve. Canonical JSONL and record digests detect corruption; a required expected-head checkpoint on append prevents stale writers. The optional externally trusted head on read also detects suffix truncation. Duplicate source IDs and identical report snapshots under another ID are rejected.

Canonical here means this schema's UTF-8 JSON with sorted string keys, compact separators and finite native JSON values; it is not a claim of RFC 8785 compliance. The report hash is over canonical report JSON, not the original file formatting. Provenance references are stored, not independently fetched or authenticated.

## Rejected alternatives

No machine-threshold fitting, inferred human feedback, automatic history migration or new production dependency. The legacy prototype seed is not silently imported: it lacks the complete annotated iteration report expected by the runtime. The historical Belye Stai v2 proxy-miss case remains a future source-verified import, not a newly reconstructed render record.

## Implementation plan

The local implementation validates incoming evidence, obtains an exclusive sidecar writer lock, verifies the complete bounded corpus and its expected head, rejects duplicates, writes a same-directory temporary file, flushes/fsyncs it and atomically replaces the corpus while preserving all prefix bytes. A final prefix comparison catches accidental noncooperating writes before replacement.

This is logical append-only storage, not WORM storage or a digital signature. Without a trusted external checkpoint, a valid shorter prefix or wholly rehashed history cannot be detected. Stale locks after process death require operator inspection, not automatic unlocking. Network filesystems, adversarial writers and power-loss durability of the directory entry are outside the v1 guarantee. Limits are 1 MiB per record and 32 MiB per corpus.

Next: land the exact local module/tests through an authorized working code-write path, then run actual annotation integration and complete repository CI before merge. Only then import source-verified annotated reports.

## Test plan and observed validation

Current actual local command:

`python -m pytest tests/test_studio_calibration_corpus_v1.py tests/test_studio_perceptual_calibration_v1.py -k 'not existing_annotation_api_integration' -q --tb=short --junitxml=evidence/local-unit-tests.xml`

Observed: **40 passed, 1 deselected** (33 corpus cases and 7 existing calibration cases). The deselected test imports the full repository's `studio_iteration`; the container lacks that runtime. It is written but not claimed passed. Compileall passed. The dependency classifier copied from the supplied evidence archive exactly matches repository Git blob `1800c10eb54471b8b61d224e392e01e5447aff12`.

Checks cover deterministic records, input isolation, exact-prefix preservation, full replay, duplicate IDs/content, mutation, rehashed annotation drift, malformed/noncanonical JSON, duplicate keys, NaN/infinity, incomplete records, stale checkpoints, truncation, protected conflicts, authority flags including integer/bool substitution, writer contention, concurrent writers, simulated fsync/replace failure, symlinks, record/corpus limits, classifier drift and record reordering.

A separately executed six-record synthetic experiment reproduced all six disagreement outcomes once each, preserving the prefix after every append. Trusted head: `4d9afa0b2cb64f4b3fed9fefcdbb53bb8e7573cba67d441b4c1b1bddab27ffa2`. These are synthetic protocol fixtures, not human listening data or musical acceptance.

Correction of the earlier ADR: the previous claim of eight passed prototype checks is not reproducible from the supplied ZIP, which contains a text result, report and seed but no executable test harness. It is not accepted as validation for this implementation. The actual pytest/JUnit run above supersedes it.

## Risks and rollback

On validation or pre-replacement I/O failure, preserve prior corpus bytes and remove only this operation's temporary file/lock. The runtime is not in the repository yet: the attempted `create_file` for the Python module returned an OpenAI safety-status determination block, without a commit. No alternate code-write route was attempted. Documentation/state writes succeeded. Direct container Git also failed DNS resolution.

No audio DSP, mastering, LIVE code, existing critic thresholds, accepted vocal settings, neural audio, paid services or audio baseline changed. A successful local test run is not a substitute for repository CI or human listening.
