# ADR: STUDIO Calibration Replay / Corpus v1

## Context

The corpus is an evidence-only record of Perceptual Critic output and later human listening feedback. It must not change production thresholds, waive protected regressions, change Autonomous Iteration routing or promote an audio baseline.

During the September 24 continuation, another implementation, tests and seed were added to the same feature branch while a separate local prototype was being developed. This was detected by a final branch comparison. The remote implementation was then independently read and copied with exact Git-blob verification: module `603e9808dcecf07d5c4dc23904d52068d83606da`, tests `e21ffe0eb0805f67c7cc79c79b487631d2fb219b`, reviewed at commit `2f0d9594e1eeca3e5713bf4a2fc24e559e82cd16`.

Decision: **requires revision; do not merge yet**. The compatible integrity correction is tested locally but not written to GitHub. Earlier assertions that remote runtime code was absent were superseded by this reconciliation.

## Options considered

1. Replace the newly added API with the separate whole-report prototype: rejected for this bounded task because it changes the record contract and would require explicit seed migration.
2. Accept the remote code based on the old text-only prototype report: rejected; actual tests expose failures.
3. Preserve the remote API and record representation while correcting append integrity: selected.

## Decision

Keep the existing `build_calibration_record`, `append_calibration_record`, `load_calibration_corpus`, `replay_calibration_record` and `replay_calibration_corpus` API. Preserve canonical record identity for native JSON evidence and all evidence-only flags.

The local compatible correction deep-copies JSON evidence before hashing, rejects a missing final newline before writing, serializes cooperating writers with an exclusive sidecar lock, validates the complete future chain before committing, and uses an fsynced same-directory temporary file with atomic replacement. Every previous corpus byte is preserved. Failed fsync/replace operations leave the old corpus intact and clean up this writer's temporary files. Stale locks are never automatically removed.

## Why this won

Three integrity defects were reproduced against the exact remote source:
- without the final newline, append concatenates JSON objects, then raises after corrupting the corpus;
- shallow copies allow caller list mutation to change a previously digested record;
- concurrent writers can create two records with the same previous-record pointer.

An existing canonical-loader test also fails because its regex expects a different error description. The correction makes the parse error explicitly identify invalid canonical JSON; it does not weaken the test.

The compatible patch fixes these defects without changing DSP, mastering, critic thresholds, human feedback, source data or the record format. A fixed native-JSON fixture retains its original digest.

## Rejected alternatives

No automatic baseline promotion, invented listening result, retrospective threshold adjustment or automatic schema migration. The separate local whole-report prototype is not the production candidate for this task. Its 40 passed tests and six synthetic records must not be attributed to the remote implementation.

A hash chain is not a signature or WORM storage. Externally trusted checkpoints, bounded corpus sizes and complete annotated-report ingestion remain separate future improvements; they are not claimed implemented by this compatible correction.

## Implementation plan

Apply the minimal compatible patch to the reviewed remote module and add `tests/test_studio_calibration_corpus_integrity.py`. Patch SHA-256: `dead6659d3fa3bb54c616ad210eae33751d6b7d3260d1d1df0b6d7aef18143c6`.

Use one writer per task/branch. Re-read current source before application if the branch advances. Run actual repository CI on Python 3.10, 3.11 and 3.12 before merge. Do not overwrite concurrent work or import reconstructed historical renders as original evidence.

## Test plan and observed evidence

Actual local environment: Python 3.13.5, a checksum-verified sparse corpus/calibration source subset rather than the full repository.

Before correction: **15 passed, 4 failed** (three new integrity regressions and one existing canonical-loader test).

After correction: **24 passed, 0 failed**: 9 existing corpus tests, 7 existing Perceptual Human Calibration tests and 8 integrity tests. The latter cover missing newline, input isolation, concurrent append, exact prefix preservation, injected fsync and replace failures, existing-lock preservation and unchanged record digest. Compileall passed. `git apply --check` and applying the patch in a clean sparse copy reproduced the tested bytes exactly.

Command: `python -m pytest tests -q --tb=short --junitxml=evidence/after-fix.xml`.

Full repository tests, target-version CI, audio renders and human listening were not performed for this correction. The earlier eight-passed prototype claim is not reproducible from its supplied ZIP, which contains a text result, report and seed but no test harness; it is not acceptance evidence.

## Risks and rollback

This run's attempted Python `create_file` was blocked by an OpenAI safety-status determination failure. Documentation and state writes succeeded. No alternate code-write route was attempted. The compatible correction therefore remains a local patch, not a merged feature.

Atomic replacement assumes a supported local filesystem and cooperating writers. Adversarial writes, network filesystems and directory-entry durability after sudden power loss are outside the guarantee. On validation/pre-commit failure preserve the original corpus. No audio DSP, mastering, LIVE code, neural inference, paid services or audio baseline changed. Subjective audio acceptance still requires human listening.
