# ADR: publish ML-SPATIAL-2026-09-11-155446 to the shared research base

## Context

The user explicitly requested transfer of the prepared spatial-instrument study into the main shared knowledge base. GitHub metadata confirms `tubeslave/AUTO-MIXER-Tubeslave` uses `master`, not `main`. The existing research directory is `Docs/mixing_learning_updates/`. The two supplied files have been read and their exact bytes verified against the uploaded Git blobs.

## Options considered

1. Leave the materials only as temporary chat exports.
2. Publish the exact research package with a discoverable index, hashes and provenance.
3. Immediately inject raw candidate records into the runtime SourceKnowledgeStore.

## Decision

Choose option 2. Publish the original Markdown and JSON, an indexed publication manifest and a documentation link. Keep the evidence intact. This decision accepts publication only, not the scientific validity of every hypothesis or any production DSP change.

## Why this won

The user gets a durable, shared, versioned copy accessible from the main branch without losing limitations, categorical confidence, source-reading depth, experiment status or prior provenance. No unrelated data or production files need to be replaced.

## Rejected alternatives

Temporary exports do not satisfy the requested shared publication. Immediate runtime insertion is unsafe because the raw research schema differs from SourceRule, categorical confidence cannot be silently cast to a measured numeric probability, and complete cross-registry semantic deduplication has not been performed.

## Implementation plan

Add the byte-preserved report and structured package under their stable update ID; add publication metadata and a shared index; link the research index from `Docs/source_grounded_learning.md`. Publish through a pull request targeting `master`. Do not enable source_knowledge, auto-apply or any audio operation.

## Test plan

Nineteen local artifact checks passed: IDs/counts, source and rule references, candidate and auto-apply flags, required bounded-rule fields, queued-video evidence restrictions, unrun experiment status, source-reading limits, analytical-example status and equality of both uploaded Git blob hashes to the local files. These are not audio tests or a full repository test run. A local clone was attempted but container DNS could not resolve GitHub. Inspect the PR checks and final changed-file list before merge; verify the publication manifest and both blob SHAs from master after merge.

## Risks and rollback

Original persistence fields describe the pre-publication study; the README and manifest explicitly distinguish them from repository publication. Cross-registry integration remains deferred and must not be claimed complete. Other-chat delivery or agent ingestion is unconfirmed. All eight rules remain candidate/auto_apply:false; all four experiments are not_run. Roll back this documentation-only publication with a reviewed revert; do not rewrite branch history or modify runtime configuration.

Review scope: the publishing assistant performed an artifact/safety review. No independent Kimi or human audio review is claimed.
