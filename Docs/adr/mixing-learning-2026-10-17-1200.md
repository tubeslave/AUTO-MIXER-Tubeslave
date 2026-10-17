# ADR: Treat FDverb as a version-pinned creative-effect candidate

- Status: accepted_for_research_workflow
- Date: 2026-10-17
- Update: [ML-2026-10-17-1200](../mixing_learning_updates/ML-2026-10-17-1200.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

The queued FDverb DAFx demo paper and official repository are openly available. The method is relevant to creative rock spatial processing, but the paper has no controlled listening study or mix-quality comparator. The paper describes v1.0, while v1.1.0 changes early-reflection behaviour and fixes several defects. The repository is AGPL-3.0-or-later, has pinned dependencies and release digests, but no discovered automated tests/CI; macOS release notes state the binaries are not notarized.

## Decision

Upgrade the source to full-text reviewed and add a pinned artifact audit. Keep all claims bounded to the algorithm and audited repository state.

For paper parity, use tag v1.0/commit `5357071c87b8bc5f2483ad8d32c5b3793d5a6c25`. For any practical experiment, pin v1.1.0/commit `f5ad84c081192e5e76774f12c3d4d484ade4c23f` and its exact asset/build digest. Do not treat the versions as independent evidence, and do not bypass operating-system security controls.

Add one blocked, one-factor A/B plan comparing only tail-noise correlation (fully correlated versus fully uncorrelated) with early reflections muted, raw measurement renders preserved and separate attenuation-only loudness-matched listening copies.

## Consequences

- One Source Card is upgraded, one Artifact Audit and ten Knowledge Cards are added.
- One bounded candidate rule is added with `auto_apply:false`.
- One A/B is added with `status:not_run_blocked` until safe build/provenance, host, PDC and project-audio gates pass.
- No plug-in is downloaded or run; no DSP, runtime, model, dependency or audio asset changes.
- The next executable experiment remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- Citation checking remains partial; absence of a contradiction or replication is not confirmation.
