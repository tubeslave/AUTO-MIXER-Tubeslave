# ADR: Markdown Memory Pipeline

## Context

Automixer now has multiple agents and automation loops that need the same
startup rules, safety boundaries, project goal, and report flow. Existing
Council folders still hold briefs, proposals, reviews, and project memory, but
agents need a smaller first-read layer that is easy to lint and keep safe.

## Options considered

1. Keep only the existing Council folders and rely on humans to find the right
   documents.
2. Put all operational memory in one large file.
3. Add a structured Markdown memory layer under `.ai/` with lightweight append
   tools and a lint check.

## Decision

Use option 3. Keep Agent Core, Project Layer, and Pipeline files under `.ai/`,
document the flow in `Docs/memory_pipeline.md`, and verify required safety
wording with `tools/memory/memory_lint.py`.

## Why this won

It gives Codex, Kimi, Paperclip-facing automation, and humans a stable first
read without granting runtime authority. The memory layer is documentation and
append-only helper tooling; it does not control WING, OSC, Dante, or live audio
runtime state.

## Rejected alternatives

- Single-file memory: rejected because safety rules, current tasks, decisions,
  experiments, and workflow templates would become harder to review.
- Runtime-integrated memory writes: rejected because report automation must not
  become a mixer-control path.
- External Paperclip workspace as source of truth: rejected because Automixer
  project memory should remain repo-local.

## Implementation plan

- Add `.ai/agent-core/` for shared rules, workflow, safety, source policy, and
  lint policy.
- Add `.ai/project-layer/` for Automixer goal, architecture, WING/OSC policy,
  gain-staging method memory, live safety, tasks, decisions, experiments,
  session reports, and glossary.
- Add `.ai/pipeline/` for capture/classify/propose/approve/apply/log/lint,
  Codex and Paperclip cycles, and append templates.
- Add `tools/memory/` append and lint helpers.
- Link the memory pipeline from `.ai/README.md`, `.ai/memory/project.md`, and
  `Docs/INDEX.md`.

## Test plan

- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python -m pytest tests/test_memory_pipeline.py -q`
- Standard project test gate when committing broader project changes.

## Risks and rollback

Risk: agents may over-treat memory notes as permission to change live mixer
state.

Rollback: revert the memory pipeline files and keep only the existing Council
folders. Safety policy remains enforced by `AGENTS.md`, `CLAUDE.md`, and the
supervised live-write gate.
