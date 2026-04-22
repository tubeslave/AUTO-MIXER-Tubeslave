# AI Council Protocol

Shared operating agreement for Codex.app, Codex CLI, Kimi CLI, and human contributors working on `AUTO-MIXER-Tubeslave-main`.

## Read First

- Read this file before planning or editing.
- Treat `CLAUDE.md` as the project canon for architecture, testing, and safety rules.
- Pull extra context from `Docs/CONVENTIONS.md`, `Docs/ARCHITECTURE.md`, and `Docs/TOOLS.md` when the task touches those areas.
- This repository controls live-sound behavior. When unsure, choose the safer action: reduce risk, preserve headroom, and avoid destructive mixer operations.

## Shared Safety Rules

- Never touch secrets, `.env`, tokens, or API keys.
- Never raise faders above `0 dBFS` without an explicit operator request.
- Preserve true-peak headroom and feedback safety checks before increasing gain.
- Prefer the smallest safe diff.
- Do not add production dependencies without a written reason in the proposal or ADR.
- In planning and review phases, do not edit production code.
- Before merge: run tests, review the diff, and record the decision in `Docs/adr/`.

## Standard Test Command

- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Council Roles

- `Codex.app` is the implementation workstation: isolated worktrees, parallel threads, focused patches, and code review.
- `Kimi CLI` is the independent architect and critic: alternate plans, design pressure-testing, and opposing reviews.
- Only one agent writes code for a task at a time.
- Do not let Codex and Kimi write inside the same worktree simultaneously.

## Shared Memory

- Task briefs live in `.ai/briefs/`
- Independent proposals live in `.ai/proposals/`
- Critiques and patch reviews live in `.ai/reviews/`
- Decision notes live in `.ai/decisions/`
- Durable project memory lives in `.ai/memory/project.md`
- Accepted architecture decisions live in `Docs/adr/`
- Tests and git history are part of project memory too

## Default Workflow

1. Create a brief in `.ai/briefs/<task-id>.md`.
2. Codex writes a proposal.
3. Kimi writes a proposal.
4. Codex critiques Kimi.
5. Kimi critiques Codex.
6. Synthesize the decision into `Docs/adr/<task-id>.md`.
7. One agent implements in its own worktree.
8. The other agent reviews the patch.
9. Durable lessons move into `.ai/memory/project.md` or this file.

## Proposal Format

1. Short thesis
2. What I understood
3. Proposed solution
4. Likely files to touch
5. Alternatives considered
6. Risks
7. Test plan
8. Where the other agent may disagree

## Critique Format

1. Strongest part of the other proposal
2. Possible mistakes or blind spots
3. Where the solution is too complex, fragile, or costly
4. What should be kept from the other proposal
5. My revised recommendation

## ADR Minimum

Use these sections in `Docs/adr/<task-id>.md`:

1. Context
2. Options considered
3. Decision
4. Why this won
5. Rejected alternatives
6. Implementation plan
7. Test plan
8. Risks and rollback

## Working Agreements

- Steelman first: restate the best version of the other agent's idea before criticizing it.
- Argue from files, tests, constraints, latency, safety, and maintainability.
- Keep output concrete: name files, commands, failure modes, and rollback options.
- If the repository state is dirty, do not overwrite unrelated local changes.
- If a task depends on uncommitted local work, say so explicitly in the proposal or review.
