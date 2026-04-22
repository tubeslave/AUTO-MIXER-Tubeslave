# Project Memory

## Stable Conventions

- Read `AGENTS.md` and `CLAUDE.md` before planning or editing.
- Use `Docs/adr/` for accepted architecture decisions.
- Default test gate: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- Keep diffs small and avoid new production dependencies without written justification.

## Safety-Critical Rules

- Prefer safer output changes over louder ones.
- Preserve true-peak headroom before increasing gain.
- Respect feedback protection and mixer-safety constraints from `CLAUDE.md`.
- Never change secrets or `.env` files as part of council work.

## Architecture Reminders

- `server.py` is a coordinator; keep logic in dedicated handlers or modules.
- `AudioCapture` is the shared audio service; do not duplicate capture pipelines casually.
- Repeated DSP or protocol rules belong in `CLAUDE.md` or `Docs/CONVENTIONS.md`, not only in task-specific notes.

## Council Lessons

- Separate planning, implementation, and review into different phases.
- The writing agent and reviewing agent should use different worktrees.
- Durable lessons should be generic enough to survive beyond a single task.
