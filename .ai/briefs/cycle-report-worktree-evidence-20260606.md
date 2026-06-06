# Brief: cycle-report-worktree-evidence-20260606

## Task

Harden the Codex cycle report validator so repo-local reports explicitly record
the exact `git status --short` command and the current branch in `## What Was
Found`.

## Context

The autoloop prompt requires each cycle to gather `git status --short`, current
branch, changed files, and new files. The validator already checked broad
worktree and branch evidence, but it did not fail a report that only said
`git status` was inspected or left the branch value blank.

## Safety Boundary

This is report-tooling work only. It must not contact a console, open network
transport, send OSC/MIDI/TCP, mutate mixer state, or widen live-write authority.

## Expected Output

- Validator rejects missing exact `git status --short` evidence.
- Validator rejects empty current-branch evidence.
- Focused tests cover both failures.
- Cycle report records validation and live-safety status.
