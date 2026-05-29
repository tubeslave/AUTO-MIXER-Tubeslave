# Agent Core Lint Policy

## Purpose

Memory lint prevents the Markdown memory layer from drifting into incomplete or
unsafe instructions.

## Required Checks

`tools/memory/memory_lint.py` checks:

- required Agent Core files exist;
- required Project Layer files exist;
- required Pipeline files and templates exist;
- key files have mandatory Markdown headings;
- safety files explicitly forbid live OSC/WING writes without approval;
- audit-only, dry-run, supervised-write, and live-write modes are separated.

## Failure Policy

If memory lint fails, fix the memory files before relying on them for a new
agent cycle. For production changes, treat a failed memory lint as a process
failure, not as proof that mixer code is broken.

## Command

```bash
python3 tools/memory/memory_lint.py
```
