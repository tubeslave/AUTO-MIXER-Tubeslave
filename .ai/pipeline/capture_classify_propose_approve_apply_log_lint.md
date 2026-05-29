# CAPTURE CLASSIFY PROPOSE APPROVE APPLY LOG LINT

## CAPTURE

Capture raw facts:

- task request;
- current git status;
- relevant files;
- test output;
- report links;
- safety constraints;
- observations from soundcheck or offline runs.

## CLASSIFY

Classify each item as:

- task;
- decision;
- experiment;
- safety rule;
- report;
- glossary term;
- ADR candidate;
- stale or rejected evidence.

## PROPOSE

Create the smallest memory update that preserves the fact. Include source path,
mode, risk, and test evidence.

## APPROVE

Use the right approval level:

- append-only session report: agent may apply after safe work;
- project-layer decision: agent may append if scoped and evidence-based;
- architecture or safety change: use ADR or human approval;
- live-write authority: requires explicit operator approval for the exact
  action and must not be granted by memory alone.

## APPLY

Append or create Markdown records. Do not delete old records during normal
cycles.

## LOG

Log the result in `.ai/project-layer/session_reports.md` and, when useful,
`.paperclip/reports/`.

## LINT

Run:

```bash
python3 tools/memory/memory_lint.py
```

Fix missing files, sections, or safety wording before relying on the memory.
