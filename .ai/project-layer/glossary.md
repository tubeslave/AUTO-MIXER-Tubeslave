# Glossary

## Purpose

This glossary keeps Automixer terms consistent across agents, docs, tests, and
reports.

## Terms

Agent Core:
Shared rules for how an agent works before project-specific knowledge.

Project Layer:
Automixer-specific memory: goal, architecture, WING policy, gain staging,
safety, tasks, decisions, experiments, session reports, and glossary.

Pipeline:
The memory cycle: CAPTURE, CLASSIFY, PROPOSE, APPROVE, APPLY, LOG, LINT.

Audit-only:
Read and report only. No runtime mutation.

Dry-run:
Compute or preview an action without mutating live state.

Supervised-write:
Explicit operator-approved bounded mutation with pre-read, safety check,
readback, and rollback.

Live-write:
Actual console or live runtime mutation. Forbidden without explicit approval.

Recommendation:
A proposed action or explanation. It is not a command.

Control command:
A real instruction to mutate mixer, OSC, WING, Dante, or live audio state.

Pre-read:
Reading the current value before a supervised parameter change.

Readback:
Reading the value after a supervised write to verify outcome.

Rollback:
Documented way to restore the pre-read value or safer fallback.

Ayaic level planes:
Default balance targets for source and stem/group loudness. Safety overrides
the target.
