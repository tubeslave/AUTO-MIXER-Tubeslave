# Decisions

Status: control-plane decision index

## Active decisions

1. `Docs/` remains the canonical docs tree. The user request used `docs/`, but
   this checkout resolves `docs` to the existing `Docs` tree, so new durable
   documents are written under `Docs/` to avoid a parallel documentation system.
2. The long-term Project Goal is preserved in `Docs/PROJECT_GOAL.md`; current
   execution remains the supervised WING Rack MVP.
3. `ops/autopilot/STOP` is the manual stop switch. It must stop new autonomous
   development at the next safe checkpoint.
4. The new autopilot respects both `ops/autopilot/WRITER_LEASE/` and the
   existing `.codex_autoloop.lock`.
5. Thread automation uses a heartbeat attached to this thread when native
   automation support is available. If native Goal transition is unavailable in
   a wake-up, the automation must report the exact manual `/goal` command.
6. No product runtime behavior is changed by control-plane setup.
7. M001 is set to `WAITING_FOR_EXTERNAL_VALIDATION`, not `READY_FOR_NEXT`,
   because autonomous evidence proves the pre-hardware gate while the next
   proof requires real WING hardware and operator review.

## ADR relationship

Accepted architecture decisions continue to live in `Docs/adr/`. This file is
an operational index and does not replace ADRs required before production
dependency changes, live write expansion or autonomous authority changes.
