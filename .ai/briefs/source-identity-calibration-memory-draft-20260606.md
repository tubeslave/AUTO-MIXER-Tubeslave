# Brief: source-identity-calibration-memory-draft-20260606

## Context

Source identity calibration can now flow through preview, operator acceptance,
and manual persistence planning. The next safe step toward long-term learning is
to draft append-only memory records from persistence candidates without writing
project memory or changing runtime trust state.

## Objective

Add a transport-free source identity calibration memory draft that validates the
persistence plan and surfaces append-only draft records for future audited
memory updates.

## Safety Constraints

- Do not write `.ai/memory/project.md` or any other memory file.
- Do not mutate source identity models or downstream proposal priority.
- Do not import draft records into the decision queue.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Treat unsafe persistence-plan envelope, summary, or candidate claims as
  invalid.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend draft builder validates persistence-plan identity, status,
  timestamps, safety flags, summary counts, summary identity lists, and
  candidate false-authority flags.
- `main_pipeline_preview` exposes draft status and draft counts without mutating
  the proposal queue.
- The processing registry and project goal matrix register the draft as
  `shadow_only` memory-learning evidence, not a live executor.
- Focused memory-draft, Product Layer, registry, and capability tests pass.
