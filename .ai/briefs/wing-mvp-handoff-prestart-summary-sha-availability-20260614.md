# Brief: WING MVP handoff prestart summary SHA availability

## Task

Make the WING read-only no-WING handoff validator fail closed when a packet
contains a prestart summary SHA256 but does not mark that digest as available.

## Context

The handoff packet now binds an existing no-WING prestart summary by SHA256.
The packet also exposes `backend_port_prestart_summary_sha256_available`, but
the no-WING handoff validator did not yet enforce that this flag matched the
digest field.

## Constraints

- Keep the change offline and read-only.
- Do not contact WING or the backend.
- Do not grant live authority.
- Preserve placeholder handoff behavior outside the no-WING generated packet
  path.

## Done

- no-WING handoff artifact validation checks SHA availability against the
  digest field.
- Summary JSON and Markdown render availability and match flags.
- Focused tests cover valid and mismatch cases.
