# Proposal: Expose handoff backend prestart command in summaries

## Short thesis

The no-WING handoff summary should show the packet-declared
`backend_port_prestart_check_command` so operators can audit the required
prestart command directly.

## What I understood

The WING read-only handoff already validates `backend_port_prestart_check_command`,
but the operator-facing handoff summary only exposed the extracted
`prestart_backend_port_check_command` from the command map. The rail text asks
for the packet-declared field specifically.

## Proposed solution

- Add `backend_port_prestart_check_command` to
  `wing_readonly_preflight_handoff_packet_artifact_summary`.
- Render it in the Markdown handoff summary next to
  `prestart_backend_port_check_command`.
- Extend tests so valid summaries show the expected command and failed summaries
  preserve a mismatched command.
- Add the report to WING MVP evidence.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Relying only on `prestart_backend_port_check_command` was rejected because it
does not prove the packet-declared field is visible. Renaming the existing field
was rejected because both views are useful: one is packet-declared, the other is
extracted from `commands`.

## Risks

The summary gains one more command field. That is acceptable because it closes a
specific Project Director visibility requirement.

## Test plan

- Focused handoff summary tests.
- Focused no-WING/Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may argue that one command field is enough. I recommend keeping both names
because the rail explicitly names `backend_port_prestart_check_command`.
