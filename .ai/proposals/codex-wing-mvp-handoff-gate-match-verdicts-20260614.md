# Codex Proposal: WING MVP Handoff Gate Match Verdicts

## Short thesis

Turn backend and supervised prestart gate equality into explicit packet and
summary verdicts.

## What I understood

The active blocker remains `WING read-only connection proof`. The current
first command requires the no-WING handoff summary to prove that backend and
supervised prestart gates match their expected contracts before any supervised
proof can be planned. Equality existed in validation, but the packet did not
declare named match flags and no-WING did not render them as named verdicts.

## Proposed solution

Add `backend_port_prestart_gate_matches_expected` and
`supervised_prestart_gate_matches_expected` to
`tools/build_wing_readonly_preflight_handoff_packet.py`. Extend
`tools/no_wing_release_check.py` to recompute those verdicts, reject stale
declared flags, and expose both computed and declared values in the summary.
Update Project Director baseline markers and rail docs to require the fields.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- WING rail docs, focused tests, Council brief, ADR, report, and index files

## Alternatives considered

Leaving object equality only keeps the validator correct but weakens operator
inspection and machine checks. Renaming the generic Markdown
`actual_matches_expected` labels was avoided to preserve existing output while
adding clearer fields.

## Risks

The fields are redundant with object comparison. The redundancy is useful:
packet declarations are now explicit, and no-WING remains authoritative by
failing closed if the declared verdicts are stale.

## Test plan

Run focused handoff, no-WING, Project Director, and WING rail tests. Generate a
handoff packet, run Direction Guard, run no-WING quick fail-closed, run memory
lint, `git diff --check`, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer relying on object equality. The counterpoint is that named
verdicts are easier for operators, scripts, and Project Director baseline
coverage to audit before supervised proof planning.
