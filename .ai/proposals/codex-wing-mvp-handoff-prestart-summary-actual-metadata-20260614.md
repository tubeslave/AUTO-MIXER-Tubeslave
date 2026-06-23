# Proposal: handoff prestart summary actual metadata

## Short thesis

Read actual no-WING prestart summary metadata in the handoff packet builder so
failed or environment-blocked summaries cannot be visually flattened into the
expected passed contract.

## What I understood

The active WING MVP blocker remains `WING read-only connection proof`. The
handoff packet is still offline, but it is operator-facing evidence before a
real WING read-only preflight. If it references a real prestart summary file,
the metadata shown in the packet should be the file's actual metadata.

## Proposed solution

- Load an existing JSON object from `--prestart-summary` when present.
- Use actual `type`, `status`, `connection_policy`, safety flags, backend check
  status, and backend detail flags from the file.
- Preserve `environment_blocked_checks` and
  `environment_blocked_check_details` in the packet and markdown.
- Keep default placeholder behavior unchanged when no file is present.
- Cover valid and environment-blocked summary cases in handoff packet tests.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `.ai/briefs/wing-mvp-handoff-prestart-summary-actual-metadata-20260614.md`
- `.ai/proposals/codex-wing-mvp-handoff-prestart-summary-actual-metadata-20260614.md`
- `Docs/adr/wing-mvp-handoff-prestart-summary-actual-metadata-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_ACTUAL_METADATA_20260614.md`

## Alternatives considered

- Keep expected constants only: rejected because it can make a failed referenced
  summary look passed in the handoff packet.
- Fail the builder immediately on failed summary metadata: rejected because the
  packet builder is an offline report generator; downstream gates already fail
  closed on mismatched status.
- Parse only the status field: rejected because policy and safety flags matter
  to the prestart contract.

## Risks

Missing or malformed real summary files now produce sparse actual metadata
when they are present but invalid. That is acceptable because downstream gates
should reject mismatches instead of silently substituting expected values.

## Test plan

- Focused handoff packet pytest.
- Focused no-WING/Project Director/WING MVP rail pytest.
- `py_compile`, memory lint, diff check, and full standard suite.

## Where the other agent may disagree

Kimi may prefer the builder to hard-fail on invalid metadata. The current
choice keeps the builder report-only while making invalid metadata explicit for
the existing fail-closed validators.
