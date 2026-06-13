# Proposal: Expose command artifact binding in WING handoff

## Short Thesis

The no-WING handoff summary should fail closed and clearly show that capture and
review commands point to the same expected plan and artifact roots as the packet.

## What I Understood

The first MVP blocker remains `WING read-only connection proof`. Offline
handoff artifacts must not close readiness, but they must be strong enough that
an operator can safely follow them for a real read-only capture. The packet
already validates expected artifact globs; the command-level bindings should be
visible too.

## Proposed Solution

- Extract capture command `--expected-plan` and `--artifact-root` values.
- Extract review command `--artifact-root`.
- Extract decision-review command `--artifact-root` and `--decision-root`.
- Fail validation if those values do not match packet fields.
- Include the fields in JSON and Markdown no-WING summaries.
- Update the WING rails/docs so Project Director keeps the requirement visible.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`

## Alternatives Considered

Relying only on whole-command equality was rejected because it is not obvious in
the operator summary. Adding runtime WING checks was rejected because this slice
must remain no-WING and read-only.

## Risks

Stale or hand-edited packets with mismatched command paths will fail. That is
intended; the packet should be regenerated before a hardware session.

## Test Plan

- Compile touched Python.
- Focused no-WING handoff tests.
- no-WING quick release check.
- Project Director/WING rail tests.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may argue whole-command equality already covers this. I recommend exposing
the binding anyway because the operator summary is the artifact used during the
real handoff.
