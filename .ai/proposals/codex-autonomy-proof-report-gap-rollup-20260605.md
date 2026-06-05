# Codex Proposal: autonomy-proof-report-gap-rollup-20260605

## Short thesis

Surface Full Autonomous readiness gaps at the top of the supervised proof
artifact report without changing any runtime authority.

## What I understood

The project is moving toward a four-mode autonomous mixing system. The current
mode gate can show that Analysis and Assisted are available, Supervised is
limited, and Full Autonomous remains blocked. The proof report should make the
reason for that block visible without requiring an operator to drill into every
nested row.

## Proposed solution

Add summary fields to `autonomy_mode_transition_gate` for Full Auto gap counts,
Full Auto gap target IDs, and the current Supervised live-write boundary. Then
add `summary.autonomy_gap_rollup` to the supervised proof artifact report,
sourced from the Auto mode gate row.

Extend `tools/no_wing_release_check.py` with a local JSON schema check after
the proof report command. The release runner should fail if the report no
longer exposes the read-only gap rollup or if the mode gate claims Auto can
apply autonomously.

The schema check should derive the expected Supervised live-write kinds from
`backend.operator_mode_policy.build_operator_mode_status()` so the release gate
tracks the canonical mode contract.

Finally, copy the schema check detail into no-WING `summary.json` and
`summary.md` so the release artifact is useful during human readiness review.

The rollup must state:

- it is read-only;
- it performs no live write;
- it does not widen live authority;
- Full Autonomous remains blocked;
- Supervised live apply remains limited to the current write kinds.

## Likely files to touch

- `backend/autonomy_mode_transition_gate.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `tools/no_wing_release_check.py`
- `backend/operator_mode_policy.py` (read-only dependency)
- `tests/test_autonomy_mode_transition_gate.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tests/test_no_wing_release_check.py`
- `Docs/OFFLINE_NO_WING_VALIDATION_RUNBOOK.md`

## Alternatives considered

One alternative is to leave the data only in nested rows. That preserves schema
minimalism, but makes the main proof report weaker as an operator-facing
readiness artifact. Another is to create a separate remediation module, which
would duplicate the mode gate before the need is proven.

## Risks

The main risk is making the report sound like Full Autonomous is nearer than it
is. The wording and safety flags should explicitly preserve the block and the
current supervised fader/gain boundary.

## Test plan

- Focused mode gate and report tests.
- no-WING release-check tests for the gap rollup artifact schema.
- no-WING summary tests for the JSON/Markdown rollup section.
- Python compile check for the edited backend/tooling files.

## Where the other agent may disagree

Kimi may prefer keeping all readiness detail nested under the mode gate. The
counterpoint is that the proof report summary is the artifact operators and
future automation review tools will read first.
