# WING Real Hardware Readiness

Generated: `2026-07-01T22:17:48Z`
Last updated: `2026-07-01T22:53:52Z`

## Status

- Overall status: `blocked_before_live_apply`
- Real WING reachable: `true`
- Read-only OSC diagnostic: `passed`
- Latest exact read-only preflight candidate: `passed`
- Manual operator review decision: `missing`
- Backend WebSocket listener at `127.0.0.1:8765`: `not_running_when_checked`
- Live authority granted: `false`

The project is ready to continue read-only real hardware validation against the
connected WING Rack. It is not ready for live apply. No fader, gain, EQ,
dynamics, gate, pan, FX, routing, scene, master, DCA, mute, or matrix write is
authorized by this report.

## Real Hardware Discovery

Direct read-only OSC discovery reached the console at `192.168.1.102`.

- discovery response: `WING,192.168.1.102,ATMOSPHERA-WNG,wing-rack,0100Q1V0604AAE,3.0.5-4-ga4ea833d:release`
- model: `wing-rack`
- firmware: `3.0.5-4-ga4ea833d:release`
- OSC send port: `2222`
- OSC receive/query port: `2223`
- direct verification read: `/ch/1/fdr`
- direct verification value: `["-oo", 0.0, -144.0]`
- scanned state key count: `36`

This direct diagnostic only proves the console is reachable and read-only OSC
queries can return values. It is not the readiness closure path.

## Latest Exact Preflight Candidate

The latest exact candidate reviewed in this run is:

`artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json`

- status: `passed`
- captured_at: `2026-07-01T22:01:46.878Z`
- connection_mode: `wing`
- contacts_backend: `true`
- queries_console: `true`
- read_only: `true`
- live_write: `false`
- mutates_console_state: `false`
- write_request_sent: `false`
- proposal_request_sent: `false`
- forbidden_request_types_sent: `[]`
- sent_request_types: `["get_operator_mode", "get_state", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter"]`
- operator_confirmed_channels: `[3]`
- operator_confirmed_session_ready: `true`
- readback_count: `3`
- physical_readback_count: `3`

Physical readbacks in that artifact:

- `/ch/3/fdr`, value type `fader`, value `-144.0`
- `/ch/3/in/set/trim`, value type `gain`, value `0.0`
- `/ch/3/pan`, value type `pan`, value `0.0`

The exact artifact review report is:

`Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md`

It reports `candidate_ready_for_manual_review` and still requires operator
review before any readiness update.

## Decision Gate

The decision review report is:

`Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md`

It reports `blocked` with:

- ledger_status: `operator_review_required`
- decision_record_count: `0`
- unreviewed_candidate_count: `1`
- manual_readiness_update_candidate_available: `false`
- blocker: `operator_must_record_readonly_preflight_review_decision`

The generated pending template is:

`artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json`

The operator must manually review `summary.json` and `events.jsonl`, then replace
the pending fields in the template. A generated pending template is not an
accepted decision.

## Artifact Root State

The default root `artifacts/wing_readonly_preflight` now contains only the
latest valid preflight candidate kept for this commit. Earlier failed generated
preflight attempts were removed from the commit set because the release gate
must not treat mixed valid/invalid root contents as readiness evidence.

- valid_candidate_count: `1`
- invalid_candidate_count: `0`
- latest release check artifact:
  `artifacts/no_wing_release_check/20260701T225352Z_no_wing_release_check/summary.json`
- quick release check: `passed`
- review ledger status: `operator_review_required`

Do not weaken this policy. If failed preflight attempts are kept for forensics,
archive them outside the default evidence root or review the current candidate
by exact artifact path.

## Commands For The Next Operator Step

Review the exact candidate:

```bash
PYTHONPATH=backend python3 tools/review_wing_readonly_preflight_artifacts.py \
  --artifact-path artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json \
  --output Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md \
  --decision-template-output artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json
```

After manual operator edits to the decision template, review decisions:

```bash
PYTHONPATH=backend python3 tools/review_wing_readonly_preflight_decisions.py \
  --artifact-path artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json \
  --decision-root artifacts/wing_readonly_preflight_review_decisions \
  --output Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md
```

Run the no-WING quick release check as a fail-closed gate. After removing the
failed generated preflight attempts from the default root, the quick gate passes
while still granting no live authority:

```bash
PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick
```

## Test Results

Targeted tests passed:

```bash
PYTHONPATH=backend python3 -m pytest tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_runner.py tests/test_no_wing_release_check.py -q
```

Result: `563 passed in 6.02s`.

Additional targeted check passed:

```bash
PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_operator_analysis_records_shadow_gain_ride_traces_without_queue_mutation -q
```

Result: `1 passed in 6.60s`.

Standard full-suite command was attempted:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

After fixing two order-dependent test failures, the full command progressed into
the smart-DSP replay evaluation block but was terminated by the tool/session
environment with signal `143` before pytest could print a final summary. A
diagnostic `-vv` run showed the same behavior around the slow
`tests/test_smart_dsp_replay_evaluation.py` block. Detached attempts left
orphan pytest shards, which were stopped so no test process remained running.

Therefore the relevant WING/read-only and release-check tests are green, but
this report does not claim a completed full-suite standard verification.

## Boundary

Allowed now:

- read-only WING discovery
- read-only backend state checks
- read-only `query_mixer_parameter`
- artifact review
- operator decision recording

Still blocked:

- any live write
- any live authority widening
- any readiness closure from mixed artifacts
- any unattended live apply
- any EQ, dynamics, gate, de-esser, FX, routing, scene, master, matrix, DCA, or
  mute write without a separate verified mapping/readback/rollback path and an
  explicit operator allowlist
