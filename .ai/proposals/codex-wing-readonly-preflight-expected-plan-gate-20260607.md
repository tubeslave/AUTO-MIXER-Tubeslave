# Proposal: WING Read-Only Preflight Expected-Plan Gate - 2026-06-07

## 1. Short thesis

Require real read-only WING preflight capture to match the offline plan before
WebSocket opens, and record a plan hash in the proof artifact.

## 2. What I understood

The current MVP needs one clean supervised WING proof path, starting with
read-only connection and physical readback evidence. The offline plan preview is
already part of the runbook; this change makes that preview enforceable for the
actual capture run.

## 3. Proposed solution

- Add `--expected-plan` to `tools/wing_readonly_preflight_runner.py`.
- Load the JSON plan, compute SHA256, and compare runtime-relevant fields:
  target URL, channels, operator confirmation, value types, artifact root,
  request sequence, request-id policy, fail-closed conditions, and review
  commands.
- Fail closed before WebSocket if the plan is missing, invalid, or mismatched.
- Add expected-plan fields to summary artifacts and evidence validation.
- Update handoff packet/no-WING/runbook commands to pass the same plan file.

## 4. Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- focused WING preflight tests
- AI Council brief/proposal/ADR/report and `Docs/INDEX.md`

## 5. Alternatives considered

- Trust runbook ordering. Rejected because the artifact would not prove the
  real run matched the preview.
- Compare only channels and value types. Rejected because URL, artifact root,
  request ids, and fail-closed conditions are also part of the proof contract.
- Make `--expected-plan` mandatory at argparse level. Rejected for backward CLI
  compatibility; evidence intake and no-WING gates still require it for a clean
  proof candidate.

## 6. Risks

The stricter evidence gate will reject older artifacts that lack a plan hash.
That is acceptable because the first real WING proof is still open.

## 7. Test plan

- Compile changed modules.
- Run focused WING preflight/no-WING/handoff tests.
- Run no-WING quick and full pytest before commit.

## 8. Where the other agent may disagree

Kimi may prefer making `--expected-plan` mandatory for every non-plan run. I
would keep the CLI compatible but make proof validation require the plan link.
