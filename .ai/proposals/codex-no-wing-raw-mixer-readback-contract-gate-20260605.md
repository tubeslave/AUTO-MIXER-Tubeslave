# Proposal: no-wing-raw-mixer-readback-contract-gate-20260605

## 1. Short thesis

no-WING release artifacts should validate the raw mixer-state readback contract
as its own schema surface before any future physical artifact can be trusted.

## 2. What I understood

The repo now has a transport-free contract for the next physical readback target.
However, release summaries currently show fixture evidence and remediation plan
alignment without a dedicated contract schema result. A broken contract could
therefore be hidden behind passing fixture tests.

## 3. Proposed solution

Add `validate_raw_mixer_state_physical_readback_contract_snapshot()` to
`tools/no_wing_release_check.py`. Run it immediately after read-surface
remediation validation. The gate checks identity, safety flags, artifact field
lists, physical-only sources, forbidden mutation fields, and target consistency
against fixture/remediation summaries. Add a compact JSON/Markdown summary.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/no-wing-raw-mixer-readback-contract-gate-20260605.md`
- `Docs/adr/no-wing-raw-mixer-readback-contract-gate-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Keep contract validation only in unit tests. Rejected because release artifacts
  need self-checking proof metadata.
- Merge the contract into fixture evidence summary. Rejected because fixture
  coverage, remediation planning, and artifact schema readiness are separate
  evidence layers.
- Require a real physical artifact now. Rejected because this no-WING path must
  stay offline and hardware-free.

## 6. Risks

Operators may read contract readiness as completed proof. Mitigation: the
summary carries `physical_fixture_gap_remains=true`,
`accepted_as_physical_fixture_candidate=false`, `live_authority_granted=false`,
and `full_autonomous_ready=false`.

## 7. Test plan

- `PYTHONPATH=backend python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_raw_mixer_state_physical_readback_contract.py tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## 8. Where the other agent may disagree

Kimi may argue this is extra release metadata before hardware proof exists. I
would still add it because it prevents schema drift from becoming invisible when
the hardware proof runner is added later.
