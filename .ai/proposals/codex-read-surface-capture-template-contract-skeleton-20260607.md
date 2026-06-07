# Proposal: Read-Surface Capture Template Contract Skeleton

## 1. Short thesis

Make the pending capture template self-describing by adding a contract-derived
payload skeleton, while keeping it invalid as proof until an operator fills and
reviews a real physical readback artifact.

## 2. What I understood

The repo is closing read-surface evidence gaps through no-WING, operator-facing
artifacts. The current capture template is safe, but it asks the operator to
replace a generic `physical_readback_payload` string without showing the exact
contract payload shape for the selected surface.

## 3. Proposed solution

- Reuse the existing physical readback contract builders.
- For each capture-template record, resolve its
  `required_contract_method_id`.
- Add `physical_readback_payload_template` with contract metadata,
  required field lists, one placeholder payload object, and nested safety flags.
- Keep `physical_readback_payload` as the operator-replaced artifact field.
- Add no-WING validation that checks the nested skeleton and rejects unsafe
  authority or candidate flags.
- Surface payload-template ids, artifact types, schema ids, and field counts in
  no-WING JSON and Markdown summaries.

## 4. Likely files to touch

- `tools/build_read_surface_physical_readback_capture_template.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_capture_template.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-capture-template-contract-skeleton-20260607.md`
- `Docs/adr/read-surface-capture-template-contract-skeleton-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_CAPTURE_TEMPLATE_CONTRACT_SKELETON_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Keep only the generic payload placeholder.
- Generate separate skeleton files per contract.
- Treat the skeleton as a candidate artifact.

## 6. Risks

The main risk is confusing the nested skeleton with a captured physical artifact.
The validator should preserve the outer pending placeholder, require nested
pending placeholders, and keep every authority, mutation, and fixture-candidate
flag false.

## 7. Test plan

- py_compile for the capture-template builder and no-WING runner.
- Focused capture-template builder tests.
- Focused no-WING artifact, unsafe payload, and summary tests.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest because the no-WING release summary contract changes.

## 8. Where the other agent may disagree

Kimi may prefer forcing operators to open the source contract file instead of
embedding a skeleton. The counterargument is that embedding a read-only,
fail-closed skeleton reduces operator transcription ambiguity without granting
proof status or live authority.
