# Codex Proposal — M113 WING CoreAudio Output Gate

## 1. Short thesis

Make the existing hardware-identity verifier fail closed on anything other
than an exact integer 48x48 CoreAudio WING at 48 kHz, and bind both channel
counts into the signed evidence surface.

## 2. What I understood

The attended M110 route requires bidirectional WING USB audio. The current
verifier proves 48 inputs but silently accepts zero, missing or malformed
output-channel metadata. That lets HW01 cross a prerequisite that the roadmap
and operator runbook explicitly define as 48 inputs and 48 outputs.

## 3. Proposed solution

- Add an explicit `output_channels` predicate beside the existing input check.
- Rely on exact equality to integer 48, which naturally rejects booleans,
  strings, missing values and every non-48 count.
- Add `output_channels: 48` to the canonical `coreaudio` binding so its digest
  and all downstream artifacts attest to the complete 48x48 identity.
- Add parameterized negative coverage for missing, 0, 47, 49, `true` and
  `"48"`, plus explicit success assertions for both bound counts.
- Correct the M110 ADR description from 48-input-only to 48x48.

## 4. Likely files to touch

`backend/wing_capture_hardware_identity.py`,
`tests/test_wing_capture_hardware_identity.py`, the M113 Council records,
`Docs/adr/m113-wing-coreaudio-output-gate-20260714.md`,
`Docs/plans/M113-wing-coreaudio-output-gate-20260714.md`, and the existing M110
ADR sentence that describes the verifier. CP05 marker files are touched only
after the non-marker commit.

## 5. Alternatives considered

- Checking only for a positive output count still accepts unsupported modes.
- Trusting the product name or USB descriptor does not prove the active
  CoreAudio mode is 48x48.
- Adding a second collector or changing the schema version is unnecessary;
  this is a stricter validation of an already required field.

## 6. Risks

Older evidence hashes change because the successful binding gains one field;
that is intentional and prevents reuse of input-only authority. Platform
metadata types must remain fail closed rather than coerced. Any non-marker edit
after CP05 rebind invalidates the final tree binding.

## 7. Test plan

Run the focused hardware-identity tests, dependent M110 route/runner tests,
WING/static rails, standard full suite, JSON/JSONL and diff checks, then verify
CP05 after a separate marker-only rebind commit. No live transport is opened.

## 8. Where the other agent may disagree

Whether to bump the v1 schema or invent a distinct failure reason. Keeping v1
and the existing aggregate `coreaudio_capture_not_physical_wing` reason is the
smaller compatible change; the detailed `output_channels: false` check makes
the rejection auditable.
