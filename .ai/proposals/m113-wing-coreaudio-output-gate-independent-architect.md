# Independent Architect Proposal — M113

## 1. Short thesis

Require exact integer 48 input, exact integer 48 output and 48 kHz before a
CoreAudio device can verify as WING, and include both counts in the canonical
binding so later OS-identity drift revokes authority.

## 2. What I understood

The current verifier ignores `coreaudio_device_output`, so a synthetic 48x0
device passes despite the M001/M110 48x48 prerequisite. This repair narrows an
existing gate and does not grant live authority.

## 3. Proposed solution

- Add an exact output-channel check without coercion.
- Bind `output_channels: 48` into the success payload and digest.
- Preserve the aggregate blocked reason while exposing
  `checks.output_channels=false` and no authority identifiers.
- Update the attended runner's independent semantic verifier and manual
  binding fixture, not only the collector.
- Prove invalid output metadata and post-issue identity drift fail closed.
- Keep schema v1 only if the ADR records that no accepted physical HW01 v1
  evidence exists.

## 4. Likely files to touch

The hardware-identity module and tests, attended runner and tests, M110
architecture wording, M113 Council/plan/ADR records, canonical truth/ledger
surfaces, and finally the two CP05 markers.

## 5. Alternatives considered

Name/VID/PID-only identity, `output >= 48`, PortAudio-only validation, delaying
the check to HW01 and an immediate v2 schema were considered. They either
remain fail-open, weaken the exact contract, act too late or add migration cost
without accepted physical v1 evidence.

## 6. Risks

Some macOS versions may omit output metadata; the correct response is a
diagnostic fail-closed block. The new digest intentionally invalidates older
input-only identities. Partial fixture updates can create misleading green
tests, and any post-rebind edit invalidates CP05.

## 7. Test plan

Cover exact 48x48 success; missing, 0, 47, 49, boolean, string and float output;
blocked diagnostics and absence of authority identifiers; success binding;
independent artifact semantic rejection; hardware-refresh drift revocation;
focused route/runner tests, WING/static rails, full suite, no-WING verifier,
diff check and marker-only CP05 rebind.

## 8. Where the other agent may disagree

The main decision is v1 versus v2. Keep v1 because HW01 has not run and no
accepted physical artifact was found, but record that compatibility judgment.
Unit tests alone are insufficient unless the independent artifact verifier and
refresh/revocation composition are also covered.
