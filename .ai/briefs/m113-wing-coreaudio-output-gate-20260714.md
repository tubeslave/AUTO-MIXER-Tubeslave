# M113 WING CoreAudio Output Gate

## Objective

Close a fail-open hardware-identity gap before any attended WING work: a
CoreAudio device may currently verify with 48 inputs but no 48-channel output,
although M001/M110 require one physical WING USB interface with exactly 48
inputs and 48 outputs at 48 kHz.

## Confirmed defect

- `verify_wing_capture_hardware_identity` checks manufacturer, USB transport,
  48 input channels and 48 kHz, but ignores `coreaudio_device_output`.
- A synthetic WING row with `coreaudio_device_output = 0` therefore verifies
  when its USB descriptor and serial binding are otherwise valid.
- The success binding records only input channels, so downstream evidence
  cannot prove that the output half of the 48x48 prerequisite was checked.

## Scope

1. Require exactly integer 48 for both CoreAudio input and output channel
   counts; missing, boolean, string and all other counts fail closed.
2. Include the verified output count in the canonical hardware binding and its
   SHA-256 digest.
3. Add adversarial tests for the accepted 48x48 case and rejected output
   variants.
4. Record the decision, validate focused/WING/full-suite contracts, commit the
   non-marker tree and then rebind CP05 in a marker-only commit.

## Hard boundaries

- No WING/backend/audio/OSC/network contact in this work item.
- No console writes, reset, soundcheck corrections or authority promotion.
- Do not alter USB/serial/location checks or widen any live capability.
- Preserve unrelated generated and untracked evidence.
- Strict live readiness remains 88/100 until attended physical proof passes.
