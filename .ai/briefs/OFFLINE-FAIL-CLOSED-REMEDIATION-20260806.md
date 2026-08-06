# Offline Fail-Closed Remediation — 2026-08-06

## Objective

Fix defects and reachable placeholders that can be corrected and verified
without a real WING or dLive console. Do not contact mixer/backend transports,
open audio streams, widen live authority, or treat mock evidence as physical
proof.

## Confirmed defects

- Unsupported AutoFOH action types were reported as control-state verified.
- Missing or failed PyAudio discovery returned plausible fake WING devices.
- Five registered automation status endpoints exposed unimplemented features
  without a typed unavailable/non-authoritative contract.
- The global handler registry silently overwrote duplicate message types.
- Standalone architect/evaluator agents returned placeholder success.
- MuQ genre mode added a constant fake genre score.
- Legacy EQ generated flattening proposals without measured spectral evidence.

## Boundaries

- Preserve all pre-existing staged, unstaged, and untracked work.
- No production dependency changes.
- No WING/dLive/backend network contact and no live write.
- Hardware, signing/notarization, calibrated SPL, and listening proof remain
  external blockers.

## Validation

- Focused regression tests for every changed behavior.
- Relevant server, soundcheck, handler, agent, and MuQ suites.
- Full repository pytest and frontend type/test/build checks.
- Canonical no-WING release check after the writer lease is released.
