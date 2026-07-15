# Codex proposal: M121 macOS release evidence hardening

Status: superseded in its npm-entrypoint section by the accepted ADR after a
real `script-shell` zero-execution false success was reproduced. The notary
direction remains the proposal input.

## 1. Short thesis

Keep the existing strict child-build gate, add a narrow npm lifecycle boundary
in front of it, and replace opaque app notarization with an explicit,
machine-verifiable app/DMG notary evidence chain.

## 2. What I understood

The defect is not missing documentation. The supported command cannot reach the
release builder under the pinned npm, while the current successful notarization
path cannot prove which Apple submission and accepted log belong to the shipped
app. Both are required-before-95 release evidence gaps, but neither grants WING
authority or increases the strict score by itself.

## 3. Proposed solution

### npm entrypoint

Add `scripts/run-macos-release.mjs` as the `mac:release` package entrypoint.
It validates the exact npm 10.9.8 lifecycle identity, permits only the known
standard lower-case lifecycle keys, rejects every unknown or upper-case npm
configuration key, scrubs all npm lifecycle/config/package metadata, and starts
`build-macos.mjs --profile release` with the current Node executable. The
existing `validateReleaseEnvironment()` continues to reject every npm config
variable as a second boundary.

### notarization evidence

Disable electron-builder's opaque `mac.notarize` step and use an `afterSign`
hook that runs a repository-owned app notarizer. Both app and DMG notarizers:

- create a private submission archive/snapshot;
- submit with keychain-profile credentials and wait for JSON status;
- fetch the Apple log even when Accepted;
- require accepted status, matching UUID and matching submitted archive SHA;
- staple and validate the final artifact;
- retain a no-follow log sidecar and v2 evidence document.

The verifier requires stable, contained app/DMG evidence and log files. Its
portable release manifest records both submission identities, log identities,
submitted hashes, final artifact identities and stapling/Gatekeeper results.
Initial and retained passes must agree exactly.

## 4. Likely files to touch

- `frontend/package.json`
- `frontend/scripts/run-macos-release.mjs`
- `frontend/scripts/build-macos.mjs`
- `frontend/electron-builder.release.cjs`
- `frontend/scripts/after-sign-macos.cjs`
- `frontend/scripts/notarytool-evidence.mjs`
- `frontend/scripts/notarize-app.mjs`
- `frontend/scripts/notarize-dmg.mjs`
- `frontend/scripts/verify-macos-release.mjs`
- focused macOS build/signing/verifier tests
- M121 plan and ADR

## 5. Alternatives considered

- Allow all npm-injected variables in the core gate: rejected because it makes
  the trusted boundary depend on mutable npm configuration.
- Tell operators to invoke Node directly: rejected because the documented
  package command remains broken.
- Correlate the latest `notarytool history` entry after electron-builder:
  rejected because concurrent submissions make that identity ambiguous.
- Parse debug/stdout from `@electron/notarize`: rejected as an unstable API and
  incomplete success evidence.

## 6. Risks

- App notarization ownership changes packaging order. The after-sign hook must
  finish stapling before electron-builder creates ZIP/DMG.
- Apple log formats may omit optional fields. Required binding fields must be
  documented and fail closed rather than guessed.
- Schema v2 can break existing test fixtures. Legacy v1 evidence may remain
  inspectable only under the existing explicit legacy option, never eligible
  for a new release.

## 7. Test plan

Run exact fixture and mutation tests for the wrapper, notarizers and verifier;
then the frontend test suite and production build. A real notarization run is
separately gated by Developer ID/keychain credentials and is not simulated as
completed.

## 8. Where the other agent may disagree

An independent reviewer may prefer a direct Node-only release command or DMG-
only log evidence to minimize scope. That would leave either the supported npm
entrypoint or the shipped app's Apple submission identity unproved, so the
proposal intentionally owns both boundaries.
